"""Portable sequential CE1 training and evaluation, starting from full FM 1M."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import torch
from .common import write_json


def train_flags(args, target):
    return ['--output',str(args.output/'training'), '--kind','gaussian_ce',
            '--data-root',str(args.data_root), '--steps',str(target),
            '--batch-size','128', '--sigma','0.5', '--center-scale','12',
            '--lr','0.0002', '--min-lr','0.0002', '--warmup','5000',
            '--ema-decay','0.9999', '--amp','none', '--ce-weight','0.001',
            '--ce-every','4', '--ce-steps','1', '--ce-batch','64',
            '--ce-logits','center', '--ce-solver','euler', '--ce-warmup','2000',
            '--ce-crop-pad','4', '--ce-crop-fraction','1', '--ce-crop-seed','20260918',
            '--ce-paired-views', '--ce-paired-crop-both', '--ce-js-weight','0.001',
            '--ce-js-temperature','2', '--ce-js-warmup','5000',
            '--grad-diagnostics-every','500', '--seed','0', '--save-every','5000',
            '--snapshot-every',str(args.eval_every), '--keep-previous-checkpoint',
            '--preview-every','0', '--resume', '--finetune-from',str(args.source),
            '--finetune-init','full', '--finetune-start','1000000',
            '--finetune-steps',str(args.added_steps), '--allow-constant-finetune-extension']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Pure FM 1M full checkpoint')
    p.add_argument('--resume-from', type=Path, help='Import a complete CE1 state into a NEW output directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('./data'))
    p.add_argument('--added-steps', type=int, default=500000)
    p.add_argument('--eval-every', type=int, default=100000)
    p.add_argument('--full-eval-every', type=int, default=100000)
    p.add_argument('--cache', type=Path, default=Path('./runs/cifar10-reference-cache'))
    p.add_argument('--classifier', type=Path, default=Path('./weights/cifar10-reference-classifier.pt'))
    p.add_argument('--train-only', action='store_true')
    args = p.parse_args()
    if args.added_steps < 2 or min(args.eval_every,args.full_eval_every) < 1:
        p.error('Invalid schedule')
    for key in ['source','output','data_root','cache','classifier','resume_from']:
        value = getattr(args,key)
        if value is not None:
            setattr(args,key,value.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    training = args.output/'training'
    training.mkdir(exist_ok=True)
    latest = training/'training-latest.pt'
    if args.resume_from:
        if latest.exists():
            p.error('Output already has a checkpoint; omit --resume-from to continue it')
        checkpoint = torch.load(args.resume_from,map_location='cpu')
        required = {'model','ema','optimizer','rng_cpu','rng_cuda','rng_numpy','ce_crop_rng_state'}
        if not required.issubset(checkpoint) or checkpoint['config'].get('ce_steps') != 1:
            p.error('--resume-from must be a full CE1 training state, not EMA-only')
        del checkpoint
        shutil.copy2(args.resume_from,latest)
    start = 1000000
    if latest.exists():
        checkpoint = torch.load(latest,map_location='cpu')
        start = checkpoint['step']
        del checkpoint
    target = 1000000+args.added_steps
    if start > target:
        p.error('Checkpoint is beyond the requested target')
    nodes = sorted(set([target] + list(range(1000000+args.eval_every,target+1,args.eval_every))
                       + list(range(1000000+args.full_eval_every,target+1,args.full_eval_every))))
    logs = args.output/'logs'
    logs.mkdir(exist_ok=True)
    write_json(args.output/'plan.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})

    def run(name,module,flags):
        command = [sys.executable,'-u','-m','cifar10_ce1.'+module] + [str(x) for x in flags]
        write_json(logs/(name+'-command.json'),{'command':command})
        write_json(args.output/'status.json',{'status':'running','phase':name,'pid':os.getpid()})
        print(name,flush=True)
        with (logs/(name+'.log')).open('a') as stream:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)

    try:
        for node in nodes:
            if node < start:
                continue
            if shutil.disk_usage(args.output).free < 15*1024**3:
                raise RuntimeError('Less than 15 GiB free disk space')
            if node > start:
                run('train-to-%d'%node,'train',train_flags(args,node))
                start = node
            snapshot = training/('model-%d.pt'%node)
            # Also supports an imported full checkpoint exactly at an evaluation node.
            if not snapshot.exists():
                checkpoint = torch.load(latest,map_location='cpu')
                if checkpoint['step'] != node:
                    raise RuntimeError('Missing checkpoint for requested evaluation')
                torch.save({k:checkpoint[k] for k in ['step','config','ema']},snapshot)
                del checkpoint
            terminal = training/('training-%d.pt'%node)
            if not terminal.exists():
                os.link(latest,terminal)
            if args.train_only:
                continue
            for solver in [1,4]:
                dest = args.output/('evaluations/step-%d-test-euler%d'%(node,solver))
                if not (dest/'metrics.json').exists():
                    run('classify-%d-euler%d'%(node,solver),'evaluate_reverse_test',[
                        '--checkpoint',snapshot,'--output',dest,'--data-root',args.data_root,
                        '--method','euler','--steps',solver])
            if (node-1000000)%args.full_eval_every == 0 or node == target:
                dest = args.output/('evaluations/ce1-%d-50k'%node)
                if not (dest/'metrics.json').exists():
                    if not args.classifier.exists():
                        raise FileNotFoundError('Download the reference scorer or pass --classifier for complete metrics')
                    run('fid50k-%d'%node,'evaluate',[
                        '--checkpoint',snapshot,'--output',dest,'--data-root',args.data_root,
                        '--cache',args.cache,'--classifier',args.classifier,'--samples',50000,
                        '--batch-size',64,'--method','heun','--steps',60,'--seed',20260912,
                        '--reverse-images',10000])
        write_json(args.output/'status.json',{'status':'complete','step':target})
    except BaseException:
        write_json(args.output/'status.json',{'status':'failed','error':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
