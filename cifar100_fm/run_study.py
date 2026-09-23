"""Sequential, restartable single-GPU training and full evaluation stages."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .common import sha256


def milestones(total, interval):
    return sorted(set(list(range(interval,total+1,interval)) + [total]))


def study_targets(total, interval, kind, origin, early_offsets):
    targets=milestones(total,interval)
    if kind=='gaussian_dual_pad4_js':
        targets += [origin+offset for offset in early_offsets if 0<offset<=total-origin]
    return sorted(set(t for t in targets if t>origin))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='./runs/cifar100-dct-fm')
    parser.add_argument('--data-root',default='./data')
    parser.add_argument('--kind',choices=['gaussian','gaussian_dual_pad4_js'],default='gaussian')
    parser.add_argument('--steps',type=int,default=5000000)
    parser.add_argument('--eval-every',type=int,default=100000)
    parser.add_argument('--finetune-from',default=None)
    parser.add_argument('--finetune-start',type=int,default=1000000)
    parser.add_argument('--early-eval-offsets',type=int,nargs='*',default=[50000,100000])
    parser.add_argument('--min-free-gib',type=float,default=15.)
    args = parser.parse_args()
    if min(args.steps,args.eval_every)<1 or args.min_free_gib<0:
        raise ValueError('steps and interval must be positive')
    root = Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    training = root/'training'
    origin=args.finetune_start if args.finetune_from else 0
    if args.steps<=origin:
        raise ValueError('target is an absolute step and must exceed initialization step')
    if (training/'config.json').exists():
        config=json.loads((training/'config.json').read_text())
        if config['kind']!=args.kind:
            raise ValueError('existing output belongs to a different training objective')
        old_source=config.get('finetune_from')
        new_source=str(Path(args.finetune_from).resolve()) if args.finetune_from else None
        if old_source!=new_source or (old_source and config['finetune_start']!=origin):
            raise ValueError('existing output has a different initialization protocol')
    for target in study_targets(args.steps,args.eval_every,args.kind,origin,args.early_eval_offsets):
        free=shutil.disk_usage(root).free/(1024**3)
        if free<args.min_free_gib:
            raise RuntimeError(f'only {free:.1f} GiB free; retain checkpoints and free space before rerunning')
        checkpoint = training/f'model-{target}.pt'
        if not checkpoint.exists():
            command = [sys.executable,'-m','cifar100_fm.train','--output',str(training),
                       '--data-root',args.data_root,'--steps',str(target),
                       '--kind',args.kind,'--keep-previous-checkpoint']
            if args.finetune_from:
                command+=['--finetune-from',args.finetune_from,'--finetune-start',str(args.finetune_start)]
            if (training/'training-latest.pt').exists():
                command.append('--resume')
            subprocess.run(command,check=True)
        if not checkpoint.exists():
            raise RuntimeError(f'{checkpoint} missing; training may already be beyond this milestone')
        evaluation = root/'evaluations'/f'{args.kind}-{target}-50k'
        if (evaluation/'metrics.json').exists():
            protocol = json.loads((evaluation/'protocol.json').read_text())
            metrics = json.loads((evaluation/'metrics.json').read_text())
            if (protocol.get('checkpoint_sha256') != sha256(checkpoint)
                    or protocol.get('method') != 'heun' or protocol.get('steps') != 60
                    or protocol.get('batch_size') != 64 or protocol.get('seed') != 20260912
                    or protocol.get('dataset') != 'cifar100'
                    or metrics.get('checkpoint_step') != target
                    or metrics.get('kind') != args.kind
                    or metrics.get('generated_images') != 50000
                    or metrics.get('reverse_images') != 10000):
                raise RuntimeError('existing evaluation does not match the study protocol')
            print(f'verified completed stage {target}',flush=True)
            continue
        subprocess.run([sys.executable,'-m','cifar100_fm.evaluate',
            '--checkpoint',str(checkpoint),'--output',str(evaluation),
            '--data-root',args.data_root,'--cache',str(root/'reference-cache'),
            '--samples','50000','--reverse-images','10000','--batch-size','64',
            '--method','heun','--steps','60'],check=True)


if __name__ == '__main__':
    main()
