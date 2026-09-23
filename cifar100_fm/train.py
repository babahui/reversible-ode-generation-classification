"""CIFAR-100 pure FM or dual-pad4 reverse CE+JS, with full-state resume/forks."""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR100
from torchvision.utils import save_image

from .common import (build_model, build_prior, initial_noise, integrate,
                     seed_all, to_uint8, write_json, sha256, ARCH)
from .dual_view import auxiliary_loss, gradient_diagnostics


def atomic_save(data, path, keep_previous=False):
    path=Path(path)
    tmp=Path(str(path)+'.tmp')
    torch.save(data,tmp)
    if keep_previous and path.exists():
        previous=path.with_name(path.stem.replace('-latest','-previous')+path.suffix)
        previous_tmp=Path(str(previous)+'.tmp')
        if previous_tmp.exists():previous_tmp.unlink()
        os.link(path,previous_tmp)
        previous_tmp.replace(previous)
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=['gaussian','gaussian_dual_pad4_js'],default='gaussian')
    p.add_argument('--data-root',default='./data')
    p.add_argument('--steps',type=int,default=5000000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--sigma',type=float,default=.5)
    p.add_argument('--center-scale',type=float,default=12.)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--warmup',type=int,default=5000)
    p.add_argument('--ema-decay',type=float,default=.9999)
    p.add_argument('--amp',choices=['none','bf16'],default='none')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save-every',type=int,default=5000)
    p.add_argument('--snapshot-every',type=int,default=100000)
    p.add_argument('--keep-previous-checkpoint',action='store_true')
    p.add_argument('--preview-every',type=int,default=25000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--finetune-from',default=None,help='Trusted full FM state; never EMA-only')
    p.add_argument('--finetune-start',type=int,default=1000000)
    p.add_argument('--full-snapshot-steps',type=int,nargs='*',default=[1000000])
    p.add_argument('--ce-weight',type=float,default=.001)
    p.add_argument('--ce-every',type=int,default=4)
    p.add_argument('--ce-steps',type=int,default=4)
    p.add_argument('--ce-batch',type=int,default=64,help='Total views: 32 originals x 2')
    p.add_argument('--ce-solver',choices=['euler','heun'],default='euler')
    p.add_argument('--ce-warmup',type=int,default=2000)
    p.add_argument('--ce-crop-pad',type=int,default=4)
    p.add_argument('--ce-crop-seed',type=int,default=20260918)
    p.add_argument('--ce-js-weight',type=float,default=.001)
    p.add_argument('--ce-js-temperature',type=float,default=2.)
    p.add_argument('--ce-js-warmup',type=int,default=5000)
    p.add_argument('--grad-diagnostics-every',type=int,default=500)
    args=p.parse_args()
    if not (1 <= args.batch_size <= 50000) or min(args.steps,args.warmup,args.save_every,args.snapshot_every)<1 or args.preview_every<0:
        raise ValueError('invalid training settings')
    if not torch.cuda.is_available():
        raise RuntimeError('training requires CUDA; use CPU for unit tests/inference only')
    dual = args.kind == 'gaussian_dual_pad4_js'
    if dual and (args.ce_batch < 2 or args.ce_batch % 2 or args.ce_batch//2 > args.batch_size
                 or min(args.ce_every,args.ce_steps)<1 or args.ce_crop_pad!=4
                 or min(args.ce_weight,args.ce_js_weight,args.ce_warmup,args.ce_js_warmup,
                        args.grad_diagnostics_every)<0 or args.ce_js_temperature<=0):
        raise ValueError('invalid dual-pad4 settings; CE batch counts both views')
    if args.lr<=0 or not 0<=args.ema_decay<1:
        raise ValueError('invalid optimizer/EMA setting')
    if args.finetune_from:
        args.finetune_from=str(Path(args.finetune_from).resolve())
        if args.steps<=args.finetune_start:
            raise ValueError('fine-tuning target must exceed the source total step')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=dict(vars(args),architecture=ARCH,objective='linear_flow_matching',
                dataset='cifar100',num_classes=100,time_encoding='scalar',
                prior_version='rgb_dct6_energy_sorted_100_v1',
                pairing='independent',data_augmentation='horizontal_flip_only',
                source_commit='7c653857de4c25b979a740ef8b6a5d5ddb09b6c1')
    config.update(ce_logits='center',ce_paired_views=dual,ce_paired_crop_both=dual,
                  ce_crop_fraction=1.0 if dual else 0.0,reverse_ce_dropout=False,
                  reverse_ce_solver=args.ce_solver if dual else None,
                  ce_enabled=dual,lr_schedule='constant_finetune' if args.finetune_from else 'warmup_constant')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seed_all(args.seed)
    model=build_model(args.kind).cuda()
    ema=copy.deepcopy(model).eval()
    for param in ema.parameters():param.requires_grad_(False)
    prior=build_prior(config,'cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    crop_rng=torch.Generator(device='cuda').manual_seed(args.ce_crop_seed) if dual else None
    checkpoint_path=out/'training-latest.pt'
    start_step=0
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError('--resume requires training-latest.pt, not an EMA-only checkpoint')
    is_fork=bool(args.finetune_from and not checkpoint_path.exists())
    if args.resume or is_fork:
        source=Path(args.finetune_from) if is_fork else checkpoint_path
        checkpoint=torch.load(source,map_location='cpu',weights_only=False)
        required={'model','ema','optimizer','rng_cpu','rng_cuda','rng_numpy','config','step'}
        if not required <= checkpoint.keys():
            raise ValueError('full training state required; EMA-only initialization is not supported')
        old=checkpoint['config']
        mutable={'steps','resume','output','data_root','save_every','preview_every',
                 'snapshot_every','keep_previous_checkpoint','full_snapshot_steps'}
        if is_fork:
            if checkpoint['step']!=args.finetune_start or old['kind']!='gaussian':
                raise ValueError('fork requires pure FM at the specified source step')
            mutable |= {'kind','finetune_from','finetune_start','ce_paired_views','ce_paired_crop_both',
                        'ce_crop_fraction','ce_enabled','reverse_ce_dropout','reverse_ce_solver',
                        'lr_schedule','grad_diagnostics_every'}
            mutable |= {key for key in config if key.startswith('ce_')}
            config['finetune_source_sha256']=sha256(source)
            mutable.add('finetune_source_sha256')
        elif args.finetune_from:
            config['finetune_source_sha256']=old['finetune_source_sha256']
        # Older pure-FM checkpoints had no auxiliary/fork metadata.
        defaults={key:value for key,value in config.items()
                  if key.startswith('ce_') or key in {'finetune_from','finetune_start',
                  'grad_diagnostics_every','reverse_ce_dropout','reverse_ce_solver','lr_schedule'}}
        if not is_fork and old['kind'] != args.kind:
            raise ValueError('objective changes require an explicit new fork')
        for key,value in config.items():
            if key not in mutable and old.get(key,defaults.get(key))!=value:
                raise ValueError('resume config mismatch: '+key)
        model.load_state_dict(checkpoint['model']);ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step=checkpoint['step']
        torch.set_rng_state(checkpoint['rng_cpu'])
        torch.cuda.set_rng_state(checkpoint['rng_cuda'])
        np.random.set_state(checkpoint['rng_numpy'])
        if crop_rng is not None and not is_fork:
            if 'ce_crop_rng_state' not in checkpoint:
                raise ValueError('missing crop RNG for exact CE+JS resume')
            crop_rng.set_state(checkpoint['ce_crop_rng_state'])
        del checkpoint
    elif checkpoint_path.exists():
        raise RuntimeError('existing training run requires --resume')
    else:
        # Data, flips, noise and t use a common RNG independent of label embedding.
        seed_all(args.seed+12345)
    write_json(out/'config.json',config)
    if start_step>=args.steps:
        print('target already reached',flush=True);return
    dataset=CIFAR100(args.data_root,train=True,download=False)
    if len(dataset) != 50000 or len(dataset.classes) != 100:
        raise ValueError('unexpected CIFAR-100 data')
    data=torch.from_numpy(dataset.data).permute(0,3,1,2).contiguous().cuda()
    targets=torch.tensor(dataset.targets,device='cuda')
    batches_per_epoch=len(data)//args.batch_size
    cached_epoch=-1;permutation=None
    trained_seconds=0.
    log_path=out/'train.jsonl'
    model.train()
    run_start=time.monotonic()
    print(json.dumps({'starting_step':start_step,'target':args.steps,'parameters':sum(p.numel() for p in model.parameters()),'kind':args.kind}),flush=True)
    for index in range(start_step,args.steps):
        tic=time.monotonic()
        epoch=index//batches_per_epoch
        if epoch!=cached_epoch:
            permutation=torch.from_numpy(np.random.RandomState(args.seed+epoch).permutation(len(data))).cuda()
            cached_epoch=epoch
        offset=(index%batches_per_epoch)*args.batch_size
        indices=permutation[offset:offset+args.batch_size]
        images=data[indices].float().div(255).mul(2).sub(1)
        labels=targets[indices]
        flips=torch.rand(len(images),device='cuda')<.5
        images=torch.where(flips[:,None,None,None],images.flip(-1),images)
        lr=args.lr if args.finetune_from else args.lr*min((index+1)/args.warmup,1.)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        noise=initial_noise(labels,config,prior)
        t=torch.rand(len(images),device='cuda')
        tt=t[:,None,None,None]
        state=(1-tt)*noise+tt*images
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.amp=='bf16'):
            prediction=model(t,state,y=None)
            flow_loss=(prediction.float()-(images-noise)).square().mean()
        if not bool(flow_loss.isfinite()):raise RuntimeError('nonfinite flow loss')
        flow_loss.backward()
        aux_metrics={}
        ce_applied=dual and (index+1)%args.ce_every==0
        if ce_applied:
            diagnostic=bool(args.grad_diagnostics_every and (index+1)%args.grad_diagnostics_every==0)
            parameters=list(model.parameters())
            flow_grads=[p.grad.detach().clone() if p.grad is not None else None
                        for p in parameters] if diagnostic else None
            ce_term,js_term,aux_metrics=auxiliary_loss(model,prior,images,labels,crop_rng,args,index+1)
            js_grads=torch.autograd.grad(js_term,parameters,retain_graph=True,allow_unused=True) if diagnostic else None
            (ce_term+js_term).backward()
            if diagnostic:
                aux_metrics.update(gradient_diagnostics(parameters,flow_grads,js_grads))
            del flow_grads,js_grads,parameters,ce_term,js_term
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-args.ema_decay)
        step=index+1
        trained_seconds+=time.monotonic()-tic
        if step==start_step+1 or step%100==0 or step==args.steps:
            record={'step':step,'flow_loss':float(flow_loss.detach()),
                    'grad_norm_before_clip':float(norm),'lr':lr,'seconds_per_step':trained_seconds/(step-start_step),
                    'elapsed_seconds_this_invocation':time.monotonic()-run_start,
                    'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'ce_applied':ce_applied,**aux_metrics}
            with log_path.open('a') as f:f.write(json.dumps(record)+'\n')
            write_json(out/'progress.json',record)
            print(json.dumps(record),flush=True)
        if step%args.save_every==0 or step%args.snapshot_every==0 or step in args.full_snapshot_steps or step==args.steps:
            payload={'config':config,'step':step,'model':model.state_dict(),'ema':ema.state_dict(),
                     'optimizer':optimizer.state_dict(),'rng_cpu':torch.get_rng_state(),
                     'rng_cuda':torch.cuda.get_rng_state(),'rng_numpy':np.random.get_state()}
            if crop_rng is not None:payload['ce_crop_rng_state']=crop_rng.get_state()
            atomic_save(payload,checkpoint_path,args.keep_previous_checkpoint)
            if step in args.full_snapshot_steps:
                atomic_save(payload,out/f'training-{step}.pt')
            atomic_save({'config':config,'step':step,'ema':ema.state_dict()},out/'model-latest.pt')
            if step==args.steps or step%args.snapshot_every==0:
                atomic_save({'config':config,'step':step,'ema':ema.state_dict()},out/f'model-{step}.pt')
        if args.preview_every and step%args.preview_every==0:
            with torch.inference_mode():
                preview_labels=torch.arange(100,device='cuda').repeat_interleave(2)
                generator=torch.Generator(device='cuda').manual_seed(20260912)
                z=initial_noise(preview_labels,config,prior,generator)
                generated=torch.cat([integrate(ema,chunk,None,'heun',30)[0] for chunk in z.split(64)])
                save_image(to_uint8(generated).float()/255,out/f'preview-{step:07d}.png',nrow=10)
    write_json(out/'stage-complete.json',{'step':args.steps,'kind':args.kind,'elapsed_seconds':time.monotonic()-run_start})


if __name__=='__main__':main()
