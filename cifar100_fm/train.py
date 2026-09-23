"""Original CIFAR-100 pure FM training, with portable paths and strict resume."""
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
                     seed_all, to_uint8, write_json, ARCH)


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
    p.add_argument('--kind',choices=['gaussian'],default='gaussian')
    p.add_argument('--data-root',default='./data')
    p.add_argument('--steps',type=int,default=500000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--sigma',type=float,default=.5)
    p.add_argument('--center-scale',type=float,default=12.)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--warmup',type=int,default=5000)
    p.add_argument('--ema-decay',type=float,default=.9999)
    p.add_argument('--amp',choices=['none','bf16'],default='none')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save-every',type=int,default=5000)
    p.add_argument('--snapshot-every',type=int,default=25000)
    p.add_argument('--keep-previous-checkpoint',action='store_true')
    p.add_argument('--preview-every',type=int,default=25000)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.kind != 'gaussian':
        raise ValueError('this isolated experiment is CIFAR-100 Gaussian-only FM')
    if not (1 <= args.batch_size <= 50000) or min(args.steps,args.warmup,args.save_every,args.snapshot_every)<1 or args.preview_every<0:
        raise ValueError('invalid training settings')
    if not torch.cuda.is_available():
        raise RuntimeError('training requires CUDA; use CPU for unit tests/inference only')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=dict(vars(args),architecture=ARCH,objective='linear_flow_matching',
                dataset='cifar100',num_classes=100,time_encoding='scalar',
                prior_version='rgb_dct6_energy_sorted_100_v1',
                pairing='independent',data_augmentation='horizontal_flip_only',
                source_commit='7c653857de4c25b979a740ef8b6a5d5ddb09b6c1')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seed_all(args.seed)
    model=build_model(args.kind).cuda()
    ema=copy.deepcopy(model).eval()
    for param in ema.parameters():param.requires_grad_(False)
    prior=build_prior(config,'cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    checkpoint_path=out/'training-latest.pt'
    start_step=0
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError('--resume requires training-latest.pt, not an EMA-only checkpoint')
    if args.resume:
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        old=checkpoint['config']
        for key,value in config.items():
            if key not in {'steps','resume','output','data_root','save_every','preview_every','snapshot_every','keep_previous_checkpoint'} and old.get(key)!=value:
                raise ValueError('resume config mismatch: '+key)
        model.load_state_dict(checkpoint['model']);ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step=checkpoint['step']
        torch.set_rng_state(checkpoint['rng_cpu'])
        torch.cuda.set_rng_state(checkpoint['rng_cuda'])
        np.random.set_state(checkpoint['rng_numpy'])
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
        lr=args.lr*min((index+1)/args.warmup,1.)
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
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-args.ema_decay)
        step=index+1
        trained_seconds+=time.monotonic()-tic
        if step==start_step+1 or step%100==0 or step==args.steps:
            record={'step':step,'flow_loss':float(flow_loss.detach()),
                    'grad_norm_before_clip':float(norm),'lr':lr,'seconds_per_step':trained_seconds/(step-start_step),
                    'elapsed_seconds_this_invocation':time.monotonic()-run_start}
            with log_path.open('a') as f:f.write(json.dumps(record)+'\n')
            write_json(out/'progress.json',record)
            print(json.dumps(record),flush=True)
        if step%args.save_every==0 or step%args.snapshot_every==0 or step==args.steps:
            payload={'config':config,'step':step,'model':model.state_dict(),'ema':ema.state_dict(),
                     'optimizer':optimizer.state_dict(),'rng_cpu':torch.get_rng_state(),
                     'rng_cuda':torch.cuda.get_rng_state(),'rng_numpy':np.random.get_state()}
            atomic_save(payload,checkpoint_path,args.keep_previous_checkpoint)
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
