"""Seed-matched RF training with optional class prior, class-wise OT and reverse CE."""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .common import (build_model, build_prior, initial_noise, integrate,
                     seed_all, to_uint8, write_json, ARCH)
from unified_transport.ot_pairing import pair_latents_by_class


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


class StandardPrior:
    def __init__(self,sigma): self.sigma=sigma
    def sample(self,labels):
        return torch.randn(len(labels),3,32,32,device=labels.device)*self.sigma


def reverse_ce_loss(prior, encoded, labels, logits='posterior'):
    scores = (prior.center_coordinates(encoded) if logits == 'center'
              else prior.log_posterior(encoded))
    return F.cross_entropy(scores, labels)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=['label','label_ot','gaussian','gaussian_ot','gaussian_ce','gaussian_ot_ce'],default='label')
    p.add_argument('--data-root',default='/root/autodl-tmp/all_datasets')
    p.add_argument('--steps',type=int,default=400000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--sigma',type=float,default=None)
    p.add_argument('--center-scale',type=float,default=12.)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--warmup',type=int,default=5000)
    p.add_argument('--ema-decay',type=float,default=.9999)
    p.add_argument('--amp',choices=['none','bf16'],default='none')
    p.add_argument('--ce-weight',type=float,default=.001)
    p.add_argument('--ce-every',type=int,default=4)
    p.add_argument('--ce-steps',type=int,default=4)
    p.add_argument('--ce-batch',type=int,default=16)
    p.add_argument('--ce-logits',choices=['posterior','center'],default='posterior')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save-every',type=int,default=5000)
    p.add_argument('--snapshot-every',type=int,default=50000)
    p.add_argument('--keep-previous-checkpoint',action='store_true')
    p.add_argument('--preview-every',type=int,default=5000)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.sigma is None: args.sigma=.5 if args.kind.startswith('gaussian') else 1.
    if args.ce_every<1 or args.ce_steps<1 or args.batch_size<1 or args.save_every<1 or args.snapshot_every<1:
        raise ValueError('invalid training settings')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=dict(vars(args),architecture=ARCH,objective='linear_flow_matching',
                pairing='classwise_sinkhorn_permutation' if '_ot' in args.kind else 'independent',
                data_augmentation='horizontal_flip_only',ot_epsilon=.005,ot_iterations=10,
                reverse_ce_solver='euler',reverse_ce_dropout=False,
                source_commit='7c653857de4c25b979a740ef8b6a5d5ddb09b6c1')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seed_all(args.seed)
    model=build_model(args.kind).cuda()
    ema=copy.deepcopy(model).eval()
    for param in ema.parameters():param.requires_grad_(False)
    prior=build_prior(config,'cuda')
    pairing_prior=prior if prior is not None else StandardPrior(args.sigma)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    checkpoint_path=out/'training-latest.pt'
    start_step=0
    if args.resume and checkpoint_path.exists():
        checkpoint=torch.load(checkpoint_path,map_location='cpu')
        old=checkpoint['config']
        for key,value in config.items():
            historical_defaults={'ce_logits':'posterior'}
            if key not in {'steps','resume','output','save_every','preview_every','snapshot_every','keep_previous_checkpoint'} and old.get(key,historical_defaults.get(key))!=value:
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
    dataset=CIFAR10(args.data_root,train=True,download=False)
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
        if '_ot' in args.kind:
            noise,ot_metrics=pair_latents_by_class(images,labels,pairing_prior,epsilon=.005,iterations=10)
        else:
            noise=initial_noise(labels,config,prior);ot_metrics={}
        t=torch.rand(len(images),device='cuda')
        tt=t[:,None,None,None]
        state=(1-tt)*noise+tt*images
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.amp=='bf16'):
            prediction=model(t,state,y=labels if args.kind.startswith('label') else None)
            flow_loss=(prediction.float()-(images-noise)).square().mean()
        if not bool(flow_loss.isfinite()):raise RuntimeError('nonfinite flow loss')
        flow_loss.backward()
        ce=torch.zeros((),device='cuda')
        if args.kind in {'gaussian_ce','gaussian_ot_ce'} and (index+1)%args.ce_every==0:
            # Deterministic ODE for both CE forward and backward; no labels to velocity.
            model.eval()
            count=min(args.ce_batch,len(images))
            encoded,_=integrate(model,images[:count],None,'euler',args.ce_steps,reverse=True)
            ce=reverse_ce_loss(prior,encoded,labels[:count],args.ce_logits)
            if not bool(ce.isfinite()):raise RuntimeError('nonfinite reverse CE')
            (args.ce_weight*ce).backward()
            model.train()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-args.ema_decay)
        step=index+1
        trained_seconds+=time.monotonic()-tic
        if step==start_step+1 or step%100==0 or step==args.steps:
            record={'step':step,'flow_loss':float(flow_loss.detach()),'reverse_ce':float(ce.detach()),
                    'grad_norm_before_clip':float(norm),'lr':lr,'seconds_per_step':trained_seconds/(step-start_step),
                    'elapsed_seconds_this_invocation':time.monotonic()-run_start,**ot_metrics}
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
                preview_labels=torch.arange(10,device='cuda').repeat_interleave(10)
                generator=torch.Generator(device='cuda').manual_seed(20260912)
                z=initial_noise(preview_labels,config,prior,generator)
                generated,_=integrate(ema,z,preview_labels if args.kind.startswith('label') else None,'heun',30)
                save_image(to_uint8(generated).float()/255,out/f'preview-{step:07d}.png',nrow=10)
    write_json(out/'stage-complete.json',{'step':args.steps,'kind':args.kind,'elapsed_seconds':time.monotonic()-run_start})


if __name__=='__main__':main()
