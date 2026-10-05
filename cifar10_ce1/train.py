"""Seed-matched RF training with optional class prior, class-wise OT and reverse CE."""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .common import (build_model, build_prior, initial_noise, integrate,
                     seed_all, to_uint8, write_json, sha256, ARCH)
from .ot_pairing import pair_latents_by_class


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


def paired_js_loss(scores, temperature):
    """Symmetric JS of paired views; both branches retain their gradients."""
    if scores.shape[0] % 2 or temperature <= 0:
        raise ValueError('JS requires equally sized view batches and positive temperature')
    log_a, log_b = F.log_softmax(scores.float() / temperature, dim=1).chunk(2)
    log_m = torch.logaddexp(log_a, log_b) - math.log(2.)
    return .5 * ((log_a.exp() * (log_a - log_m)).sum(1).mean()
                 + (log_b.exp() * (log_b - log_m)).sum(1).mean())


def crop_ce_images(images, pad, fraction, generator):
    """Crop only CE views, leaving FM tensors and its global RNG untouched."""
    count = int(len(images) * fraction)
    if pad == 0 or count == 0:
        return images
    if generator is None:
        raise ValueError('CE augmentation requires an independent generator')
    height, width = images.shape[-2:]
    selected = torch.randperm(len(images), device=images.device, generator=generator)[:count]
    offsets = torch.randint(2 * pad + 1, (count, 2), device=images.device, generator=generator)
    padded = F.pad(images[selected], (pad, pad, pad, pad), mode='reflect').permute(0, 2, 3, 1)
    rows = offsets[:, 0, None] + torch.arange(height, device=images.device)
    cols = offsets[:, 1, None] + torch.arange(width, device=images.device)
    cropped = padded[torch.arange(count, device=images.device)[:, None, None],
                     rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    result = images.clone()
    result[selected] = cropped
    return result


def paired_ce_images(images, pad, generator, crop_both=False):
    """Sample both crops from the same input, without resetting the augmentation RNG."""
    first=crop_ce_images(images,pad,1.,generator) if crop_both else images
    second=crop_ce_images(images,pad,1.,generator)
    return torch.cat((first,second))


def learning_rate(args, index):
    if args.finetune_from:
        fraction = (index - args.finetune_start) / max(args.finetune_steps - 1, 1)
        fraction = min(max(fraction, 0.), 1.)
        return args.min_lr + .5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * fraction))
    return args.lr * min((index + 1) / args.warmup, 1.)


def ce_weight(args, index):
    origin = args.finetune_start if args.finetune_from else 0
    ramp = min(max((index + 1 - origin) / args.ce_warmup, 0.), 1.) if args.ce_warmup else 1.
    return args.ce_weight * ramp


def validate_constant_extension(old, args):
    if (not args.allow_constant_finetune_extension or not old.get('finetune_from')
            or args.finetune_steps <= old.get('finetune_steps', 0)
            or not (old.get('lr') == old.get('min_lr') == args.lr == args.min_lr)):
        raise ValueError('Only an explicitly authorized constant-LR extension is allowed')


def initialize_from_ema(model, ema, checkpoint):
    model.load_state_dict(checkpoint['ema'], strict=True)
    ema.load_state_dict(checkpoint['ema'], strict=True)


def validate_ema_fork(old, args):
    if old['kind'] == 'gaussian' and args.kind in {'gaussian', 'gaussian_ce'}:
        return
    if old['kind'] != 'gaussian_ce' or args.kind != 'gaussian_ce':
        raise ValueError('EMA fork requires a Gaussian FM or same-objective FM+CE source')
    unchanged = ('ce_weight', 'ce_every', 'ce_steps', 'ce_batch', 'ce_logits', 'ce_solver',
                 'ce_crop_pad', 'ce_crop_fraction', 'ce_crop_seed', 'ce_paired_views',
                 'ce_paired_crop_both', 'ce_js_weight', 'ce_js_temperature')
    for key in unchanged:
        if old.get(key) != getattr(args, key):
            raise ValueError('Same-objective EMA fork cannot change ' + key)
    elapsed = args.finetune_start - old.get('finetune_start', 0)
    if elapsed < max(old.get('ce_warmup', 0), old.get('ce_js_warmup', 0)):
        raise ValueError('Source auxiliary warmup has not finished')
    if args.ce_warmup or args.ce_js_warmup:
        raise ValueError('Keep saturated auxiliary weights: use zero warmup for the EMA fork')


def gradient_comparison(parameters, flow_grads):
    # After the second backward, .grad contains FM + weighted CE gradients.
    dot, fm2, ce2 = [], [], []
    for parameter, flow in zip(parameters, flow_grads):
        if flow is None:
            continue
        correction = parameter.grad.detach() - flow
        dot.append((flow * correction).sum())
        fm2.append(flow.square().sum())
        ce2.append(correction.square().sum())
    a, b, c = torch.stack(fm2).sum(), torch.stack(ce2).sum(), torch.stack(dot).sum()
    return dict(fm_grad_norm=float(a.sqrt()), weighted_ce_grad_norm=float(b.sqrt()),
                fm_ce_grad_cosine=float(c / (a*b).sqrt().clamp_min(1e-20)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=['label','label_ot','gaussian','gaussian_ot','gaussian_ce','gaussian_ot_ce'],default='label')
    p.add_argument('--data-root',default='./data')
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
    p.add_argument('--ce-solver',choices=['euler','heun'],default='euler')
    p.add_argument('--ce-warmup',type=int,default=0)
    p.add_argument('--ce-crop-pad',type=int,default=0)
    p.add_argument('--ce-crop-fraction',type=float,default=.5)
    p.add_argument('--ce-crop-seed',type=int,default=20260918)
    p.add_argument('--ce-paired-views',action='store_true')
    p.add_argument('--ce-paired-crop-both',action='store_true')
    p.add_argument('--ce-js-weight',type=float,default=0.)
    p.add_argument('--ce-js-temperature',type=float,default=2.)
    p.add_argument('--ce-js-warmup',type=int,default=5000)
    p.add_argument('--grad-diagnostics-every',type=int,default=0)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save-every',type=int,default=5000)
    p.add_argument('--snapshot-every',type=int,default=50000)
    p.add_argument('--keep-previous-checkpoint',action='store_true')
    p.add_argument('--preview-every',type=int,default=5000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--finetune-from',default=None)
    p.add_argument('--finetune-init',choices=['full','ema'],default='full')
    p.add_argument('--finetune-start',type=int,default=1000000)
    p.add_argument('--finetune-steps',type=int,default=50000)
    p.add_argument('--min-lr',type=float,default=1e-5)
    p.add_argument('--allow-constant-finetune-extension',action='store_true')
    p.add_argument('--objective-fork',action='store_true',help='Explicit full-state CE/perturbation ablation fork')
    p.add_argument('--ce-perturbation',choices=['none','input','trajectory','trajectory75','gradient','margin','selective_margin','hard_negative','hard_negative_allpos'],default='none')
    p.add_argument('--ce-perturb-rms',type=float,default=.02)
    p.add_argument('--ce-perturb-prob',type=float,default=.5)
    p.add_argument('--ce-perturb-seed',type=int,default=20260925)
    p.add_argument('--ce-select-fraction',type=float,default=.25)
    p.add_argument('--ce-perturb-warmup',type=int,default=0)
    args=p.parse_args()
    if args.sigma is None: args.sigma=.5 if args.kind.startswith('gaussian') else 1.
    if args.ce_perturb_rms < 0 or not 0 <= args.ce_perturb_prob <= 1:
        raise ValueError('Invalid perturbation settings')
    if not 0 < args.ce_select_fraction <= 1 or args.ce_perturb_warmup < 0:
        raise ValueError('Invalid selective perturbation settings')
    if args.ce_perturbation != 'none' and not (args.ce_paired_views and args.ce_solver=='euler' and args.ce_steps==4):
        raise ValueError('Perturbation ablation requires paired Euler4')
    if args.objective_fork and (not args.finetune_from or args.finetune_init!='full'):
        raise ValueError('Objective fork requires a full-state source')
    if args.ce_every<1 or args.ce_steps<1 or args.batch_size<1 or args.save_every<1 or args.snapshot_every<1:
        raise ValueError('invalid training settings')
    if args.ce_batch<1 or args.ce_warmup<0 or args.ce_weight<0 or args.grad_diagnostics_every<0:
        raise ValueError('invalid CE settings')
    if not 0 <= args.ce_crop_pad < 32 or not 0 <= args.ce_crop_fraction <= 1:
        raise ValueError('invalid CE crop settings')
    if args.ce_crop_pad and args.kind not in {'gaussian_ce','gaussian_ot_ce'}:
        raise ValueError('CE crops require a CE training objective')
    if args.ce_js_weight < 0 or args.ce_js_temperature <= 0 or args.ce_js_warmup < 0:
        raise ValueError('Invalid JS settings')
    if args.ce_js_weight and not args.ce_paired_views:
        raise ValueError('JS requires paired views')
    if args.ce_paired_crop_both and not args.ce_paired_views:
        raise ValueError('Cropping both views requires paired CE')
    if args.ce_paired_views and (args.kind != 'gaussian_ce' or args.ce_logits != 'center'
            or args.ce_batch % 2 or args.ce_batch // 2 > args.batch_size
            or args.ce_crop_pad == 0 or args.ce_crop_fraction != 1.):
        raise ValueError('Paired CE requires center logits, even CE batch, and all second views cropped')
    if args.finetune_init=='ema' and not args.finetune_from:
        raise ValueError('EMA initialization requires an explicit source')
    if args.finetune_from:
        args.finetune_from=str(Path(args.finetune_from).resolve())
        if (args.finetune_steps < 2 or not 0 < args.min_lr <= args.lr
                or not args.finetune_start < args.steps <= args.finetune_start + args.finetune_steps):
            raise ValueError('invalid fine-tuning schedule')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=dict(vars(args),architecture=ARCH,objective='linear_flow_matching',
                pairing='classwise_sinkhorn_permutation' if '_ot' in args.kind else 'independent',
                data_augmentation='horizontal_flip_only',ot_epsilon=.005,ot_iterations=10,
                reverse_ce_solver=args.ce_solver,reverse_ce_dropout=False,
                source_commit='7c653857de4c25b979a740ef8b6a5d5ddb09b6c1')
    if args.finetune_from:
        config.update(finetune_source_sha256=sha256(args.finetune_from),
                      lr_schedule='constant_finetune' if args.lr==args.min_lr else 'cosine_finetune',
                      restore_policy='ema_reset_adam_rng' if args.finetune_init=='ema' else 'model_ema_adam_rng')
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
    ce_crop_rng = (torch.Generator(device='cuda').manual_seed(args.ce_crop_seed)
                   if args.ce_crop_pad else None)
    perturb_rng = (torch.Generator(device='cuda').manual_seed(args.ce_perturb_seed)
                   if args.ce_perturbation != 'none' else None)
    checkpoint_path=out/'training-latest.pt'
    start_step=0
    is_fork=bool(args.finetune_from and not checkpoint_path.exists())
    if (args.resume and checkpoint_path.exists()) or is_fork:
        checkpoint=torch.load(args.finetune_from if is_fork else checkpoint_path,map_location='cpu')
        old=checkpoint['config']
        mutable={'data_root','finetune_from','steps','resume','output','save_every','preview_every','snapshot_every','keep_previous_checkpoint',
                 'allow_constant_finetune_extension'}
        historical_defaults={'ce_logits':'posterior','finetune_from':None,
                             'finetune_start':1000000,'finetune_steps':50000,'min_lr':1e-5,
                             'finetune_init':'full','ce_solver':'euler','ce_warmup':0,
                             'grad_diagnostics_every':0,'ce_crop_pad':0,
                             'ce_crop_fraction':.5,'ce_crop_seed':20260918,
                             'ce_paired_views':False,'ce_paired_crop_both':False,'ce_js_weight':0.,
                             'ce_js_temperature':2.,'ce_js_warmup':5000}
        historical_defaults.update(objective_fork=False,ce_perturbation='none',ce_perturb_rms=.02,
                                   ce_perturb_prob=.5,ce_perturb_seed=20260925)
        historical_defaults.update(ce_select_fraction=.25,ce_perturb_warmup=0)
        if is_fork:
            if checkpoint['step'] != args.finetune_start:
                raise ValueError('fine-tuning source step mismatch')
            mutable |= {'lr','finetune_from','finetune_start','finetune_steps','min_lr',
                        'finetune_source_sha256','lr_schedule','restore_policy'}
            if args.objective_fork:
                if not (old['kind']==args.kind=='gaussian_ce' and old.get('ce_paired_crop_both')
                        and args.ce_paired_crop_both and args.ce_warmup==args.ce_js_warmup==0
                        and args.lr==args.min_lr==old['lr']==old['min_lr']):
                    raise ValueError('Objective fork must preserve paired CE, full state and constant LR')
                if args.finetune_start-old['finetune_start'] < max(old['ce_warmup'],old['ce_js_warmup']):
                    raise ValueError('Source CE/JS warmup must have finished')
                mutable |= {'objective_fork','ce_js_weight','ce_warmup','ce_js_warmup',
                            'ce_perturbation','ce_perturb_rms','ce_perturb_prob','ce_perturb_seed'}
                mutable |= {'ce_select_fraction','ce_perturb_warmup'}
            if old['kind']=='gaussian' and args.kind=='gaussian_ce':
                # Explicit objective change at a new fork, never during resume.
                mutable |= {'kind','ce_weight','ce_every','ce_steps','ce_batch','ce_logits',
                            'ce_solver','ce_warmup','grad_diagnostics_every','reverse_ce_solver',
                            'ce_crop_pad','ce_crop_fraction','ce_crop_seed',
                            'ce_paired_views','ce_paired_crop_both','ce_js_weight','ce_js_temperature','ce_js_warmup'}
            if args.finetune_init=='ema':
                validate_ema_fork(old,args)
                mutable |= {'kind','finetune_init','ce_weight','ce_every','ce_steps','ce_batch',
                            'ce_logits','ce_solver','ce_warmup','ce_js_warmup',
                            'grad_diagnostics_every','reverse_ce_solver'}
        elif args.finetune_steps != old.get('finetune_steps',50000):
            validate_constant_extension(old,args)
            mutable.add('finetune_steps')
        for key,value in config.items():
            if key not in mutable and old.get(key,historical_defaults.get(key))!=value:
                raise ValueError('resume config mismatch: '+key)
        if is_fork and args.finetune_init=='ema':
            initialize_from_ema(model,ema,checkpoint)
            seed_all(args.seed+12345)
        else:
            model.load_state_dict(checkpoint['model']);ema.load_state_dict(checkpoint['ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            torch.set_rng_state(checkpoint['rng_cpu'])
            torch.cuda.set_rng_state(checkpoint['rng_cuda'])
            np.random.set_state(checkpoint['rng_numpy'])
        start_step=checkpoint['step']
        if ce_crop_rng is not None and (not is_fork or args.objective_fork):
            if 'ce_crop_rng_state' not in checkpoint:
                raise ValueError('Missing CE crop RNG state for exact resume')
            ce_crop_rng.set_state(checkpoint['ce_crop_rng_state'])
        if perturb_rng is not None and not is_fork:
            perturb_rng.set_state(checkpoint['ce_perturb_rng_state'])
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
    class_indices=[torch.nonzero(targets==c,as_tuple=False).flatten() for c in range(10)]
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
        lr=learning_rate(args,index)
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
        js=torch.zeros((),device='cuda')
        js_weight=0.
        consistency_metrics={}
        gradient_metrics={}
        weight=ce_weight(args,index)
        if args.kind in {'gaussian_ce','gaussian_ot_ce'} and (index+1)%args.ce_every==0:
            # Deterministic ODE for both CE forward and backward; no labels to velocity.
            model.eval()
            count=min(args.ce_batch,len(images))
            diagnostic=bool(args.grad_diagnostics_every and (index+1)%args.grad_diagnostics_every==0)
            parameters=list(model.parameters())
            flow_grads=[p.grad.detach().clone() if p.grad is not None else None for p in parameters] if diagnostic else None
            ce_labels=labels[:count]
            if args.ce_paired_views:
                count=args.ce_batch//2
                base=images[:count]
                ce_images=paired_ce_images(base,args.ce_crop_pad,ce_crop_rng,args.ce_paired_crop_both)
                ce_labels=labels[:count].repeat(2)
            else:
                ce_images=crop_ce_images(images[:count],args.ce_crop_pad,args.ce_crop_fraction,ce_crop_rng)
            if args.ce_perturbation == 'none':
                encoded,_=integrate(model,ce_images,None,args.ce_solver,args.ce_steps,reverse=True)
            else:
                from .trajectory_augmentation import reverse_with_perturbation
                perturb_origin = args.finetune_start if args.finetune_from else 0
                perturb_ramp = (min(max((index+1-perturb_origin)/args.ce_perturb_warmup,0.),1.)
                                if args.ce_perturb_warmup else 1.)
                encoded,_,perturb_metrics=reverse_with_perturbation(
                    model,ce_images,args.ce_perturbation,perturb_rng,args.ce_perturb_rms*perturb_ramp,args.ce_perturb_prob,
                    labels=ce_labels,prior=prior,training_data=data,
                    class_indices=class_indices,select_fraction=args.ce_select_fraction)
                consistency_metrics.update(perturb_metrics)
            ce=reverse_ce_loss(prior,encoded,ce_labels,args.ce_logits)
            if not bool(ce.isfinite()):raise RuntimeError('nonfinite reverse CE')
            if args.ce_paired_views:
                scores=prior.center_coordinates(encoded)
                js=paired_js_loss(scores,args.ce_js_temperature)
                origin=args.finetune_start if args.finetune_from else 0
                ramp=min((index+1-origin)/args.ce_js_warmup,1.) if args.ce_js_warmup else 1.
                js_weight=args.ce_js_weight*ramp
                if not bool(js.isfinite()):raise RuntimeError('nonfinite JS loss')
                with torch.no_grad():
                    pa,pb=scores.argmax(1).chunk(2)
                    consistency_metrics.update(view_disagreement=float((pa!=pb).float().mean()),
                        views_both_correct=float(((pa==labels[:count]) & (pb==labels[:count])).float().mean()))
            js_term=js_weight*args.ce_js_temperature**2*js
            js_grads=(torch.autograd.grad(js_term,parameters,retain_graph=True,allow_unused=True)
                      if diagnostic and args.ce_js_weight else None)
            (weight*ce+js_term).backward()
            if diagnostic:
                gradient_metrics=gradient_comparison(parameters,flow_grads)
                if js_grads is not None:
                    ce2=js2=dot=0.
                    for p,fg,jg in zip(parameters,flow_grads,js_grads):
                        if p.grad is None:continue
                        cg=p.grad.detach()-(fg if fg is not None else 0)-(jg if jg is not None else 0)
                        ce2=ce2+cg.square().sum()
                        if jg is not None:
                            js2=js2+jg.square().sum();dot=dot+(cg*jg).sum()
                    gradient_metrics.update(weighted_aux_grad_norm=gradient_metrics.pop('weighted_ce_grad_norm'),
                        fm_aux_grad_cosine=gradient_metrics.pop('fm_ce_grad_cosine'),
                        weighted_ce_grad_norm=float(ce2.sqrt()),weighted_js_grad_norm=float(js2.sqrt()),
                        ce_js_grad_cosine=float(dot/(ce2*js2).sqrt().clamp_min(1e-20)))
                    del js_grads
                del flow_grads
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
                    'elapsed_seconds_this_invocation':time.monotonic()-run_start,
                    'ce_weight':weight,**ot_metrics,**gradient_metrics}
            if args.ce_paired_views:
                record.update(js_loss=float(js.detach()),js_weight=js_weight,
                              weighted_js_loss=float((js_weight*args.ce_js_temperature**2*js).detach()),
                              **consistency_metrics)
            with log_path.open('a') as f:f.write(json.dumps(record)+'\n')
            write_json(out/'progress.json',record)
            print(json.dumps(record),flush=True)
        if step%args.save_every==0 or step%args.snapshot_every==0 or step==args.steps:
            payload={'config':config,'step':step,'model':model.state_dict(),'ema':ema.state_dict(),
                     'optimizer':optimizer.state_dict(),'rng_cpu':torch.get_rng_state(),
                     'rng_cuda':torch.cuda.get_rng_state(),'rng_numpy':np.random.get_state()}
            if ce_crop_rng is not None:
                payload['ce_crop_rng_state']=ce_crop_rng.get_state()
            if perturb_rng is not None:
                payload['ce_perturb_rng_state']=perturb_rng.get_state()
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
