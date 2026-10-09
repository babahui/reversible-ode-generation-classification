"""ImageNet conditional velocity FM; single GPU or torchrun DDP.

SIGTERM/SIGINT requests an optimizer-boundary checkpoint. Checkpoints include
rank-specific RNG, optimizer, EMA and deterministic data position. Evaluation
runs after saving, and an interrupted periodic evaluation resumes before training.
"""
import argparse
import contextlib
import datetime
import fcntl
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from .config import DEFAULT_CONFIG, load_config, validate_config, training_signature, RAEV2_COMMIT, DINOV3_COMMIT
from .data import ImageNetImages, StepBatchSampler
from .flow import fm_loss, learning_rate
from .model import build_model
from .rae import Encoder, verify_weights, WEIGHT_SHA256
from .state import EMA, atomic_save, write_json, rng_state, restore_rng, now

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def update(model, encoder, iterator, optimizer, train_config, num_classes, device, world_size=1):
    optimizer.zero_grad(set_to_none=True)
    accum = train_config['global_batch'] // (train_config['microbatch'] * world_size)
    total = torch.zeros((), device=device)
    for i in range(accum):
        images, labels = next(iterator)
        z = encoder(images)
        labels = labels.to(device, non_blocking=True)
        sync = model.no_sync() if isinstance(model, DistributedDataParallel) and i < accum-1 else contextlib.nullcontext()
        with sync:
            loss = fm_loss(model, z, labels, num_classes, train_config['condition_dropout'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite FM loss')
            (loss / accum).backward()
        total += loss.detach() / accum
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return total, float(grad)


def synchronized_stop(device, world):
    value = torch.tensor(int(STOP), device=device)
    if world > 1: dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def save_checkpoint(model, ema, optimizer, config, dataset, step, trained, args, rank, world, device):
    states = [None] * world
    state = rng_state()
    if world > 1: dist.all_gather_object(states, state)
    else: states[0] = state
    if rank == 0:
        obj = dict(format_version=1, model=model.state_dict(), ema=ema.state,
                   optimizer=optimizer.state_dict(), rng_by_rank=states, world_size=world,
                   config=config, training_signature=training_signature(config), dataset=dataset,
                   step=step, training_seconds=trained,
                   provenance=dict(raev2_commit=RAEV2_COMMIT, dinov3_commit=DINOV3_COMMIT,
                                   rae_weight_sha256=WEIGHT_SHA256))
        atomic_save(obj, Path(args.output) / 'checkpoints/latest.pt')
        if step % config['train']['eval_every'] == 0:
            milestone = Path(args.output) / f'checkpoints/ema-{step:09d}.pt'
            if not milestone.exists():
                atomic_save({k: obj[k] for k in ('format_version', 'ema', 'config', 'dataset', 'step', 'provenance')}, milestone)
    if world > 1: dist.barrier()


def periodic_evaluation(model, ema, config, args, step, classes, rank, world, device):
    global STOP
    if rank == 0:
        from .evaluate import evaluate, EvaluationInterrupted
        original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        random_state = rng_state()
        flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            model.load_state_dict(ema.state, strict=True)
            evaluate(model, config, args, step, classes, stop=lambda: STOP)
        except EvaluationInterrupted:
            STOP = True
        finally:
            model.load_state_dict(original, strict=True)
            model.train()
            restore_rng(random_state)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
            del original
            torch.cuda.empty_cache()
    if world > 1: dist.barrier()
    return synchronized_stop(device, world)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--data-root', required=True, help='Contains train/<wnid> and val/<wnid>')
    p.add_argument('--weights', required=True, help='Matching encoder checkpoint, decoder.pt, stats.pt')
    p.add_argument('--output', required=True)
    p.add_argument('--rae-repo', default='third_party/RAEv2')
    p.add_argument('--dino-repo', default='third_party/dinov3')
    p.add_argument('--resume', default='auto', help='auto, none, or a full-state checkpoint path')
    p.add_argument('--max-steps', type=int, help='Absolute target optimizer step; default config/null is continuous')
    p.add_argument('--microbatch', type=int)
    p.add_argument('--num-workers', type=int)
    p.add_argument('--allow-subset', action='store_true', help='Pipeline checks only; relax dataset counts, keep 1000 outputs')
    p.add_argument('--inception-weights', help='Optional offline torch-fidelity Inception weights')
    p.add_argument('--classifier-weights', help='Optional offline ResNet50 IMAGENET1K_V2 weights')
    return p


def main(args=None):
    args = args or parser().parse_args()
    config = load_config(args.config)
    for name in ('max_steps', 'microbatch', 'num_workers'):
        value = getattr(args, name)
        if value is not None: config['train'][name] = value
    world, rank, local_rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0))
    validate_config(config, world)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for full ImageNet training; use CPU tests/preflight to validate code')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(4)
    if world > 1: dist.init_process_group('nccl', timeout=datetime.timedelta(hours=24))
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (output / 'train.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args._owns_output_lock = True
        write_json(output / 'status.json', dict(phase='initializing', pid=os.getpid(), updated_utc=now()))
        verify_weights(args.weights)
    if world > 1: dist.barrier()
    data = ImageNetImages(args.data_root, 'train', allow_subset=args.allow_subset)
    ImageNetImages(args.data_root, 'val', classes=data.classes, allow_subset=args.allow_subset)
    dataset = data.signature()
    tc = config['train']
    seed_all(tc['seed'])
    model = build_model(config['model'], args.rae_repo, tc['gradient_checkpointing']).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['lr'], betas=tc['betas'], weight_decay=tc['weight_decay'], foreach=False)
    ema = EMA(model, tc['ema_decay'], tc['ema_device']) if rank == 0 else None
    latest = output / 'checkpoints/latest.pt'
    resume = latest if args.resume == 'auto' else (None if args.resume == 'none' else Path(args.resume))
    if args.resume == 'none' and latest.exists(): raise ValueError('Use a new output directory for a fresh experiment')
    if args.resume not in ('auto', 'none') and not resume.is_file(): raise FileNotFoundError(resume)
    step, trained = 0, 0.
    saved_rng = None
    if resume is not None and resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if 'optimizer' not in saved: raise ValueError('An EMA-only checkpoint cannot resume training')
        if saved['training_signature'] != training_signature(config) or saved['dataset'] != dataset:
            raise ValueError('Training configuration or dataset manifest changed')
        if saved['world_size'] != world: raise ValueError('Exact-state resume requires the original world size')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        if ema is not None: ema.load(saved['ema'])
        step, trained = saved['step'], saved['training_seconds']
        saved_rng = saved['rng_by_rank'][rank]
        del saved
    elif args.resume == 'auto' and (output / 'train.jsonl').exists():
        raise ValueError('Output has training history but no checkpoint; choose a new directory or explicit --resume none')
    encoder = Encoder(args.rae_repo, args.dino_repo, args.weights, device)
    if world > 1: wrapped = DistributedDataParallel(model, device_ids=[local_rank])
    else: wrapped = model
    sampler = StepBatchSampler(len(data), tc['global_batch'], tc['microbatch'], tc['seed'], step, tc['max_steps'], rank, world)
    loader = DataLoader(data, batch_sampler=sampler, num_workers=tc['num_workers'], pin_memory=True,
                        persistent_workers=tc['num_workers'] > 0,
                        generator=torch.Generator().manual_seed(tc['seed']+50000+rank))
    iterator = iter(loader)
    if saved_rng is not None: restore_rng(saved_rng)
    else: seed_all(tc['seed']+rank*100003)
    if rank == 0:
        write_json(output / 'config.json', config)
        write_json(output / 'dataset.json', dataset)
        write_json(output / 'launch.json', dict(pid=os.getpid(), step=step, world_size=world, started_utc=now(), args=vars(args)))
    if step > 0 and step % tc['eval_every'] == 0:
        if periodic_evaluation(model, ema, config, args, step, dataset['classes'], rank, world, device):
            if rank == 0: write_json(output / 'status.json', dict(phase='paused', step=step, updated_utc=now()))
            if world > 1: dist.destroy_process_group()
            return
    torch.cuda.reset_peak_memory_stats(device)
    stopping = False
    while tc['max_steps'] is None or step < tc['max_steps']:
        tic = time.monotonic()
        step += 1
        lr = learning_rate(step, tc)
        for group in optimizer.param_groups: group['lr'] = lr
        loss, grad = update(wrapped, encoder, iterator, optimizer, tc, config['model']['num_classes'], device, world)
        if ema is not None: ema.update(model)
        if world > 1:
            dist.all_reduce(loss); loss /= world
        torch.cuda.synchronize(device)
        elapsed = time.monotonic()-tic; trained += elapsed
        stopping = synchronized_stop(device, world)
        final = tc['max_steps'] is not None and step >= tc['max_steps']
        if rank == 0 and (step % tc['log_every'] == 0 or step == 1):
            record = dict(phase='training', step=step, max_steps=tc['max_steps'], next_evaluation_step=(step//tc['eval_every']+1)*tc['eval_every'],
                          loss=float(loss), grad_norm=grad, lr=lr, seconds_per_step=elapsed, training_seconds=trained,
                          peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30, updated_utc=now())
            with (output / 'train.jsonl').open('a') as f:
                import json
                f.write(json.dumps(record)+'\n')
            write_json(output / 'status.json', record)
            print(record, flush=True)
        if step % tc['save_every'] == 0 or step % tc['eval_every'] == 0 or stopping or final:
            save_checkpoint(model, ema, optimizer, config, dataset, step, trained, args, rank, world, device)
        if stopping: break
        if step % tc['eval_every'] == 0:
            if periodic_evaluation(model, ema, config, args, step, dataset['classes'], rank, world, device):
                stopping = True
                break
    if rank == 0:
        write_json(output / 'status.json', dict(phase='paused' if stopping else 'complete', step=step, updated_utc=now()))
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__':
    cli_args = parser().parse_args()
    try:
        main(cli_args)
    except BlockingIOError:
        # Another trainer owns this output directory; do not overwrite its status.
        raise
    except Exception:
        import traceback
        if int(os.environ.get('RANK', 0)) == 0 and getattr(cli_args, '_owns_output_lock', False):
            write_json(Path(cli_args.output) / 'status.json',
                       dict(phase='failed', error=traceback.format_exc(), updated_utc=now()))
        raise
