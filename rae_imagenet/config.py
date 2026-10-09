import copy
import hashlib
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs/rae/imagenet256_large.json'
RAEV2_COMMIT = '8a0d238f8dc3b261aba98b217f6c79c0182e8e94'
DINOV3_COMMIT = '94a96ac83c2446f15f9bdcfae23cad3c6a9d4988'
LATENT_SHAPE = (1024, 16, 16)


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(c, world_size=1):
    m, t, e = c['model'], c['train'], c['evaluation']
    if (m['in_channels'], m['input_size'], m['input_size']) != LATENT_SHAPE:
        raise ValueError('The frozen K7 RAE requires latent shape 1024x16x16')
    if m['num_classes'] != 1000:
        raise ValueError('This release uses all 1000 ImageNet classes')
    for width, heads in zip(m['hidden_size'], m['num_heads']):
        if width % heads or (width // heads) % 4:
            raise ValueError('Attention head dimensions must support 2D RoPE')
    for key in ('global_batch', 'microbatch', 'save_every', 'eval_every', 'log_every'):
        if not isinstance(t[key], int) or t[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if t['global_batch'] % (world_size * t['microbatch']):
        raise ValueError('global_batch must be divisible by world_size * microbatch')
    if not 0 <= t['condition_dropout'] <= 1 or not 0 <= t['ema_decay'] < 1:
        raise ValueError('Invalid dropout or EMA decay')
    if t['decay_steps'] <= t['warmup_steps'] or t['warmup_steps'] < 0:
        raise ValueError('decay_steps must be larger than warmup_steps')
    if t['lr'] <= 0 or not 0 <= t['min_lr'] <= t['lr']:
        raise ValueError('Invalid learning-rate range')
    if t['ema_device'] not in ('cpu', 'cuda'):
        raise ValueError('ema_device must be cpu or cuda')
    if t['num_workers'] < 0:
        raise ValueError('num_workers must be nonnegative')
    if t['max_steps'] is not None and t['max_steps'] < 1:
        raise ValueError('max_steps must be positive or null (continuous)')
    if e['samples'] < 1000 or e['samples'] % 1000:
        raise ValueError('Evaluation samples must be a positive multiple of 1000')
    if min(e['batch_size'], e['decode_batch'], e['steps'], e['kid_subsets']) < 1:
        raise ValueError('Evaluation batch sizes, steps and KID subsets must be positive')
    if not e['cfg_scales'] or any(s < 0 for s in e['cfg_scales']):
        raise ValueError('At least one nonnegative CFG scale is required')
    return c


def training_signature(config):
    """Allow operational changes while preserving the mathematical experiment."""
    c = copy.deepcopy(config)
    c.pop('evaluation')
    for k in ('num_workers', 'microbatch', 'save_every', 'eval_every', 'log_every',
              'max_steps', 'ema_device', 'gradient_checkpointing'):
        c['train'].pop(k)
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
