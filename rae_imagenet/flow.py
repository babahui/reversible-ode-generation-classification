import contextlib
import math
import torch


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()


def learning_rate(step, config):
    warm = config['warmup_steps']
    if warm and step <= warm:
        return config['lr'] * step / warm
    frac = min(max((step - warm) / (config['decay_steps'] - warm), 0.), 1.)
    return config['min_lr'] + .5 * (config['lr'] - config['min_lr']) * (1 + math.cos(math.pi * frac))


def fm_loss(model, z, labels, num_classes, dropout):
    noise = torch.randn_like(z)
    time = torch.rand(len(z), device=z.device)
    condition = labels.clone()
    condition[torch.rand(len(z), device=z.device) < dropout] = num_classes
    xt = torch.lerp(noise, z, time[:, None, None, None])
    with autocast(z.device):
        velocity = model(xt, time, condition)
    return (velocity.float() - (z - noise)).square().mean()


@torch.no_grad()
def integrate(model, noise, labels, num_classes=1000, steps=100, scale=1.):
    if steps < 1 or scale < 0:
        raise ValueError('steps must be positive and CFG scale nonnegative')
    x = noise.float().clone()
    for i in range(steps):
        time = torch.full((len(x),), i / steps, device=x.device)
        with autocast(x.device):
            conditional = model(x, time, labels).float()
            if scale == 1:
                velocity = conditional
            else:
                null = model(x, time, torch.full_like(labels, num_classes)).float()
                velocity = null + scale * (conditional - null)
        x = x + velocity / steps
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite generated latent')
    return x


def indexed_noise(indices, shape, seed, device):
    """Each sample has a reproducible CPU noise draw independent of batch size."""
    rows = [torch.randn(shape, generator=torch.Generator().manual_seed(
        (seed + int(i) * 1000003) % (2**63 - 1))) for i in indices]
    return torch.stack(rows).to(device)
