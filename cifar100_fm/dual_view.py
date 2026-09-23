"""Label-free reverse ODE supervision, matching the CIFAR-10 dual-pad4 study."""
import math

import torch
import torch.nn.functional as F

from .common import integrate


def paired_js_loss(scores, temperature):
    """Both views receive gradients; the caller applies the external T**2 factor."""
    if scores.shape[0] % 2 or temperature <= 0:
        raise ValueError('JS requires equally sized view batches and positive temperature')
    log_a, log_b = F.log_softmax(scores.float() / temperature, dim=1).chunk(2)
    log_m = torch.logaddexp(log_a, log_b) - math.log(2.)
    return .5 * ((log_a.exp() * (log_a - log_m)).sum(1).mean()
                 + (log_b.exp() * (log_b - log_m)).sum(1).mean())


def crop_ce_images(images, pad, fraction, generator):
    """Keep the historical RNG draw order, including the all-images permutation."""
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


def paired_ce_images(images, pad, generator):
    first = crop_ce_images(images, pad, 1., generator)
    second = crop_ce_images(images, pad, 1., generator)
    return torch.cat((first, second))


def ramp_weight(maximum, step, origin, warmup):
    return maximum * (min(max((step-origin)/warmup, 0.), 1.) if warmup else 1.)


def auxiliary_loss(model, prior, images, labels, generator, args, step):
    """One Euler4 solve of two crops; the velocity model never receives labels."""
    count = args.ce_batch // 2
    views = paired_ce_images(images[:count], args.ce_crop_pad, generator)
    training = model.training
    model.eval()
    try:
        encoded, _ = integrate(model, views, None, args.ce_solver, args.ce_steps, reverse=True)
        scores = prior.center_coordinates(encoded)
        ce = F.cross_entropy(scores, labels[:count].repeat(2))
        js = paired_js_loss(scores, args.ce_js_temperature)
    finally:
        # The caller performs backward before any further model forward.
        model.train(training)
    origin = args.finetune_start if args.finetune_from else 0
    ce_weight = ramp_weight(args.ce_weight, step, origin, args.ce_warmup)
    js_weight = ramp_weight(args.ce_js_weight, step, origin, args.ce_js_warmup)
    ce_term = ce_weight * ce
    js_term = js_weight * args.ce_js_temperature**2 * js
    if not bool((ce_term+js_term).isfinite()):
        raise RuntimeError('nonfinite reverse CE/JS loss')
    with torch.no_grad():
        pa, pb = scores.argmax(1).chunk(2)
        metrics = dict(reverse_ce=float(ce), js_loss=float(js), ce_weight=ce_weight,
                       js_weight=js_weight, weighted_js_loss=float(js_term),
                       view_disagreement=float((pa != pb).float().mean()),
                       views_both_correct=float(((pa == labels[:count]) &
                                                 (pb == labels[:count])).float().mean()))
    return ce_term, js_term, metrics


def gradient_diagnostics(parameters, flow_grads, js_grads):
    """Gradients are measured before the final joint norm clipping."""
    fm2 = aux2 = ce2 = js2 = fm_aux = ce_js = 0.
    for p, fg, jg in zip(parameters, flow_grads, js_grads):
        if p.grad is None:
            continue
        flow = fg if fg is not None else torch.zeros_like(p.grad)
        jsg = jg if jg is not None else torch.zeros_like(p.grad)
        aux = p.grad.detach() - flow
        ceg = aux - jsg
        fm2 += flow.square().sum()
        aux2 += aux.square().sum()
        ce2 += ceg.square().sum()
        js2 += jsg.square().sum()
        fm_aux += (flow * aux).sum()
        ce_js += (ceg * jsg).sum()
    return dict(fm_grad_norm=float(fm2.sqrt()), weighted_aux_grad_norm=float(aux2.sqrt()),
                weighted_ce_grad_norm=float(ce2.sqrt()), weighted_js_grad_norm=float(js2.sqrt()),
                fm_aux_grad_cosine=float(fm_aux/(fm2*aux2).sqrt().clamp_min(1e-20)),
                ce_js_grad_cosine=float(ce_js/(ce2*js2).sqrt().clamp_min(1e-20)))
