"""Training-only, single-kick perturbations of paired reverse Euler solves."""
import torch

from .common import integrate


def sample_perturbation(images, generator, probability=.5):
    """The C/D arms draw exactly the same masks, directions and internal times."""
    if len(images) % 2:
        raise ValueError('Expected two views per image')
    pairs = len(images)//2
    enabled = torch.rand(pairs, device=images.device, generator=generator) < probability
    which = torch.randint(2, (pairs,), device=images.device, generator=generator)
    mask = torch.zeros(len(images), device=images.device, dtype=torch.bool)
    mask[torch.arange(pairs, device=images.device)+which*pairs] = enabled
    positions = torch.randint(1, 4, (len(images),), device=images.device, generator=generator)
    direction = torch.randn(images.shape, device=images.device, dtype=images.dtype, generator=generator)
    direction = direction/direction.square().mean((1,2,3), keepdim=True).sqrt().clamp_min(1e-12)
    return mask, positions, direction


def kick(state, mask, direction, strength):
    # Treat the radius as a fixed augmentation scale, without gradient through it.
    radius = state.detach().square().mean((1,2,3), keepdim=True).sqrt()
    return state + strength*radius*direction*mask[:,None,None,None]


def selective_margin_mask(scores, labels, generator, fraction=.25, probability=1.):
    """Attack one view of the lowest-margin, correctly classified TRAIN pairs."""
    pairs = len(labels) // 2
    if len(labels) % 2 or not torch.equal(labels[:pairs], labels[pairs:]):
        raise ValueError('Selective margin requires aligned paired labels')
    scores = scores.detach()
    competitors = scores.clone()
    competitors[torch.arange(len(labels), device=labels.device), labels] = -float('inf')
    margin = scores.gather(1, labels[:, None]).squeeze(1) - competitors.max(1).values
    pair_margin = torch.minimum(margin[:pairs], margin[pairs:])
    eligible = torch.nonzero(pair_margin > 0, as_tuple=False).flatten()
    count = min(int(pairs * fraction), len(eligible))
    selected = eligible[torch.argsort(pair_margin[eligible])[:count]]
    enabled = torch.rand(len(selected), device=labels.device, generator=generator) < probability
    selected = selected[enabled]
    which = torch.randint(2, (len(selected),), device=labels.device, generator=generator)
    mask = torch.zeros(len(labels), device=labels.device, dtype=torch.bool)
    mask[selected + which * pairs] = True
    metrics = {'selected_pair_fraction': len(selected) / pairs,
               'eligible_pair_fraction': len(eligible) / pairs,
               'clean_pair_margin_mean': float(pair_margin.mean()),
               'selected_pair_margin_mean': float(pair_margin[selected].mean()) if len(selected) else None}
    return mask, metrics


def reverse_with_perturbation(model, images, mode, generator, strength=.02, probability=.5,
                              forced_plan=None, labels=None, prior=None,
                              training_data=None, class_indices=None, select_fraction=.25):
    if mode == 'none' or strength == 0 or probability == 0:
        z, nfe = integrate(model, images, None, 'euler', 4, reverse=True)
        return z, nfe, {'perturbed_view_fraction': 0.}
    if mode not in ('input', 'trajectory', 'trajectory75', 'gradient', 'margin', 'selective_margin', 'hard_negative', 'hard_negative_allpos'):
        raise ValueError('Unknown perturbation mode')
    if mode in ('gradient','margin','selective_margin','hard_negative','hard_negative_allpos') and (labels is None or prior is None):
        raise ValueError('Gradient perturbation requires labels and prior')
    if mode in ('gradient','margin','selective_margin'):
        # Integrate to t=.75, calculate endpoint-CE ascent per sample, then
        # detach the attack direction before the trainable tail (first-order).
        if mode != 'selective_margin':
            mask, _, _ = (sample_perturbation(images, generator, probability)
                          if forced_plan is None else forced_plan)
        selection_metrics = {}
        x = images
        t = torch.as_tensor(1., device=x.device, dtype=x.dtype)
        z_base = x - .25*model(t, x, y=None)
        z_attack = z_base.detach().requires_grad_(True)
        clean_tail = z_attack
        for index in range(1, 4):
            t = torch.as_tensor(1.-index*.25, device=x.device, dtype=x.dtype)
            clean_tail = clean_tail - .25*model(t, clean_tail, y=None)
        scores = prior.center_coordinates(clean_tail)
        if mode == 'selective_margin':
            if forced_plan is not None:
                raise ValueError('Selective margin determines its mask from clean TRAIN logits')
            mask, selection_metrics = selective_margin_mask(
                scores, labels, generator, select_fraction, probability)
        if mode == 'gradient':
            attack_loss = torch.nn.functional.cross_entropy(scores, labels, reduction='sum')
        else:
            competitors=scores.detach().clone()
            competitors[torch.arange(len(labels),device=labels.device),labels]=-float('inf')
            k=competitors.argmax(1)
            margins_up=(scores[torch.arange(len(labels),device=labels.device),k]-
                        scores[torch.arange(len(labels),device=labels.device),labels])
            attack_loss=(margins_up * mask).sum() if mode == 'selective_margin' else margins_up.sum()
        grad = torch.autograd.grad(attack_loss, z_attack, retain_graph=False,
                                   create_graph=False)[0].detach()
        radius = z_base.detach().square().mean((1,2,3), keepdim=True).sqrt()
        grad_radius = grad.square().mean((1,2,3), keepdim=True).sqrt().clamp_min(1e-12)
        z = z_base + strength*radius*grad/grad_radius*mask[:,None,None,None]
        if mode == 'selective_margin':
            actual = (z.detach() - z_base.detach()).square().mean((1,2,3)).sqrt()
            relative = actual / radius.flatten().clamp_min(1e-12)
            selection_metrics['actual_relative_rms_selected'] = float(relative[mask].mean()) if bool(mask.any()) else 0.
        for index in range(1, 4):
            t = torch.as_tensor(1.-index*.25, device=x.device, dtype=x.dtype)
            z = z - .25*model(t, z, y=None)
        return z, 4, {'perturbed_view_fraction': float(mask.float().mean()),
                      'perturb_relative_rms': strength,
                      'perturb_position': .75,
                      'perturb_direction': mode, **selection_metrics}
    if mode in ('hard_negative', 'hard_negative_allpos'):
        if training_data is None or class_indices is None:
            raise ValueError('Hard-negative perturbation requires training data and class indices')
        if mode == 'hard_negative_allpos':
            # Use one common five-segment path so .70, .50 and .25 are true
            # trajectory states.  The same detached hard-negative trajectory
            # is injected at all three nodes, each with the requested fraction.
            segments = ((1.00, -.25, .75), (.75, -.05, .70),
                        (.70, -.20, .50), (.50, -.25, .25),
                        (.25, -.25, 0.00))
            mask, _, _ = (sample_perturbation(images, generator, probability)
                          if forced_plan is None else forced_plan)
            clean_states = {1.00: images}
            clean = images
            for t_value, dt, node in segments:
                t = torch.as_tensor(t_value, device=images.device, dtype=images.dtype)
                clean = clean + dt * model(t, clean, y=None)
                clean_states[node] = clean
            with torch.no_grad():
                clean_scores = prior.center_coordinates(clean_states[0.00])
                competitors = clean_scores.clone()
                competitors[torch.arange(len(labels), device=labels.device), labels] = -float('inf')
                hard_cls = competitors.argmax(1)
                distractor_idx = torch.empty_like(hard_cls)
                for cls, pool in enumerate(class_indices):
                    selected = hard_cls == cls
                    if bool(selected.any()):
                        picks = torch.randint(len(pool), (int(selected.sum()),), device=images.device, generator=generator)
                        distractor_idx[selected] = pool[picks]
                distractor = training_data[distractor_idx].float().div(255).mul(2).sub(1)
                hard_states = {1.00: distractor}
                hard = distractor
                for t_value, dt, node in segments:
                    t = torch.as_tensor(t_value, device=images.device, dtype=images.dtype)
                    hard = hard + dt * model(t, hard, y=None)
                    hard_states[node] = hard.detach()
            z = clean_states[.70]
            z = z + strength * (hard_states[.70] - clean_states[.70].detach()) * mask[:, None, None, None]
            for t_value, dt, node in segments[2:]:
                t = torch.as_tensor(t_value, device=images.device, dtype=images.dtype)
                z = z + dt * model(t, z, y=None)
                if node in (.50, .25):
                    z = z + strength * (hard_states[node] - clean_states[node].detach()) * mask[:, None, None, None]
            return z, 8, {'perturbed_view_fraction': float(mask.float().mean()),
                          'perturb_relative_rms': strength,
                          'perturb_position': '.70/.50/.25',
                          'perturb_direction': 'hard_negative_allpos',
                          'perturb_count': 3}
        mask, _, _ = (sample_perturbation(images, generator, probability)
                      if forced_plan is None else forced_plan)
        t1 = torch.as_tensor(1., device=images.device, dtype=images.dtype)
        z_base = images - .25*model(t1, images, y=None)
        with torch.no_grad():
            clean_tail=z_base.detach()
            for index in range(1,4):
                t=torch.as_tensor(1.-index*.25,device=images.device,dtype=images.dtype)
                clean_tail=clean_tail-.25*model(t,clean_tail,y=None)
            scores=prior.center_coordinates(clean_tail)
            competitors=scores.clone()
            competitors[torch.arange(len(labels),device=labels.device),labels]=-float('inf')
            hard_cls=competitors.argmax(1)
            distractor_idx=torch.empty_like(hard_cls)
            for cls,pool in enumerate(class_indices):
                selected=(hard_cls==cls)
                if bool(selected.any()):
                    picks=torch.randint(len(pool),(int(selected.sum()),),device=images.device,generator=generator)
                    distractor_idx[selected]=pool[picks]
            distractor=training_data[distractor_idx].float().div(255).mul(2).sub(1)
            zd_base=distractor-.25*model(t1,distractor,y=None)
        # Preserve the clean view's trainable prefix, while the distractor
        # trajectory acts as a detached semantic direction.
        z=z_base+strength*(zd_base.detach()-z_base.detach())*mask[:,None,None,None]
        for index in range(1,4):
            t=torch.as_tensor(1.-index*.25,device=images.device,dtype=images.dtype)
            z=z-.25*model(t,z,y=None)
        return z,4,{'perturbed_view_fraction':float(mask.float().mean()),
                    'perturb_relative_rms':strength,'perturb_position':.75,
                    'perturb_direction':'hard_negative'}
    mask, positions, direction = (sample_perturbation(images, generator, probability)
                                 if forced_plan is None else forced_plan)
    if mode == 'trajectory75':
        positions = torch.ones_like(positions)
    x = kick(images, mask, direction, strength) if mode == 'input' else images
    for index in range(4):
        if mode in ('trajectory', 'trajectory75') and index > 0:
            x = kick(x, mask & (positions == index), direction, strength)
        t = torch.as_tensor(1.-index*.25, device=x.device, dtype=x.dtype)
        x = x - .25*model(t, x, y=None)
    return x, 4, {'perturbed_view_fraction': float(mask.float().mean()),
                  'perturb_relative_rms': strength}
