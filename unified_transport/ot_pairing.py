"""Class-wise minibatch optimal-transport couplings for flow matching."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _squared_cost(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return mean squared distances without materializing a BxBxD tensor."""
    left = left.flatten(1).float()
    right = right.flatten(1).float()
    dimension = left.shape[1]
    cost = (
        left.square().sum(1, keepdim=True)
        + right.square().sum(1, keepdim=True).T
        - 2.0 * left @ right.T
    ) / float(dimension)
    return cost.clamp_min(0.0)


@torch.no_grad()
def _sinkhorn_plan(cost: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    """Compute an entropic OT plan with uniform row/column marginals."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    count = cost.shape[0]
    log_kernel = -cost.float() / epsilon
    log_mass = -torch.log(torch.tensor(float(count), device=cost.device))
    log_row = torch.zeros(count, device=cost.device)
    log_column = torch.zeros(count, device=cost.device)
    for _ in range(iterations):
        log_row = log_mass - torch.logsumexp(log_kernel + log_column[None], dim=1)
        log_column = log_mass - torch.logsumexp(log_kernel + log_row[:, None], dim=0)
    return (log_kernel + log_row[:, None] + log_column[None]).exp()


@torch.no_grad()
def pair_latents_by_class(
    images: torch.Tensor,
    labels: torch.Tensor,
    latent,
    epsilon: float = 0.05,
    iterations: int = 50,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sample class Gaussians and round a class-wise Sinkhorn plan to a matching.

    The returned tensor is ordered like ``images``: each image receives exactly
    one sampled latent from the same class.  Rounding with a linear assignment
    preserves the Gaussian sample marginal much better than barycentric
    averaging, which would collapse samples toward each other.
    """
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels must have one entry per image")
    if images.device != labels.device:
        raise ValueError("images and labels must be on the same device")
    sampled = latent.sample(labels)
    paired = torch.empty_like(sampled)
    costs = []
    entropies = []

    # scipy is only needed for the small BxB rounding step and is imported
    # lazily so ordinary training/inference does not require it.
    from scipy.optimize import linear_sum_assignment

    for class_label in labels.unique(sorted=True):
        indices = torch.where(labels == class_label)[0]
        if indices.numel() == 1:
            paired[indices] = sampled[indices]
            continue
        class_images = images[indices]
        class_noise = sampled[indices]
        cost = _squared_cost(class_noise, class_images)
        plan = _sinkhorn_plan(cost, epsilon, iterations)
        # Round the entropic plan to a one-to-one assignment.  The assignment
        # maximizes transported mass, while preserving every sampled z.
        rows, columns = linear_sum_assignment((-plan).cpu().numpy())
        row_tensor = torch.as_tensor(rows, device=images.device)
        column_tensor = torch.as_tensor(columns, device=images.device)
        paired[indices[column_tensor]] = class_noise[row_tensor]
        costs.append(cost[row_tensor, column_tensor].mean())
        plan_safe = plan.clamp_min(1e-12)
        entropies.append(-(plan_safe * plan_safe.log()).sum())

    if costs:
        mean_cost = torch.stack(costs).mean()
        mean_entropy = torch.stack(entropies).mean()
    else:
        mean_cost = images.new_zeros(())
        mean_entropy = images.new_zeros(())
    return paired, {
        "ot_pair_cost": float(mean_cost),
        "ot_plan_entropy": float(mean_entropy),
        "ot_classes_in_batch": int(labels.unique().numel()),
    }


@torch.no_grad()
def path_diagnostics(
    state: torch.Tensor,
    target_velocity: torch.Tensor,
    predicted_velocity: torch.Tensor,
    time: torch.Tensor,
    bins: int = 10,
    neighbors: int = 4,
) -> Dict[str, object]:
    """Measure direction and local velocity ambiguity for one training batch.

    Neighbors are found after adaptive 8x8 average pooling, avoiding a large
    BxBxD image-space distance matrix.  The local statistics are a diagnostic,
    not an additional training loss.
    """
    batch = state.shape[0]
    if batch < 2:
        return {"path_diagnostics": []}
    neighbors = max(1, min(neighbors, batch - 1))
    state_features = F.adaptive_avg_pool2d(state.float(), (8, 8)).flatten(1)
    distances = torch.cdist(state_features, state_features)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.topk(neighbors, largest=False, dim=1).indices
    target_flat = target_velocity.float().flatten(1)
    predicted_flat = predicted_velocity.float().flatten(1)
    cosine = F.cosine_similarity(predicted_flat, target_flat, dim=1, eps=1e-8)
    local_targets = target_flat[nearest]
    local_cosine = F.cosine_similarity(
        target_flat[:, None, :], local_targets, dim=2, eps=1e-8
    )
    local_variance = (local_targets - target_flat[:, None, :]).square().mean(dim=(1, 2))
    conflict = 1.0 - local_cosine
    records = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (time >= lower) & (time < upper if index + 1 < bins else time <= upper)
        if not bool(mask.any()):
            continue
        records.append(
            {
                "t_start": lower,
                "t_end": upper,
                "count": int(mask.sum()),
                "velocity_direction_cosine": float(cosine[mask].mean()),
                "local_velocity_variance": float(local_variance[mask].mean()),
                "near_neighbor_direction_conflict": float(conflict[mask].mean()),
            }
        )
    return {"path_diagnostics": records}
