from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TrainingState:
    x: torch.Tensor
    alpha: torch.Tensor
    previous: torch.Tensor
    recent_velocity: torch.Tensor


def _vertices(indices: torch.Tensor, count: int, dtype: torch.dtype) -> torch.Tensor:
    return F.one_hot(indices, count).to(dtype=dtype)


class MultimarginalInterpolant(nn.Module):
    """Barycentric stochastic-interpolant objective.

    The network learns g_k(alpha, x) = E[x_k | x(alpha)=x]. Minimizing MSE
    against each sampled marginal has the same minimizer as the quadratic
    objective in Albergo et al. The optional history extends the Markov state.
    """

    def __init__(
        self,
        model: nn.Module,
        history_step: float = 0.02,
        history_dropout: float = 0.1,
        history_noise: float = 0.01,
        simplex_probability: float = 0.25,
    ):
        super().__init__()
        if not 0 <= history_dropout <= 1:
            raise ValueError("history_dropout must lie in [0, 1]")
        if not 0 <= simplex_probability <= 1:
            raise ValueError("simplex_probability must lie in [0, 1]")
        self.model = model
        self.history_step = history_step
        self.history_dropout = history_dropout
        self.history_noise = history_noise
        self.simplex_probability = simplex_probability

    @property
    def num_marginals(self) -> int:
        return self.model.num_marginals

    def sample_state(self, marginals: torch.Tensor) -> TrainingState:
        if marginals.ndim != 5:
            raise ValueError("marginals must have shape [B, K, C, H, W]")
        batch, count = marginals.shape[:2]
        if count != self.num_marginals:
            raise ValueError(
                f"expected {self.num_marginals} marginals, got {count}"
            )

        device, dtype = marginals.device, marginals.dtype
        source_index = torch.randint(count, (batch,), device=device)
        source = _vertices(source_index, count, dtype)

        target_index = torch.randint(count - 1, (batch,), device=device)
        target_index = target_index + (target_index >= source_index).long()
        edge_target = _vertices(target_index, count, dtype)

        concentration = torch.ones(batch, count, device=device, dtype=dtype)
        interior_target = torch.distributions.Dirichlet(concentration).sample()
        use_simplex = (
            torch.rand(batch, 1, device=device) < self.simplex_probability
        )
        target = torch.where(use_simplex, interior_target, edge_target)

        progress = torch.rand(batch, 1, device=device, dtype=dtype)
        previous_progress = (progress - self.history_step).clamp_min(0)
        alpha = (1 - progress) * source + progress * target
        previous_alpha = (
            (1 - previous_progress) * source + previous_progress * target
        )

        x = torch.einsum("bk,bkchw->bchw", alpha, marginals)
        previous = torch.einsum("bk,bkchw->bchw", previous_alpha, marginals)
        elapsed = (progress - previous_progress).clamp_min(1e-6)
        recent_velocity = (x - previous) / elapsed[:, :, None, None]
        recent_velocity = torch.where(
            (progress > 0)[:, :, None, None], recent_velocity, torch.zeros_like(x)
        )

        if self.history_noise > 0:
            previous = previous + torch.randn_like(previous) * self.history_noise
        if self.history_dropout > 0:
            keep = (
                torch.rand(batch, 1, 1, 1, device=device) >= self.history_dropout
            ).to(dtype)
            previous = previous * keep
            recent_velocity = recent_velocity * keep

        return TrainingState(x, alpha, previous, recent_velocity)

    def forward(self, marginals: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        state = self.sample_state(marginals)
        prediction = self.model(
            state.x, state.alpha, state.previous, state.recent_velocity
        )
        per_marginal = (prediction - marginals).square().mean(dim=(0, 2, 3, 4))
        loss = per_marginal.mean()
        metrics = {
            "loss": float(loss.detach()),
            "history_speed": float(state.recent_velocity.square().mean().sqrt().detach()),
        }
        return loss, metrics


class TransportSampler:
    def __init__(self, model: nn.Module):
        self.model = model

    def _velocity(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        previous: torch.Tensor,
        recent_velocity: torch.Tensor,
        alpha_dot: torch.Tensor,
    ) -> torch.Tensor:
        marginal_means = self.model(x, alpha, previous, recent_velocity)
        return torch.einsum("bk,bkchw->bchw", alpha_dot, marginal_means)

    @torch.inference_mode()
    def integrate(
        self,
        initial: torch.Tensor,
        source: int,
        target: int,
        steps: int = 50,
        method: str = "heun",
        return_path: bool = False,
    ) -> torch.Tensor:
        if source == target:
            raise ValueError("source and target must differ")
        if not 0 <= source < self.model.num_marginals:
            raise ValueError("invalid source marginal")
        if not 0 <= target < self.model.num_marginals:
            raise ValueError("invalid target marginal")
        if steps < 1:
            raise ValueError("steps must be positive")
        if method not in {"euler", "heun"}:
            raise ValueError("method must be 'euler' or 'heun'")

        progress = torch.linspace(
            0, 1, steps + 1, device=initial.device, dtype=initial.dtype
        )
        alpha_path = torch.zeros(
            steps + 1,
            self.model.num_marginals,
            device=initial.device,
            dtype=initial.dtype,
        )
        alpha_path[:, source] = 1 - progress
        alpha_path[:, target] = progress
        return self.integrate_path(initial, alpha_path, method, return_path)

    @torch.inference_mode()
    def integrate_path(
        self,
        initial: torch.Tensor,
        alpha_path: torch.Tensor,
        method: str = "heun",
        return_path: bool = False,
    ) -> torch.Tensor:
        """Integrate along an arbitrary piecewise-linear simplex path."""
        if alpha_path.ndim != 2 or alpha_path.shape[1] != self.model.num_marginals:
            raise ValueError(
                f"alpha_path must have shape [steps + 1, {self.model.num_marginals}]"
            )
        if alpha_path.shape[0] < 2:
            raise ValueError("alpha_path must contain at least two coordinates")
        if method not in {"euler", "heun"}:
            raise ValueError("method must be 'euler' or 'heun'")
        if torch.any(alpha_path < -1e-6) or not torch.allclose(
            alpha_path.sum(dim=1), torch.ones_like(alpha_path[:, 0]), atol=1e-5
        ):
            raise ValueError("every alpha_path row must lie on the probability simplex")

        alpha_path = alpha_path.to(device=initial.device, dtype=initial.dtype)
        steps = alpha_path.shape[0] - 1

        x = initial
        # At the source vertex there is no earlier motion, so the causal
        # previous state is the source itself and its recent velocity is zero.
        previous = x.clone()
        recent_velocity = torch.zeros_like(x)
        path = [x.clone()] if return_path else None
        dt = 1.0 / steps

        for index in range(steps):
            alpha = alpha_path[index].expand(x.shape[0], -1)
            next_alpha = alpha_path[index + 1].expand(x.shape[0], -1)
            alpha_dot = (next_alpha - alpha) / dt
            velocity = self._velocity(
                x, alpha, previous, recent_velocity, alpha_dot
            )

            if method == "heun":
                proposal = x + dt * velocity
                proposal_velocity = self._velocity(
                    proposal, next_alpha, x, velocity, alpha_dot
                )
                next_x = x + 0.5 * dt * (velocity + proposal_velocity)
            else:
                next_x = x + dt * velocity

            next_recent_velocity = (next_x - x) / dt
            previous, x = x, next_x
            recent_velocity = next_recent_velocity
            if path is not None:
                path.append(x.clone())

        return torch.stack(path, dim=1) if path is not None else x


def differentiable_edge_integrate(
    model: nn.Module,
    initial: torch.Tensor,
    source: int,
    target: int,
    steps: int,
) -> torch.Tensor:
    """Small Euler solve used only for occasional endpoint supervision."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if source == target:
        raise ValueError("source and target must differ")
    if getattr(model, "use_history", False):
        raise ValueError("differentiable endpoint supervision requires a Markov model")
    dt = 1.0 / steps
    x = initial
    for index in range(steps):
        progress = index / steps
        alpha = torch.zeros(
            x.shape[0], model.num_marginals, device=x.device, dtype=x.dtype
        )
        alpha[:, source] = 1 - progress
        alpha[:, target] = progress
        marginal_means = model(x, alpha)
        x = x + dt * (marginal_means[:, target] - marginal_means[:, source])
    return x
