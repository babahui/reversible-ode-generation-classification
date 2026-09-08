from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .ot_pairing import path_diagnostics


def differentiable_reverse_integrate(
    model: nn.Module, initial: torch.Tensor, steps: int
) -> torch.Tensor:
    """Differentiably integrate a generation-only flow from image to noise."""
    if steps < 1:
        raise ValueError("steps must be positive")
    state = initial
    dt = -1.0 / float(steps)
    direction = torch.zeros_like(state) if getattr(model, "direction_adapter", False) else None
    for index in range(steps):
        time = 1.0 - index / float(steps)
        times = torch.full(
            (state.shape[0],), time, device=state.device, dtype=state.dtype
        )
        alpha = torch.stack((times, 1 - times), dim=1)
        if direction is None:
            velocity = model(state, alpha)
        else:
            velocity = model(state, alpha, recent_velocity=direction)
        state = state + dt * velocity
        # Keep the history causal without building a graph through all ODE
        # steps.  For Euler this is exactly the accepted physical velocity.
        if direction is not None:
            direction = velocity.detach()
    return state


class ConditionalFlowMatching(nn.Module):
    """Generation-only flow matching from class Gaussian noise to images."""

    def __init__(
        self,
        model: nn.Module,
        latent=None,
        center_loss_weight: float = 0.0,
        ode_class_weight: float = 0.0,
        ode_class_steps: int = 5,
        ode_class_batch: int = 64,
        ode_class_loss: str = "projection",
        ode_class_temperature: float = 1.0,
        direction_dropout: float = 0.0,
        direction_noise: float = 0.0,
    ):
        super().__init__()
        if getattr(model, "output_marginals", True):
            raise ValueError("generation-only model must output one velocity tensor")
        if getattr(model, "use_history", False):
            raise ValueError("generation-only flow must be Markovian")
        if center_loss_weight < 0:
            raise ValueError("center_loss_weight must be non-negative")
        if ode_class_weight < 0:
            raise ValueError("ode_class_weight must be non-negative")
        if ode_class_steps < 1 or ode_class_batch < 1:
            raise ValueError("ode class steps and batch must be positive")
        if ode_class_loss not in {"projection", "ce"}:
            raise ValueError("ode_class_loss must be 'projection' or 'ce'")
        if ode_class_temperature <= 0:
            raise ValueError("ode_class_temperature must be positive")
        if not 0 <= direction_dropout < 1:
            raise ValueError("direction_dropout must be in [0, 1)")
        if direction_noise < 0:
            raise ValueError("direction_noise must be non-negative")
        self.model = model
        self.latent = latent
        self.center_loss_weight = float(center_loss_weight)
        self.ode_class_weight = float(ode_class_weight)
        self.ode_class_steps = int(ode_class_steps)
        self.ode_class_batch = int(ode_class_batch)
        self.ode_class_loss = ode_class_loss
        self.ode_class_temperature = float(ode_class_temperature)
        self.direction_dropout = float(direction_dropout)
        self.direction_noise = float(direction_noise)

    def sample_state(
        self, images: torch.Tensor, noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.shape != noise.shape:
            raise ValueError("images and noise must have the same shape")
        time = torch.rand(
            images.shape[0], device=images.device, dtype=images.dtype
        )
        time_view = time[:, None, None, None]
        state = (1 - time_view) * noise + time_view * images
        # Marginal ordering remains [image, noise] for direct comparison.
        alpha = torch.stack((time, 1 - time), dim=1)
        target_velocity = images - noise
        return state, alpha, target_velocity

    def forward(
        self,
        images: torch.Tensor,
        noise: torch.Tensor,
        labels: torch.Tensor = None,
        diagnostics: bool = False,
        compute_ode_class: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        state, alpha, target_velocity = self.sample_state(images, noise)
        if getattr(self.model, "direction_adapter", False):
            direction = target_velocity
            if self.direction_noise > 0:
                direction = direction + self.direction_noise * torch.randn_like(direction)
            if self.direction_dropout > 0:
                keep = (
                    torch.rand(direction.shape[0], device=direction.device)
                    >= self.direction_dropout
                ).to(direction.dtype)
                direction = direction * keep[:, None, None, None]
            predicted_velocity = self.model(
                state, alpha, recent_velocity=direction
            )
        else:
            predicted_velocity = self.model(state, alpha)
        flow_loss = (predicted_velocity - target_velocity).square().mean()
        loss = flow_loss
        center_loss = None
        if self.center_loss_weight > 0:
            if self.latent is None or labels is None:
                raise ValueError(
                    "latent and labels are required when center_loss_weight > 0"
                )
            # Estimate the starting Gaussian endpoint from the learned velocity.
            time_view = alpha[:, 0, None, None, None]
            predicted_noise = state - time_view * predicted_velocity
            center_loss = self.latent.endpoint_projection_loss(
                predicted_noise, labels
            )
            loss = flow_loss + self.center_loss_weight * center_loss
        ode_class_loss = None
        if self.ode_class_weight > 0 and compute_ode_class:
            if self.latent is None or labels is None:
                raise ValueError(
                    "latent and labels are required when ode_class_weight > 0"
                )
            count = min(self.ode_class_batch, images.shape[0])
            encoded = differentiable_reverse_integrate(
                self.model, images[:count], self.ode_class_steps
            )
            if self.ode_class_loss == "ce":
                # Directly optimize the posterior over fixed Gaussian centers.
                # This gives a discriminative margin rather than only regressing
                # center coordinates.
                ode_class_loss = F.cross_entropy(
                    self.latent.center_coordinates(encoded)
                    / self.ode_class_temperature,
                    labels[:count],
                )
            else:
                ode_class_loss = self.latent.endpoint_projection_loss(
                    encoded, labels[:count]
                )
            loss = loss + self.ode_class_weight * ode_class_loss
        metrics = {
            "loss": float(loss.detach()),
            "flow_loss": float(flow_loss.detach()),
            "target_speed": float(
                target_velocity.square().mean().sqrt().detach()
            ),
        }
        if center_loss is not None:
            metrics["center_loss"] = float(center_loss.detach())
        if ode_class_loss is not None:
            metrics["ode_class_loss"] = float(ode_class_loss.detach())
        if diagnostics:
            metrics.update(
                path_diagnostics(
                    state.detach(),
                    target_velocity.detach(),
                    predicted_velocity.detach(),
                    alpha[:, 0].detach(),
                )
            )
        return loss, metrics


class GenerationOnlySampler:
    def __init__(self, model: nn.Module):
        if getattr(model, "output_marginals", True):
            raise ValueError("generation-only sampler requires a velocity model")
        self.model = model

    def _velocity(
        self,
        state: torch.Tensor,
        time: float,
        recent_velocity: torch.Tensor = None,
    ) -> torch.Tensor:
        times = torch.full(
            (state.shape[0],), time, device=state.device, dtype=state.dtype
        )
        alpha = torch.stack((times, 1 - times), dim=1)
        if getattr(self.model, "direction_adapter", False):
            if recent_velocity is None:
                recent_velocity = torch.zeros_like(state)
            return self.model(state, alpha, recent_velocity=recent_velocity)
        return self.model(state, alpha)

    @torch.inference_mode()
    def integrate(
        self,
        initial: torch.Tensor,
        steps: int = 30,
        method: str = "heun",
        return_path: bool = False,
    ) -> torch.Tensor:
        return self.integrate_interval(
            initial, 0.0, 1.0, steps, method, return_path
        )

    @torch.inference_mode()
    def integrate_interval(
        self,
        initial: torch.Tensor,
        t_start: float,
        t_end: float,
        steps: int = 30,
        method: str = "heun",
        return_path: bool = False,
    ) -> torch.Tensor:
        """Integrate the learned velocity on an arbitrary time interval.

        The same vector field supports generation (``0 -> 1``) and inverse
        classification (``1 -> 0``); reverse integration is obtained by the
        negative time step rather than by negating the model output.
        """
        if steps < 1:
            raise ValueError("steps must be positive")
        if t_start == t_end:
            raise ValueError("t_start and t_end must differ")
        if method not in {"euler", "heun"}:
            raise ValueError("method must be 'euler' or 'heun'")
        state = initial
        path = [state.clone()] if return_path else None
        dt = (float(t_end) - float(t_start)) / steps
        direction = (
            torch.zeros_like(state)
            if getattr(self.model, "direction_adapter", False)
            else None
        )
        for index in range(steps):
            time = float(t_start) + index * dt
            next_time = time + dt
            velocity = self._velocity(state, time, direction)
            if method == "heun":
                proposal = state + dt * velocity
                next_velocity = self._velocity(proposal, next_time, velocity)
                next_state = state + 0.5 * dt * (velocity + next_velocity)
            else:
                next_state = state + dt * velocity
            if direction is not None:
                direction = ((next_state - state) / dt).detach()
            state = next_state
            if path is not None:
                path.append(state.clone())
        return torch.stack(path, dim=1) if path is not None else state
