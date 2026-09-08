from typing import Dict, Tuple

import torch
from torch import nn


class LabelConditionalFlowMatching(nn.Module):
    """Rectified flow objective with standard Gaussian noise and label input."""

    def __init__(self, model: nn.Module):
        super().__init__()
        if not getattr(model, "class_conditioning", False):
            raise ValueError("model must have class conditioning enabled")
        if getattr(model, "output_marginals", True):
            raise ValueError("label baseline must output one velocity tensor")
        self.model = model

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor, noise: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        time = torch.rand(images.shape[0], device=images.device, dtype=images.dtype)
        time_view = time[:, None, None, None]
        state = (1 - time_view) * noise + time_view * images
        alpha = torch.stack((time, 1 - time), dim=1)
        target = images - noise
        prediction = self.model(state, alpha, labels=labels)
        loss = (prediction - target).square().mean()
        return loss, {
            "loss": float(loss.detach()),
            "target_speed": float(target.square().mean().sqrt().detach()),
        }


class LabelConditionalSampler:
    def __init__(self, model: nn.Module):
        if not getattr(model, "class_conditioning", False):
            raise ValueError("model must have class conditioning enabled")
        self.model = model

    def _velocity(self, state: torch.Tensor, time: float, labels: torch.Tensor):
        times = torch.full(
            (state.shape[0],), time, device=state.device, dtype=state.dtype
        )
        alpha = torch.stack((times, 1 - times), dim=1)
        return self.model(state, alpha, labels=labels)

    @torch.inference_mode()
    def integrate(
        self,
        initial: torch.Tensor,
        labels: torch.Tensor,
        steps: int = 30,
        method: str = "heun",
        return_path: bool = False,
    ) -> torch.Tensor:
        if initial.shape[0] != labels.shape[0]:
            raise ValueError("initial and labels must have the same batch size")
        if steps < 1:
            raise ValueError("steps must be positive")
        if method not in {"euler", "heun"}:
            raise ValueError("method must be 'euler' or 'heun'")
        state = initial
        path = [state.clone()] if return_path else None
        dt = 1.0 / steps
        for index in range(steps):
            time = index / steps
            velocity = self._velocity(state, time, labels)
            if method == "heun":
                proposal = state + dt * velocity
                next_velocity = self._velocity(
                    proposal, (index + 1) / steps, labels
                )
                state = state + 0.5 * dt * (velocity + next_velocity)
            else:
                state = state + dt * velocity
            if path is not None:
                path.append(state.clone())
        return torch.stack(path, dim=1) if path is not None else state
