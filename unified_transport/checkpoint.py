from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .latent import OrthogonalGaussianLatent
from .model import MarginalUNet


class ExponentialMovingAverage:
    def __init__(
        self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 1000
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        model_state = model.state_dict()
        for name, value in self.model.state_dict().items():
            source = model_state[name].detach()
            if self.updates <= self.warmup_steps or not value.is_floating_point():
                value.copy_(source)
            else:
                value.lerp_(source, 1 - self.decay)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "updates": self.updates,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.updates = int(state.get("updates", 0))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))


def build_model(config: Dict[str, Any]) -> MarginalUNet:
    return MarginalUNet(
        image_channels=int(config["image_channels"]),
        num_marginals=int(config["num_marginals"]),
        base_channels=int(config["base_channels"]),
        channel_mults=tuple(config["channel_mults"]),
        use_history=bool(config["use_history"]),
        condition_on_alpha=bool(config.get("condition_on_alpha", True)),
        output_marginals=config.get("task", "unified")
        not in {"generation_only", "label_conditioned_generation"},
        class_conditioning=bool(config.get("class_conditioning", False)),
        num_classes=int(config.get("num_classes", 0)),
        direction_adapter=bool(config.get("direction_adapter", False)),
    )


def build_latent(config: Dict[str, Any]) -> Optional[OrthogonalGaussianLatent]:
    if config["mode"] not in {"mnist", "cifar10", "tinyimagenet", "stl10"}:
        return None
    return OrthogonalGaussianLatent(
        num_classes=int(config["num_classes"]),
        image_shape=(config["image_channels"], config["image_size"], config["image_size"]),
        center_scale=float(config.get("center_scale", 4.0)),
        sigma=float(config.get("latent_sigma", 0.5)),
        seed=int(config["seed"]),
        center_mode=str(config.get("center_mode", "random")),
    )


def save_checkpoint(
    path: str,
    config: Dict[str, Any],
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    step: int,
    latent: Optional[OrthogonalGaussianLatent] = None,
    scaler: Optional[Any] = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "config": config,
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "latent": latent.state_dict() if latent is not None else None,
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_for_inference(
    path: str, device: torch.device, use_ema: bool = True
) -> Tuple[nn.Module, Optional[OrthogonalGaussianLatent], Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    state = checkpoint["ema"]["model"] if use_ema else checkpoint["model"]
    model.load_state_dict(state)
    model.eval()
    latent = build_latent(config)
    if latent is not None and checkpoint.get("latent") is not None:
        latent.load_state_dict(checkpoint["latent"])
        latent = latent.to(device)
    return model, latent, checkpoint
