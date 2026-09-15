"""Shared, explicitly configured RF model, prior and ODE for the strong baseline study."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .vendor_unet.unet import UNetModelWrapper
from unified_transport.latent import OrthogonalGaussianLatent


ARCH = dict(dim=(3, 32, 32), num_channels=128, num_res_blocks=2,
            channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
            attention_resolutions="16", dropout=0.1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(kind="label", use_checkpoint=False):
    # Construct a common trunk before the optional label embedding, so seed-matched
    # experiments start with exactly the same trunk parameters.
    model = UNetModelWrapper(**ARCH, class_cond=False, use_checkpoint=use_checkpoint)
    if kind in {"label", "label_ot"}:
        model.num_classes = 10
        model.label_emb = torch.nn.Embedding(10, 512)
    return model


def build_prior(config, device):
    if not config.get("kind", "unconditional").startswith("gaussian"):
        return None
    return OrthogonalGaussianLatent(10, (3, 32, 32),
        center_scale=config.get("center_scale", 12.0),
        sigma=config.get("sigma", 0.5), center_mode="low_frequency", seed=0).to(device)


def load_model(path, device="cuda"):
    checkpoint = torch.load(path, map_location="cpu")
    if "config" in checkpoint and "ema" in checkpoint:
        config = checkpoint["config"]
        state = checkpoint["ema"]
    else:
        config = {"kind": "unconditional", "source": "official_torchcfm_icfm"}
        state = checkpoint["ema_model"]
    model = build_model(config["kind"])
    state = {k.removeprefix("module.") if hasattr(k, "removeprefix") else
             (k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    step = int(checkpoint.get("step", 400000))
    del checkpoint, state
    return model, config, build_prior(config, device), step


def initial_noise(labels, config, prior, generator=None):
    if prior is not None:
        return prior.sample(labels, generator=generator)
    return torch.randn(len(labels), 3, 32, 32, device=labels.device,
                       generator=generator) * config.get("sigma", 1.0)


def integrate(model, initial, labels=None, method="heun", steps=30,
              reverse=False, tol=1e-5):
    """Differentiable fixed-grid solves; forward and reverse share one vector field.

    No clipping within the ODE. Callers choose whether to retain gradients.
    """
    nfe = 0
    def velocity(t, x):
        nonlocal nfe
        nfe += 1
        tt = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        return model(tt, x, y=labels)
    start, end = (1., 0.) if reverse else (0., 1.)
    if method == "dopri5":
        from torchdiffeq import odeint
        trajectory = odeint(velocity, initial,
            torch.tensor([start, end], device=initial.device),
            method="dopri5", atol=tol, rtol=tol)
        return trajectory[-1], nfe
    if steps < 1 or method not in {"heun", "euler"}:
        raise ValueError("invalid ODE settings")
    x = initial
    dt = (end - start) / steps
    for index in range(steps):
        t = start + index * dt
        v = velocity(t, x)
        if method == "heun":
            w = velocity(start + (index + 1) * dt, x + dt * v)
            x = x + dt * 0.5 * (v + w)
        else:
            x = x + dt * v
    return x, nfe


def to_uint8(x):
    # Exactly the conversion in the authors' compute_fid.py.
    return (x * 127.5 + 128).clamp(0, 255).to(torch.uint8)
