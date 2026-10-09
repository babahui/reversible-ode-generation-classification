"""Frozen matching K7 encoder/decoder with official latent normalization."""
import hashlib
import os
from pathlib import Path
import torch
from .model import upstream_path
from .flow import autocast
from .config import LATENT_SHAPE

WEIGHT_SHA256 = {
    'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth': '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035',
    'decoder.pt': '779871dc7e81d4c31cc5dc80824760c79b40e4a8b90e0d743d1e79c10d515d87',
    'stats.pt': '40e57d9d38a267dc258043c081094d276382c29ebccc1bd8c38f4dd11e81ba77',
}


def verify_weights(weights):
    for name, expected in WEIGHT_SHA256.items():
        path = Path(weights) / name
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Wrong or incomplete RAE weight file: {path}')
    return dict(WEIGHT_SHA256)


def stats(weights, device):
    d = torch.load(Path(weights) / 'stats.pt', map_location=device, weights_only=True)
    mean, var = d['mean'].float(), d['var'].float()
    if mean.shape != LATENT_SHAPE or var.shape != LATENT_SHAPE:
        raise ValueError('Wrong latent statistics shape')
    if not torch.isfinite(mean).all() or not torch.isfinite(var).all() or (var < 0).any():
        raise ValueError('Invalid latent statistics')
    return mean, (var + 1e-5).sqrt()


class Encoder:
    def __init__(self, rae_repo, dino_repo, weights, device):
        upstream_path(rae_repo)
        os.environ['DINOV3_REPO_DIR'] = str(Path(dino_repo).resolve())
        os.environ['DINOV3_CKPT_DIR'] = str(Path(weights).resolve())
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        from encoders.vision_encoder import create_encoder
        self.device = torch.device(device)
        self.model = create_encoder('dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]',
                                    self.device, resolution=256)
        self.model.requires_grad_(False).eval()
        self.mean, self.std = stats(weights, self.device)

    @torch.no_grad()
    def __call__(self, images):
        with autocast(self.device):
            tokens = self.model(images.to(self.device, non_blocking=True) * 255.)
        if tuple(tokens.shape[1:]) != (256, 1024):
            raise ValueError(f'Unexpected encoder output: {tokens.shape}')
        z = tokens.float().transpose(1, 2).reshape(-1, *LATENT_SHAPE)
        return (z - self.mean) / self.std


class Decoder:
    def __init__(self, rae_repo, weights, device):
        upstream_path(rae_repo)
        from transformers import AutoConfig
        from stage1.decoders import GeneralDecoder
        self.device = torch.device(device)
        config = AutoConfig.from_pretrained(str(Path(rae_repo) / 'configs/decoder/ViTXL'), local_files_only=True)
        config.hidden_size, config.patch_size, config.image_size = 1024, 16, 256
        self.model = GeneralDecoder(config, num_patches=256)
        self.model.load_state_dict(torch.load(Path(weights) / 'decoder.pt', map_location='cpu', weights_only=True), strict=True)
        self.model.requires_grad_(False).eval().to(self.device)
        self.mean, self.std = stats(weights, self.device)

    @torch.no_grad()
    def __call__(self, z):
        tokens = (z.float() * self.std + self.mean).flatten(2).transpose(1, 2)
        with autocast(self.device):
            prediction = self.model(tokens, drop_cls_token=False).logits
        return self.model.unpatchify(prediction).float().clamp(0, 1)
