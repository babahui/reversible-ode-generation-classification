from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class MNISTReferenceClassifier(nn.Module):
    """Small independent MNIST classifier used only for evaluation."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.projection = nn.Linear(64 * 7 * 7, feature_dim)
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(
        self, images: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        hidden = self.features(images).flatten(1)
        features = F.silu(self.projection(hidden))
        if return_features:
            return features
        return self.classifier(features)


class CifarResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(images)))
        hidden = self.norm2(self.conv2(hidden))
        return F.silu(hidden + self.skip(images))


class CIFAR10ReferenceClassifier(nn.Module):
    """CIFAR-sized residual classifier used only for transport evaluation."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            CifarResidualBlock(64, 64),
            CifarResidualBlock(64, 64),
            CifarResidualBlock(64, 128, stride=2),
            CifarResidualBlock(128, 128),
            CifarResidualBlock(128, 256, stride=2),
            CifarResidualBlock(256, 256),
        )
        self.projection = nn.Linear(256, feature_dim)
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(
        self, images: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        hidden = self.blocks(self.stem(images)).mean(dim=(2, 3))
        features = F.silu(self.projection(hidden))
        if return_features:
            return features
        return self.classifier(features)


class ImageNetResNet18Classifier(nn.Module):
    """ImageNet-pretrained evaluator for medium-resolution RGB datasets."""

    architecture = "resnet18"

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.register_buffer(
            "input_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "input_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(
        self, images: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        images = ((images + 1.0) * 0.5 - self.input_mean) / self.input_std
        features = self.backbone(images)
        if return_features:
            return features
        return self.classifier(features)


def save_reference_classifier(
    path: str,
    model: nn.Module,
    epoch: int,
    accuracy: float,
    history: Dict[str, Any],
    mode: str = "mnist",
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "feature_dim": model.feature_dim,
        "epoch": epoch,
        "accuracy": accuracy,
        "history": history,
        "mode": mode,
        "architecture": getattr(model, "architecture", "small_resnet"),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_reference_classifier(
    path: str, device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    mode = checkpoint.get("mode", "mnist")
    architecture = checkpoint.get("architecture", "small_resnet")
    if architecture == "resnet18":
        model = ImageNetResNet18Classifier(pretrained=False).to(device)
    else:
        model_class = MNISTReferenceClassifier if mode == "mnist" else CIFAR10ReferenceClassifier
        model = model_class(int(checkpoint.get("feature_dim", 128))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return model, checkpoint
