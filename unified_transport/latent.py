import math
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class OrthogonalGaussianLatent(nn.Module):
    """A continuous class-conditional latent mixture with orthogonal means."""

    def __init__(
        self,
        num_classes: int,
        image_shape: Sequence[int],
        center_scale: float = 4.0,
        sigma: float = 0.5,
        seed: int = 0,
        center_mode: str = "random",
        priors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        dimension = math.prod(image_shape)
        if num_classes > dimension:
            raise ValueError("num_classes cannot exceed the flattened latent dimension")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if center_mode not in {"random", "low_frequency", "coded_low_frequency"}:
            raise ValueError(
                "center_mode must be 'random', 'low_frequency', or "
                "'coded_low_frequency'"
            )

        generator = torch.Generator().manual_seed(seed)
        if center_mode == "random":
            basis = torch.randn(dimension, num_classes, generator=generator)
            orthogonal, _ = torch.linalg.qr(basis, mode="reduced")
            centers = orthogonal.T.reshape(num_classes, *image_shape)
        else:
            channels, height, width = image_shape
            # Low-frequency separable DCT modes survive UNet downsampling.
            yy = torch.arange(height, dtype=torch.float32)[:, None]
            xx = torch.arange(width, dtype=torch.float32)[None, :]
            modes = []
            frequency_pairs = [
                (u, v) for u in range(4) for v in range(4) if not (u == 3 and v == 3)
            ]
            max_modes = len(frequency_pairs) + (
                1 if center_mode == "coded_low_frequency" else 0
            )
            if num_classes > max_modes:
                raise ValueError(
                    "low_frequency mode supports at most "
                    f"{max_modes} classes"
                )
            mode_pairs = list(frequency_pairs)
            if center_mode == "coded_low_frequency":
                # The sixteenth 4x4 mode completes a power-of-two codebook.
                mode_pairs.append((3, 3))
            for index, (u, v) in enumerate(mode_pairs):
                pattern = torch.cos(torch.pi * (yy + 0.5) * u / height)
                pattern = pattern * torch.cos(torch.pi * (xx + 0.5) * v / width)
                basis = torch.zeros(image_shape, dtype=torch.float32)
                basis[index % channels] = pattern
                modes.append(basis.flatten())
            mode_basis = F.normalize(torch.stack(modes, dim=0), dim=1)
            if center_mode == "low_frequency":
                centers = mode_basis[:num_classes]
            else:
                # Spread each class code over all robust low-frequency modes.
                # Complete Hadamard rows give orthogonal, redundant codewords
                # for the ten classes when all sixteen modes are available.
                hadamard = torch.ones(1, 1)
                while hadamard.shape[0] < mode_basis.shape[0] + 1:
                    hadamard = torch.cat(
                        (
                            torch.cat((hadamard, hadamard), dim=1),
                            torch.cat((hadamard, -hadamard), dim=1),
                        ),
                        dim=0,
                    )
                code = hadamard[:num_classes, : mode_basis.shape[0]]
                centers = F.normalize(code @ mode_basis, dim=1)
            centers = centers.reshape(num_classes, *image_shape)
        centers = centers * center_scale
        if priors is None:
            priors = torch.full((num_classes,), 1.0 / num_classes)
        priors = priors.float() / priors.sum()

        self.num_classes = num_classes
        self.image_shape = tuple(image_shape)
        self.sigma = float(sigma)
        self.center_mode = center_mode
        self.register_buffer("centers", centers)
        self.register_buffer("log_priors", priors.log())

    def sample(
        self, labels: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        centers = self.centers[labels]
        noise = torch.randn(
            centers.shape,
            device=centers.device,
            dtype=centers.dtype,
            generator=generator,
        )
        return centers + self.sigma * noise

    def log_posterior(self, latent: torch.Tensor) -> torch.Tensor:
        difference = latent[:, None] - self.centers[None]
        squared_distance = difference.flatten(2).square().sum(dim=2)
        return -0.5 * squared_distance / (self.sigma**2) + self.log_priors[None]

    def classify(self, latent: torch.Tensor) -> torch.Tensor:
        return self.log_posterior(latent).argmax(dim=1)

    def center_coordinates(self, latent: torch.Tensor) -> torch.Tensor:
        """Project a latent tensor onto unit vectors for the class centers."""
        flat_centers = self.centers.flatten(1).to(latent.device)
        unit_centers = F.normalize(flat_centers, dim=1)
        return latent.flatten(1).float() @ unit_centers.float().T

    def endpoint_projection_loss(
        self, predicted_latent: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Reweight errors in the low-dimensional class-center subspace."""
        predicted = self.center_coordinates(predicted_latent)
        flat_centers = self.centers.flatten(1).to(predicted_latent.device)
        unit_centers = F.normalize(flat_centers, dim=1)
        target_latents = flat_centers[labels.to(predicted_latent.device)]
        target = target_latents.float() @ unit_centers.float().T
        return F.mse_loss(predicted, target)
