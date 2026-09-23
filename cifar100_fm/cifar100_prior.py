"""Fixed 100-class DCT Gaussian prior; RGB x 6 x 6, ordered by frequency."""
import torch
from torch import nn
import torch.nn.functional as F


def frequency_modes():
    return sorted([(c, u, v) for c in range(3) for u in range(6) for v in range(6)],
                  key=lambda m: (m[1]**2 + m[2]**2, m[1], m[2], m[0]))[:100]


class Cifar100LowFrequencyPrior(nn.Module):
    def __init__(self, center_scale=12., sigma=.5):
        super().__init__()
        if center_scale <= 0 or sigma <= 0:
            raise ValueError('positive scale and sigma required')
        self.sigma = float(sigma)
        self.num_classes = 100
        self.image_shape = (3, 32, 32)
        self.modes = frequency_modes()
        yy = torch.arange(32, dtype=torch.float64)[:, None]
        xx = torch.arange(32, dtype=torch.float64)[None, :]
        centers = []
        for channel, u, v in self.modes:
            basis = torch.zeros(3, 32, 32, dtype=torch.float64)
            basis[channel] = (torch.cos(torch.pi * (yy + .5) * u / 32)
                              * torch.cos(torch.pi * (xx + .5) * v / 32))
            centers.append(basis / basis.norm() * center_scale)
        self.register_buffer('centers', torch.stack(centers).float())
        self.register_buffer('log_priors', torch.full((100,), 1/100).log())

    def sample(self, labels, generator=None):
        centers = self.centers[labels]
        noise = torch.randn(centers.shape, device=centers.device,
                            dtype=centers.dtype, generator=generator)
        return centers + self.sigma * noise

    def center_coordinates(self, latent):
        return latent.flatten(1).float() @ F.normalize(self.centers.flatten(1), dim=1).T

    def log_posterior(self, latent):
        flat = latent.flatten(1).float()
        centers = self.centers.flatten(1)
        # Omit the common -||z||²/(2 sigma²); identical posterior softmax/argmax.
        return (flat @ centers.T - .5 * centers.square().sum(1)) / self.sigma**2 + self.log_priors

    def classify(self, latent):
        return self.log_posterior(latent).argmax(1)
