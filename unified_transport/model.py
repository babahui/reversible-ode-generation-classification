import math
from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class AlphaEmbedding(nn.Module):
    """Embed a simplex coordinate without assigning a privileged time axis."""

    def __init__(self, num_marginals: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_marginals, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.net(alpha)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.cond = nn.Linear(cond_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class MarginalUNet(nn.Module):
    """Predict all marginal conditional means g_k(alpha, x, history).

    History is represented by the previous state and the most recent finite-
    difference velocity. This has constant memory in the number of ODE steps.
    """

    def __init__(
        self,
        image_channels: int,
        num_marginals: int,
        base_channels: int = 64,
        channel_mults: Iterable[int] = (1, 2, 4),
        use_history: bool = True,
        condition_on_alpha: bool = True,
        output_marginals: bool = True,
        class_conditioning: bool = False,
        num_classes: int = 0,
        direction_adapter: bool = False,
    ):
        super().__init__()
        mults = tuple(channel_mults)
        if not mults:
            raise ValueError("channel_mults must not be empty")
        if num_marginals < 2:
            raise ValueError("num_marginals must be at least 2")
        if use_history and direction_adapter:
            raise ValueError("use_history and direction_adapter are mutually exclusive")

        self.image_channels = image_channels
        self.num_marginals = num_marginals
        self.use_history = use_history
        self.condition_on_alpha = condition_on_alpha
        self.output_marginals = output_marginals
        self.class_conditioning = class_conditioning
        self.num_classes = num_classes
        # The direction adapter is an additive, zero-initialized branch.  It
        # preserves the original Markov U-Net exactly when no direction is
        # supplied and lets a fine-tune learn only a local tangent correction.
        self.direction_adapter = bool(direction_adapter)
        input_multiplier = 3 if use_history else 1
        widths = [base_channels * mult for mult in mults]
        cond_dim = base_channels * 4
        if class_conditioning:
            if num_classes < 1:
                raise ValueError("num_classes must be positive with class conditioning")
            self.class_embedding = nn.Embedding(num_classes, cond_dim)

        self.alpha_embedding = AlphaEmbedding(num_marginals, cond_dim)
        self.input_conv = nn.Conv2d(
            image_channels * input_multiplier, widths[0], 3, padding=1
        )
        if self.direction_adapter:
            self.direction_adapter_net = nn.Sequential(
                nn.Conv2d(image_channels, widths[0], 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(widths[0], widths[0], 3, padding=1),
            )
            # At initialization the adapter is an exact no-op, so loading a
            # Markov checkpoint gives a bitwise-compatible baseline forward.
            nn.init.zeros_(self.direction_adapter_net[-1].weight)
            nn.init.zeros_(self.direction_adapter_net[-1].bias)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = widths[0]
        for index, width in enumerate(widths):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(current, width, cond_dim),
                        ResidualBlock(width, width, cond_dim),
                    ]
                )
            )
            current = width
            if index < len(widths) - 1:
                self.downsamples.append(nn.Conv2d(width, widths[index + 1], 4, 2, 1))
                current = widths[index + 1]

        self.middle = nn.ModuleList(
            [ResidualBlock(current, current, cond_dim), ResidualBlock(current, current, cond_dim)]
        )

        self.up_blocks = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(current + width, width, cond_dim),
                        ResidualBlock(width, width, cond_dim),
                    ]
                )
            )
            current = width

        self.output_norm = nn.GroupNorm(_group_count(current), current)
        self.output_conv = nn.Conv2d(
            current,
            (num_marginals if output_marginals else 1) * image_channels,
            3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        previous: Optional[torch.Tensor] = None,
        recent_velocity: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if alpha.shape != (x.shape[0], self.num_marginals):
            raise ValueError(
                f"alpha must have shape [B, {self.num_marginals}], got {tuple(alpha.shape)}"
            )

        if self.use_history:
            previous = torch.zeros_like(x) if previous is None else previous
            recent_velocity = (
                torch.zeros_like(x) if recent_velocity is None else recent_velocity
            )
            if previous.shape != x.shape or recent_velocity.shape != x.shape:
                raise ValueError("history tensors must have the same shape as x")
            x = torch.cat((x, previous, recent_velocity), dim=1)

        if self.direction_adapter:
            recent_velocity = (
                torch.zeros_like(x) if recent_velocity is None else recent_velocity
            )
            if recent_velocity.shape != x.shape:
                raise ValueError("recent_velocity must have the same shape as x")

        cond = self.alpha_embedding(alpha)
        if not self.condition_on_alpha:
            cond = torch.zeros_like(cond)
        if self.class_conditioning:
            if labels is None or labels.shape != (x.shape[0],):
                raise ValueError("class-conditioned model requires labels with shape [B]")
            cond = cond + self.class_embedding(labels.to(dtype=torch.long))
        h = self.input_conv(x)
        if self.direction_adapter:
            h = h + self.direction_adapter_net(recent_velocity)
        skips = []
        for index, blocks in enumerate(self.down_blocks):
            h = blocks[0](h, cond)
            h = blocks[1](h, cond)
            skips.append(h)
            if index < len(self.downsamples):
                h = self.downsamples[index](h)

        h = self.middle[0](h, cond)
        h = self.middle[1](h, cond)

        for blocks, skip in zip(self.up_blocks, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat((h, skip), dim=1)
            h = blocks[0](h, cond)
            h = blocks[1](h, cond)

        output = self.output_conv(F.silu(self.output_norm(h)))
        if not self.output_marginals:
            return output
        return output.view(
            output.shape[0],
            self.num_marginals,
            self.image_channels,
            output.shape[-2],
            output.shape[-1],
        )
