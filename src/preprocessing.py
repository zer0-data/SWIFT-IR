"""Preprocessing modules for thermal weather mitigation.

Provides two nn.Module classes:

- FogEnhancer: adaptive gamma correction based on image mean.
- LSRB: lightweight spatial residual block for rain removal.

Both modules accept single-channel thermal images with values in [0,1].
"""
from typing import Optional

import torch
import torch.nn as nn


class FogEnhancer(nn.Module):
    """Adaptive gamma-based fog enhancer.

    Computes the per-sample mean intensity \u03bc and adaptive gamma
    \u03b3 = -log(0.5) / log(\u03bc + eps), then applies elementwise
    power transform: I_out = I_in ** \u03b3.

    The implementation is fully differentiable (uses torch ops) and
    clamps \u03bc to avoid numerical issues.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B,1,H,W), (B,H,W) or (H,W). Values should be in [0,1].

        Returns:
            Tensor of same shape as input with fog-enhanced output.
        """
        orig_shape = x.shape
        # Normalize to (B,1,H,W)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            # Could be (B,H,W) or (1,H,W)
            if x.shape[0] == 1:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError("Unsupported tensor shape for FogEnhancer")

        x = x.to(dtype=torch.float32)

        # compute per-sample mean over channel and spatial dims -> shape (B,1,1,1)
        mu = x.mean(dim=(1, 2, 3), keepdim=True)

        # clamp to avoid division by zero or log(0)
        mu = mu.clamp(min=self.eps, max=1.0 - self.eps)

        half = torch.tensor(0.5, device=x.device, dtype=x.dtype)
        gamma = -torch.log(half) / torch.log(mu + self.eps)

        # Apply gamma correction (differentiable)
        out = torch.pow(x, gamma)

        # Restore original shape
        if len(orig_shape) == 2:
            out = out.squeeze(0).squeeze(0)
        elif len(orig_shape) == 3 and orig_shape[0] != 1:
            out = out.squeeze(1)

        return out


class LSRB(nn.Module):
    """Lightweight Spatial Residual Block for rain-streak removal.

    Architecture:
        Conv2d(1->16, kernel=3, padding=1)
        Depthwise Conv2d(16->16, kernel=3, padding=1, groups=16)
        Pointwise Conv2d(16->1, kernel=1)

    Forward: rain_mask = conv_pw(conv_dw(ReLU(conv)))
             output = input - sigmoid(rain_mask)

    The module preserves input shape and expects single-channel inputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # depthwise convolution implemented with groups=channels
        self.depthwise = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
        self.pointwise = nn.Conv2d(16, 1, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B,1,H,W), (B,H,W) or (H,W).

        Returns:
            Tensor of same shape as input with rain removed: x - sigmoid(mask)
        """
        orig_shape = x.shape
        # Normalize to (B,1,H,W)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            if x.shape[0] == 1:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError("Unsupported tensor shape for LSRB")

        x_in = x
        out = self.conv(x_in)
        out = self.activation(out)
        out = self.depthwise(out)
        out = self.activation(out)
        out = self.pointwise(out)
        mask = torch.sigmoid(out)

        result = x_in - mask

        # Restore original shape
        if len(orig_shape) == 2:
            result = result.squeeze(0).squeeze(0)
        elif len(orig_shape) == 3 and orig_shape[0] != 1:
            result = result.squeeze(1)

        return result


__all__ = ["FogEnhancer", "LSRB"]

