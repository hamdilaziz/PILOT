from __future__ import annotations

import math
import random
from contextlib import nullcontext
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import pad


__all__ = [
    "PositionalEncoding2D",
    "MixDropout",
    "ConvBlock",
    "DepthSepConv2D",
    "DSCBlockOriginal",
    "ChannelAttention",
    "SpatialAttention",
    "PILOTEncoder",
]


def _autocast_disabled(device_type: str):
    """
    Return an autocast context with autocast explicitly disabled when supported.
    Falls back to a no-op context on unsupported backends.
    """
    try:
        return torch.amp.autocast(device_type=device_type, enabled=False)
    except Exception:
        if device_type == "cuda":
            try:
                return torch.cuda.amp.autocast(enabled=False)
            except Exception:
                pass
        return nullcontext()


def _assert_finite(x: Tensor, where: str) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"Non-finite values detected {where}.")


class PositionalEncoding2D(nn.Module):
    """
    Fixed 2D sinusoidal positional encoding for image feature maps.

    Args:
        dim: Channel dimension of the feature map. Must be divisible by 4.
        h_max: Maximum supported height.
        w_max: Maximum supported width.
        device: Optional device used to initialize the encoding.
    """

    def __init__(
        self,
        dim: int,
        h_max: int,
        w_max: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        super().__init__()

        if dim % 4 != 0:
            raise ValueError(
                f"PositionalEncoding2D requires dim % 4 == 0, got dim={dim}."
            )

        self.h_max = int(h_max)
        self.w_max = int(w_max)
        self.dim = int(dim)

        pe = torch.zeros((1, dim, h_max, w_max), dtype=torch.float32, device=device)

        div = torch.exp(
            -torch.arange(0.0, dim // 2, 2, dtype=torch.float32, device=device)
            / dim
            * math.log(10000.0)
        ).unsqueeze(1)

        h_pos = torch.arange(0.0, h_max, dtype=torch.float32, device=device)
        w_pos = torch.arange(0.0, w_max, dtype=torch.float32, device=device)

        pe[:, : dim // 2 : 2, :, :] = (
            torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        )
        pe[:, 1 : dim // 2 : 2, :, :] = (
            torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        )
        pe[:, dim // 2 :: 2, :, :] = (
            torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        )
        pe[:, dim // 2 + 1 :: 2, :, :] = (
            torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        )

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add 2D positional encoding to a tensor of shape (B, C, H, W)."""
        h, w = x.shape[-2:]
        if h > self.h_max or w > self.w_max:
            raise ValueError(
                f"Requested positional encoding of size {(h, w)} exceeds "
                f"configured maximum {(self.h_max, self.w_max)}."
            )
        return x + self.pe[:, :, :h, :w]

    def get_pe_by_size(
        self,
        h: int,
        w: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        if h > self.h_max or w > self.w_max:
            raise ValueError(
                f"Requested positional encoding of size {(h, w)} exceeds "
                f"configured maximum {(self.h_max, self.w_max)}."
            )
        pe = self.pe[:, :, :h, :w]
        return pe if device is None else pe.to(device)


class MixDropout(nn.Module):
    """
    Randomly applies standard dropout or spatial dropout during training.

    The dropout modules themselves handle train/eval behavior; the random
    selection is only relevant during training.
    """

    def __init__(self, dropout_proba: float = 0.4, dropout2d_proba: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_proba)
        self.dropout2d = nn.Dropout2d(dropout2d_proba)

    def forward(self, x: Tensor) -> Tensor:
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)


class ConvBlock(nn.Module):
    """
    Three-convolution encoder block used in the initial stem.
    """

    def __init__(
        self,
        in_: int,
        out_: int,
        stride: Tuple[int, int] = (1, 1),
        k: int = 3,
        activation: Type[nn.Module] = nn.SiLU,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.activation = activation()
        self.conv1 = nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv2 = nn.Conv2d(in_channels=out_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv3 = nn.Conv2d(
            out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride
        )
        self.norm_layer = nn.InstanceNorm2d(
            out_, eps=1e-3, momentum=0.99, track_running_stats=False
        )
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x: Tensor) -> Tensor:
        dropout_position = random.randint(1, 3)

        x = self.activation(self.conv1(x))
        if dropout_position == 1:
            x = self.dropout(x)

        x = self.activation(self.conv2(x))
        if dropout_position == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.activation(self.conv3(x))
        if dropout_position == 3:
            x = self.dropout(x)

        return x


class DepthSepConv2D(nn.Module):
    """
    Depthwise-separable 2D convolution.

    The explicit asymmetric post-padding branch preserves the behavior of the
    original implementation for even kernels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        activation: Optional[nn.Module] = None,
        padding: Union[bool, Tuple[int, int]] = True,
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()

        if len(kernel_size) != 2:
            raise ValueError("kernel_size must contain exactly two values.")

        self.post_pad: Optional[Tuple[int, int, int, int]] = None

        if padding is True:
            if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                pad_h = kernel_size[1] - 1
                pad_w = kernel_size[0] - 1
                self.post_pad = (
                    pad_h // 2,
                    pad_h - pad_h // 2,
                    pad_w // 2,
                    pad_w - pad_w // 2,
                )
                conv_padding = (0, 0)
            else:
                conv_padding = tuple(int((k - 1) / 2) for k in kernel_size)
        elif padding is False:
            conv_padding = (0, 0)
        else:
            conv_padding = padding

        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=tuple(kernel_size),
            dilation=dilation,
            stride=stride,
            padding=conv_padding,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
            kernel_size=(1, 1),
        )
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.depth_conv(x)
        if self.post_pad is not None:
            x = pad(x, self.post_pad)
        if self.activation is not None:
            x = self.activation(x)
        x = self.point_conv(x)
        return x


class DSCBlockOriginal(nn.Module):
    """
    Depthwise-separable residual-style block used in the encoder body.
    """

    def __init__(
        self,
        in_: int,
        out_: int,
        stride: Tuple[int, int] = (2, 1),
        activation: Type[nn.Module] = nn.SiLU,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(
            out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride
        )
        self.norm_layer = nn.InstanceNorm2d(
            out_, eps=1e-3, momentum=0.99, track_running_stats=False
        )
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x: Tensor) -> Tensor:
        dropout_position = random.randint(1, 3)

        x = self.activation(self.conv1(x))
        if dropout_position == 1:
            x = self.dropout(x)

        x = self.activation(self.conv2(x))
        if dropout_position == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        if dropout_position == 3:
            x = self.dropout(x)

        return x


class ChannelAttention(nn.Module):
    """
    Efficient channel attention (ECA-style).
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    """
    Lightweight spatial attention (CBAM-style).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class PILOTEncoder(nn.Module):
    """
    PILOT visual encoder.

    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.__name__ = "PILOTEncoder"

        self.dropout = float(params["dropout"])
        self.use_checkpointing = bool(params.get("use_checkpointing", False))

        self.init_blocks = nn.Sequential(
            ConvBlock(3, 32, stride=(1, 1), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 256, stride=(2, 2), dropout=self.dropout),
            ConvBlock(256, 512, stride=(2, 1), dropout=self.dropout),
            ConvBlock(512, 512, stride=(2, 1), dropout=self.dropout),
        )

        self.blocks = nn.Sequential(
            DSCBlockOriginal(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlockOriginal(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlockOriginal(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlockOriginal(512, 1024, stride=(1, 1), dropout=self.dropout),
            ChannelAttention(1024),
            SpatialAttention(),
        )

    def forward(self, x: Tensor) -> Tensor:
        _assert_finite(x, "at the input of the encoder")

        for i, block in enumerate(self.init_blocks):
            if i >= 5:
                with _autocast_disabled(x.device.type):
                    x = x.float()
                    x = block(x)
            else:
                x = block(x)
            _assert_finite(x, f"after init block {i}")

        with _autocast_disabled(x.device.type):
            x = x.float()
            for j, block in enumerate(self.blocks):
                xt = block(x)
                _assert_finite(xt, f"after encoder body block {j}")

                x = x + xt if x.shape == xt.shape else xt
                _assert_finite(x, f"after residual connection of encoder body block {j}")

        return x
