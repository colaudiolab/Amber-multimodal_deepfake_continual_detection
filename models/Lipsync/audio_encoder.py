from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class _ConvBNReLU(nn.Sequential):
    """
    Conv2d -> BatchNorm2d -> ReLU building block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Tuple[int, int] = (1, 1),
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _ResidualBlock(nn.Module):
    """
    Simple residual block used for audio spectrogram encoding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.conv1 = _ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Optional projection if the number of channels / stride changes.
        if stride != (1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out


class AudioEncoder(nn.Module):
    """
    2D ResNet‑style encoder over log Mel‑spectrograms for lip‑sync.

    **Input**
    - `x`: `(B, 1, F, T)` where
      - `B` = batch size
      - `F` = number of Mel bins / frequency bins
      - `T` = audio time steps

    **Output**
    - `(B, D_a, T')` where
      - `D_a = feature_dim`
      - `T'` is the downsampled temporal length (kept relatively high and
        later aligned / interpolated in the fusion module).

    - ResNet‑style stages with residual connections
    - Aggressive downsampling in frequency, light in time
    - Global average pooling over frequency to keep temporal resolution
    """

    def __init__(
        self,
        feature_dim: int = 256,
        in_channels: int = 1,
        base_channels: int = 64,
        dropout: float = 0.1,
        preserve_audio_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        # When True: layer3 stride (2,1) -> T'≈T/4. When False: (2,2) -> T'≈T/8 (original).
        self.preserve_audio_temporal = preserve_audio_temporal

        # Stem: larger kernel and stride in time to quickly aggregate
        # local temporal context while keeping frequency reasonably resolved.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=(2, 2),  # downsample in both F and T a bit
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1),
        )

        # ResNet‑like stages. We mostly downsample along frequency and
        # are conservative along time to preserve sync information.
        layer3_stride = (2, 1) if preserve_audio_temporal else (2, 2)
        self.layer1 = _ResidualBlock(base_channels, base_channels, stride=(1, 1))
        self.layer2 = _ResidualBlock(base_channels, base_channels * 2, stride=(2, 2))
        self.layer3 = _ResidualBlock(
            base_channels * 2,
            base_channels * 4,
            stride=layer3_stride,  # (2,1) keeps more T for lip-sync; (2,2) original
        )
        self.layer4 = _ResidualBlock(
            base_channels * 4,
            feature_dim,
            stride=(2, 1),  # stronger downsample in F, lighter in T
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming init for convs, standard init for batch norms / linears.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Audio spectrogram, shape `(B, 1, F, T)`.
            lengths: Optional tensor of valid temporal lengths per batch
                element (before downsampling). This encoder does not
                currently use `lengths` directly, but it is accepted for
                API compatibility and future masking logic.

        Returns:
            Tensor of shape `(B, D_a, T')`.
        """
        if x.dim() != 4:
            raise ValueError(
                f"AudioEncoder expected input of shape (B, 1, F, T), got {tuple(x.shape)}"
            )

        # (B, C, F, T)
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # (B, D_a, F', T')

        out = self.dropout(out)

        # Pool over frequency only; preserve temporal resolution (AdaptiveAvgPool2d (1, T')).
        t_len = out.size(3)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, t_len))
        out = out.squeeze(2)  # (B, D_a, T')
        return out

