from typing import Tuple

import torch
from torch import Tensor, nn


class _Conv3dBNReLU(nn.Sequential):
    """
    Conv3d -> BatchNorm3d -> ReLU building block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class _ResidualBlock3D(nn.Module):
    """
    Simple residual block for spatio‑temporal (video) encoding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        self.conv1 = _Conv3dBNReLU(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=(1, 1, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        )

        if stride != (1, 1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
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


class VisualEncoder(nn.Module):
    """
    Lightweight 3D ResNet‑style encoder over mouth‑crop video clips.

    **Input**
    - `x`: `(B, 3, T, H, W)` mouth crops (RGB, already normalized to [0,1] or similar).

    **Output**
    - `(B, D_v, T')` where
      - `D_v = feature_dim`
      - `T'` is the downsampled temporal resolution.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        base_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        # Stem: moderate temporal kernel, stronger spatial stride.
        self.stem = nn.Sequential(
            nn.Conv3d(
                3,
                base_channels,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )

        # Residual stages. We downsample spatially more than temporally
        # to preserve sync‑relevant timing.
        self.layer1 = _ResidualBlock3D(
            base_channels,
            base_channels,
            stride=(1, 1, 1),
        )
        self.layer2 = _ResidualBlock3D(
            base_channels,
            base_channels * 2,
            stride=(1, 2, 2),
        )
        self.layer3 = _ResidualBlock3D(
            base_channels * 2,
            base_channels * 4,
            stride=(1, 2, 2),
        )
        self.layer4 = _ResidualBlock3D(
            base_channels * 4,
            feature_dim,
            stride=(1, 2, 2),
        )

        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, return_map: bool = False):
        """
        Args:
            x: Tensor of shape `(B, 3, T, H, W)`.
            return_map: If True, also return the final spatio-temporal feature map
                `(B, D_v, T', H', W')` before spatial pooling (useful for artifact
                detection).

        Returns:
            If `return_map=False`: `(B, D_v, T')`
            If `return_map=True`: `(pooled, feature_map)` where
              - pooled: `(B, D_v, T')`
              - feature_map: `(B, D_v, T', H', W')`
        """
        if x.dim() != 5:
            raise ValueError(
                f"VisualEncoder expected input of shape (B, 3, T, H, W), got {tuple(x.shape)}"
            )

        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # (B, D_v, T', H', W')

        out = self.dropout(out)
        feature_map = out

        # Preserve full temporal resolution: pool only over spatial dims (H, W).
        # AdaptiveAvgPool3d((T', 1, 1)) keeps T' intact for lip-sync (errors in 2–3 frames).
        t_len = out.size(2)
        pooled = torch.nn.functional.adaptive_avg_pool3d(out, (t_len, 1, 1))
        pooled = pooled.squeeze(-1).squeeze(-1)  # (B, D_v, T')
        if return_map:
            return pooled, feature_map
        return pooled

