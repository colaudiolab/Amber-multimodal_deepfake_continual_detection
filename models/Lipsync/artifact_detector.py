"""
Artifact detection module for detecting AI manipulation in lip-sync videos.

This module focuses on detecting visual artifacts and inconsistencies that
AI manipulation tools (Wav2Lip, DeepFaceLab, etc.) introduce.
"""

from typing import Optional

import torch
from torch import Tensor, nn


def _laplacian_kernel_3ch() -> Tensor:
    """Spatial Laplacian kernel (high-pass). Shape (3, 3, 3, 3) for Conv2d(3, 3, 3)."""
    # [[0, 1, 0], [1, -4, 1], [0, 1, 0]] per channel (each out_ch sees one in_ch)
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    w = torch.zeros(3, 3, 3, 3)
    for i in range(3):
        w[i, i, :, :] = k
    return w


class HighFrequencyDetector(nn.Module):
    """
    Detects spatial high-frequency artifacts (GAN smoothing, face blending,
    deepfake boundaries) via Laplacian high-pass on raw video then Conv3D.
    """

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        # Laplacian (high-pass) per frame: Conv2d applied per timestep
        self.laplacian = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.laplacian.weight.data = _laplacian_kernel_3ch()

        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, video: Tensor) -> Tensor:
        """
        Args:
            video: (B, 3, T, H, W) - raw video frames

        Returns:
            (B, out_dim) - high-frequency artifact features
        """
        B, C, T, H, W = video.shape
        # (B, 3, T, H, W) -> (B*T, 3, H, W)
        x = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.laplacian(x)  # (B*T, 3, H, W)
        x = x.reshape(B, T, 3, H, W).permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W)
        x = self.conv3d(x)
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)


class TemporalInconsistencyDetector(nn.Module):
    """
    Detects temporal inconsistencies (flickering, frame-to-frame artifacts)
    that are common in AI-manipulated videos.
    """

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        # 3D conv to detect temporal inconsistencies
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(
                feature_dim,
                feature_dim // 2,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                feature_dim // 2,
                feature_dim // 4,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(feature_dim // 4),
            nn.ReLU(inplace=True),
        )
        # Global pooling over spatial and temporal
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, D, T, H, W) - visual features

        Returns:
            (B, D//4) - inconsistency features
        """
        out = self.temporal_conv(x)  # (B, D//4, T', H', W')
        out = self.pool(out)  # (B, D//4, 1, 1, 1)
        return out.squeeze(-1).squeeze(-1).squeeze(-1)  # (B, D//4)


class ArtifactDetector(nn.Module):
    """
    Detects manipulation artifacts in video by analyzing:
    1. Temporal inconsistencies (flickering, frame jumps) — raw feature map
    2. Delta map (frame-to-frame differences)
    3. High-frequency residual (Laplacian on raw video) — GAN smoothing, boundaries
    """

    def __init__(
        self,
        visual_feature_dim: int = 256,
        embed_dim: int = 256,
        use_delta_map: bool = True,
        use_high_freq: bool = True,
        high_freq_dim: int = 64,
    ) -> None:
        super().__init__()
        self.use_delta_map = use_delta_map
        self.use_high_freq = use_high_freq
        self.temporal_detector = TemporalInconsistencyDetector(visual_feature_dim)

        # Artifact feature dimension: raw + optional delta + optional high-freq
        artifact_dim = visual_feature_dim // 4
        detector_multiplier = 2 if use_delta_map else 1
        total_artifact_dim = artifact_dim * detector_multiplier
        if use_high_freq:
            self.high_freq_detector = HighFrequencyDetector(out_dim=high_freq_dim)
            total_artifact_dim = total_artifact_dim + high_freq_dim
        else:
            self.high_freq_detector = None

        # Combine CLS with artifact features
        self.artifact_fusion = nn.Sequential(
            nn.Linear(embed_dim + total_artifact_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        visual_features: Tensor,
        cls_output: Tensor,
        raw_video: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            visual_features: (B, D_v, T, H, W) - raw visual encoder output
            cls_output: (B, D_e) - CLS token from Temporal Transformer
            raw_video: (B, 3, T, H, W) - optional raw video for high-freq (Laplacian) branch

        Returns:
            artifact_features: (B, D_e//2) - artifact branch output for final concat
        """
        # Raw map
        artifact_feat = self.temporal_detector(visual_features)  # (B, D_v//4)

        if self.use_delta_map:
            if visual_features.size(2) > 1:
                delta_map = visual_features[:, :, 1:] - visual_features[:, :, :-1]
            else:
                delta_map = torch.zeros_like(visual_features)
            delta_feat = self.temporal_detector(delta_map)  # (B, D_v//4)
            artifact_feat = torch.cat([artifact_feat, delta_feat], dim=-1)

        if self.use_high_freq and self.high_freq_detector is not None and raw_video is not None:
            hf_feat = self.high_freq_detector(raw_video)  # (B, high_freq_dim)
            artifact_feat = torch.cat([artifact_feat, hf_feat], dim=-1)

        # Combine CLS and artifact features
        combined = torch.cat([cls_output, artifact_feat], dim=-1)
        artifact_features = self.artifact_fusion(combined)  # (B, D_e//2)

        return artifact_features
