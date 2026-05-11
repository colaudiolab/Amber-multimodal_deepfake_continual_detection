from typing import Optional

import torch
from torch import Tensor, nn


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer with CLS token for sequence aggregation.

    Adds a learnable CLS token, runs a Transformer encoder over the sequence,
    and returns the CLS output as the aggregated representation.

    Input:  (B, T, D)
    Output: (B, D)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        pre_conv: bool = True,
        multi_scale_pre_conv: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pre_conv_enabled = pre_conv
        self.multi_scale_pre_conv = multi_scale_pre_conv
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Multi-scale temporal conv: k=3 (micro lip), k=5 (phoneme), k=7 (syllable) -> concat -> linear.
        if multi_scale_pre_conv:
            self.branch_k3 = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
            )
            self.branch_k5 = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
            )
            self.branch_k7 = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
            )
            self.pre_scale_proj = nn.Linear(3 * embed_dim, embed_dim)
        else:
            self.branch_k3 = self.branch_k5 = self.branch_k7 = None
            self.pre_scale_proj = None
            self.pre_temporal_conv = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Tensor of shape `(B, T, D)` – sequence of fused features.
            lengths: Optional 1D tensor `(B,)` with valid lengths per sequence.
                Not used in this implementation (full attention over sequence).

        Returns:
            Tensor of shape `(B, D)` – CLS token output.
        """
        if x.dim() != 3:
            raise ValueError(
                f"TemporalTransformer expected input of shape (B, T, D), got {tuple(x.shape)}"
            )

        b, t, d = x.shape
        if self.pre_conv_enabled:
            if self.multi_scale_pre_conv and self.branch_k3 is not None:
                x_t = x.transpose(1, 2)  # (B, D, T)
                c3 = self.branch_k3(x_t)   # (B, D, T)
                c5 = self.branch_k5(x_t)   # (B, D, T)
                c7 = self.branch_k7(x_t)   # (B, D, T)
                x_conv = torch.cat([c3, c5, c7], dim=1).transpose(1, 2)  # (B, T, 3*D)
                x_conv = self.pre_scale_proj(x_conv)  # (B, T, D)
            else:
                x_conv = self.pre_temporal_conv(x.transpose(1, 2)).transpose(1, 2)
            x = x + x_conv

        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, x], dim=1)  # (B, 1+T, D)

        out = self.transformer(tokens)
        return out[:, 0]  # (B, D)


class TemporalAggregation(nn.Module):
    """
    Temporal aggregation for fused audio‑visual features.

    Default behavior is simple global average pooling over the time axis,
    which is robust and efficient. The API is intentionally kept small
    but can be extended later to support attention‑based pooling while
    remaining backward compatible.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Tensor of shape `(B, T, D)` – sequence of fused features.
            lengths: Optional 1D tensor `(B,)` with valid lengths per
                sequence (before padding). If provided, padded positions
                (t >= lengths[b]) are ignored in the mean.

        Returns:
            Tensor of shape `(B, D)`.
        """
        if x.dim() != 3:
            raise ValueError(
                f"TemporalAggregation expected input of shape (B, T, D), got {tuple(x.shape)}"
            )

        if lengths is None:
            return x.mean(dim=1)

        # Masked temporal mean for variable‑length sequences.
        if lengths.dim() != 1 or lengths.size(0) != x.size(0):
            raise ValueError(
                "lengths must be 1D with shape (B,) and match batch size of x"
            )

        device = x.device
        max_t = x.size(1)
        # Shape: (B, T)
        time_indices = (
            x.new_tensor(range(max_t), dtype=lengths.dtype, device=device)
            .unsqueeze(0)
            .expand(x.size(0), -1)
        )
        # mask[b, t] = True if t < lengths[b]
        mask = time_indices < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # (B, T, 1)

        x_masked = x * mask  # zeros out padded steps
        # Avoid division by zero by clamping lengths
        denom = lengths.clamp_min(1).to(x.dtype).unsqueeze(-1)
        return x_masked.sum(dim=1) / denom

