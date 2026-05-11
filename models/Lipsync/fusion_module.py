from typing import Tuple

import torch
from torch import Tensor, nn


class CrossModalAttention(nn.Module):
    """
    Gated cross-modal attention: Video attends to Audio, Audio attends to Video,
    then modality gating blends the two (trust video more when audio is noisy,
    trust audio when lips are occluded).

    Inputs:
        visual_emb: (B, T_v, D_e)
        audio_emb:  (B, T_a, D_e)

    Output:
        fused:      (B, T_v, D_e)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.v2a_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.a2v_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Modality gate: sigmoid(Linear([v_out, a_out])) -> blend v_out vs a_out per token
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, visual_emb: Tensor, audio_emb: Tensor) -> Tensor:
        if visual_emb.dim() != 3 or audio_emb.dim() != 3:
            raise ValueError(
                "CrossModalAttention expects visual_emb and audio_emb of shape (B, T, D_e)"
            )

        b_v, t_v, d_v = visual_emb.shape
        b_a, t_a, d_a = audio_emb.shape
        if b_v != b_a or d_v != d_a:
            raise ValueError(
                "visual_emb and audio_emb must have the same batch size and feature dim"
            )

        if t_v != t_a:
            audio_emb = torch.nn.functional.interpolate(
                audio_emb.transpose(1, 2),
                size=t_v,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        # Video attends to Audio
        v_attended, _ = self.v2a_attn(visual_emb, audio_emb, audio_emb)
        v_out = visual_emb + v_attended

        # Audio attends to Video
        a_attended, _ = self.a2v_attn(audio_emb, visual_emb, visual_emb)
        a_out = audio_emb + a_attended

        # Gated fusion: gate * v_out + (1 - gate) * a_out (per-token modality weighting)
        gate_input = torch.cat([v_out, a_out], dim=-1)  # (B, T, 2*D_e)
        g = self.gate(gate_input)  # (B, T, 1)
        fused = g * v_out + (1.0 - g) * a_out  # (B, T, D_e)
        return self.fuse(fused)


class FeatureProjection(nn.Module):
    """
    Projects visual and audio encodings to a shared embedding dimension.

    Expected inputs:
        visual_feat: (B, D_v, T_v)
        audio_feat:  (B, D_a, T_a)

    Outputs:
        visual_emb: (B, T_v, D_e)
        audio_emb:  (B, T_a, D_e)
    """

    def __init__(self, visual_dim: int, audio_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)

    def forward(
        self,
        visual_feat: Tensor,
        audio_feat: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if visual_feat.dim() != 3 or audio_feat.dim() != 3:
            raise ValueError(
                "FeatureProjection expects visual_feat and audio_feat of shape (B, D, T)"
            )

        # (B, D, T) -> (B, T, D)
        v = visual_feat.transpose(1, 2)
        a = audio_feat.transpose(1, 2)

        v = self.visual_proj(v)
        a = self.audio_proj(a)
        return v, a


class FusionModule(nn.Module):
    """
    Time‑wise fusion of projected visual and audio embeddings.

    If temporal lengths differ, the audio sequence is linearly interpolated
    to match the visual sequence length (simple but effective for lip‑sync).

    Inputs:
        visual_emb: (B, T_v, D_e)
        audio_emb:  (B, T_a, D_e)

    Output:
        fused:      (B, T_v, D_e)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, visual_emb: Tensor, audio_emb: Tensor) -> Tensor:
        if visual_emb.dim() != 3 or audio_emb.dim() != 3:
            raise ValueError(
                "FusionModule expects visual_emb and audio_emb of shape (B, T, D_e)"
            )

        b_v, t_v, d_v = visual_emb.shape
        b_a, t_a, d_a = audio_emb.shape
        if b_v != b_a or d_v != d_a:
            raise ValueError(
                "visual_emb and audio_emb must have the same batch size and feature dim"
            )

        if t_v != t_a:
            # Interpolate audio along the temporal dimension to match visual length.
            audio_emb = torch.nn.functional.interpolate(
                audio_emb.transpose(1, 2),  # (B, D_e, T_a)
                size=t_v,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)  # (B, T_v, D_e)

        x = torch.cat([visual_emb, audio_emb], dim=-1)  # (B, T_v, 2D_e)
        x = self.fc(x)
        return x

