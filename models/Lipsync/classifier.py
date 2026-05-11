from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """
    MLP head that outputs **logits** for binary classification.

    Use `torch.nn.BCEWithLogitsLoss` during training and apply `torch.sigmoid`
    at inference time to convert logits -> probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(hidden_dim),
            # nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape `(B, D)`.

        Returns:
            Tensor of shape `(B,)` with **logits** (unbounded real values).
        """
        if x.dim() != 2:
            raise ValueError(
                f"ClassificationHead expected input of shape (B, D), got {tuple(x.shape)}"
            )
        return self.net(x)

