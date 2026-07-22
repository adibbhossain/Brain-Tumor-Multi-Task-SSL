import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from lightly.models.modules import SimCLRProjectionHead


class SimCLRModel(nn.Module):
    """
    SimCLR Contrastive Self-Supervised Learning Model.
    ResNet-18 Backbone paired with a multi-layer projection head.
    """
    def __init__(
        self,
        pretrained_weights: bool = True,
        input_dim: int = 512,
        hidden_dim: int = 2048,
        output_dim: int = 128
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_weights else None
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()

        self.projection_head = SimCLRProjectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        embeddings = self.projection_head(features)
        return embeddings

    def get_encoder(self) -> nn.Module:
        """Returns the underlying ResNet encoder."""
        return self.encoder
