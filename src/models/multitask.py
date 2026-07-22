import torch
import torch.nn as nn
from typing import Tuple


class MultiTaskTumorModel(nn.Module):
    """
    Multi-Task Deep Architecture for Brain Tumor Classification & Uncertainty Estimation.
    
    Shared Backbone: ResNet-18 feature extractor.
    Task Head 1: Classification head predicting 4 tumor/normal categories.
    Task Head 2: Confidence (Uncertainty) head predicting instance predictive reliability.
    """
    def __init__(self, pretrained_encoder: nn.Module, feature_dim: int = 512, num_classes: int = 4):
        super().__init__()
        self.encoder = pretrained_encoder
        self.classification_head = nn.Linear(feature_dim, num_classes)
        self.confidence_head = nn.Sequential(nn.Linear(feature_dim, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        class_logits = self.classification_head(features)
        confidence_logit = self.confidence_head(features).squeeze(-1)
        return class_logits, confidence_logit
