# models/vit_backbone.py

import torch
import torch.nn as nn
import timm

class ViTForBreastCancer(nn.Module):
    """
    Vision Transformer model adapted for binary classification:
    class 0 = non-cancer, class 1 = cancer.
    """
    def __init__(self, model_name: str = "vit_base_patch16_224", num_classes: int = 2):
        super().__init__()

        # Load a pretrained ViT backbone from timm
        self.backbone = timm.create_model(model_name, pretrained=True)

        # Get number of features going into the classification head
        in_features = self.backbone.head.in_features

        # Remove the original head and add our own classifier
        self.backbone.head = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input image tensor of shape [batch_size, 3, 224, 224]
        returns: logits of shape [batch_size, 2]
        """
        features = self.backbone(x)      # [batch_size, in_features]
        logits = self.classifier(features)
        return logits
