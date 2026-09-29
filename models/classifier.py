import torch
import torch.nn as nn
from models.encoder import ASTEncoder
from models.resnet_encoder import ResNetEncoder


class Classifer(nn.Module):
    """
    Unified Classifier for FSD50K multi-label sound-event classification.
    Supports AST, ResNet18, ResNet34, and ResNet152 backbones.
    """
    def __init__(
        self,
        encoder_type="resnet18",      # "resnet18", "resnet34", "resnet152", hoặc "ast"
        pretrained=True,
        tok_dim=768,
        num_classes=200,
        c_in=1,
        overlap=6,
        patch_size=16,
        size=(128, 500),
        num_head=8,
        num_layer=12,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "ast":
            self.encoder = ASTEncoder(
                tok_dim=tok_dim,
                c_in=c_in,
                overlap=overlap,
                patch_size=patch_size,
                size=size,
                num_head=num_head,
                num_layer=num_layer,
            )
            feature_dim = tok_dim
        elif encoder_type in ["resnet18", "resnet34", "resnet152"]:
            self.encoder = ResNetEncoder(
                model_name=encoder_type,
                pretrained=pretrained,
                in_channels=c_in
            )
            feature_dim = self.encoder.out_dim
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 128, 500)
        if self.encoder_type == "ast":
            tokens = self.encoder(x)             # (B, 588, 768)
            pooled = tokens.mean(dim=1)          # (B, 768)
        else:
            pooled = self.encoder(x)             # (B, out_dim)

        pooled = self.dropout(pooled)
        logits = self.head(pooled)               # (B, num_classes)
        return logits