import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from models.encoder import ASTEncoder
from models.resnet_encoder import ResNetEncoder
from torchinfo import summary

class Classifer(nn.Module):
    """
    Unified Classifier for FSD50K multi-label sound-event classification.
    Supports AST (ViT-Tiny/Small/Base with DINO), ResNet18, ResNet34, and ResNet152 backbones.
    """
    def __init__(
        self,
        encoder_type="ast",            # "ast", "resnet18", "resnet34", "resnet152"
        tok_dim=192,
        num_classes=200,
        c_in=1,
        overlap=6,
        patch_size=16,
        size=(128, 1000),
        num_head=3,
        num_layer=12,
        dropout=0.1,
        use_dino=True,
        pretrained_dino=True,
        pretrained_url=None,
        use_cls_dist=True,
        pretrained=True,               # For ResNet backbones
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.use_cls_dist = use_cls_dist

        if encoder_type == "ast":
            self.encoder = ASTEncoder(
                tok_dim=tok_dim,
                c_in=c_in,
                overlap=overlap,
                patch_size=patch_size,
                size=size,
                num_head=num_head,
                num_layer=num_layer,
                use_dino=use_dino,
                pretrained_dino=pretrained_dino,
                pretrained_url=pretrained_url,
            )
            feature_dim = tok_dim
            self.head_norm = nn.LayerNorm(tok_dim, eps=1e-6)
        elif encoder_type in ["resnet18", "resnet34", "resnet152"]:
            self.encoder = ResNetEncoder(
                model_name=encoder_type,
                pretrained=pretrained,
                in_channels=c_in
            )
            feature_dim = self.encoder.out_dim
            self.head_norm = None
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}. Choose from ['ast', 'resnet18', 'resnet34', 'resnet152']")

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 128, target_frames)
        if self.encoder_type == "ast":
            tokens = self.encoder(x)          # (B, 2 + N, tok_dim)
            if self.use_cls_dist and tokens.shape[1] >= 2:
                # Dual-token average from AST (CLS + DIST)
                pooled = (tokens[:, 0] + tokens[:, 1]) / 2.0
            else:
                pooled = tokens[:, 2:].mean(dim=1) if tokens.shape[1] > 2 else tokens.mean(dim=1)
            pooled = self.head_norm(pooled)
        else:
            pooled = self.encoder(x)             # (B, out_dim)

        pooled = self.dropout(pooled)
        logits = self.head(pooled)        # (B, num_classes)
        return logits


Classifier = Classifer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand([1, 1, 128, 500], device=device)
    model = Classifer().to(device=device)
    print("Output shape:", model(x).shape)
    summary(model, [1, 1, 128, 500])
