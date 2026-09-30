import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from models.encoder import ASTEncoder, EfficientNetEncoder
try:
    from torchinfo import summary
except ImportError:
    summary = None

class Classifier(nn.Module):
    """
    Classifier for FSD50K multi-label sound-event classification.
    Supports both AST (Vision Transformer) and EfficientNet (B0) backbones.
    """
    def __init__(
        self,
        encoder_type="ast",
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
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.use_cls_dist = use_cls_dist

        if self.encoder_type == "efficientnet":
            self.encoder = EfficientNetEncoder(pretrained=pretrained_dino)
            feat_dim = self.encoder.out_dim
            self.head_norm = nn.LayerNorm(feat_dim, eps=1e-6)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(feat_dim, num_classes)
        else:
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
            self.head_norm = nn.LayerNorm(tok_dim, eps=1e-6)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(tok_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        if self.encoder_type == "efficientnet":
            # x: (B, 1, 128, target_frames)
            features_2d = self.encoder(x)          # (B, 1280, H, W)
            pooled = features_2d.mean(dim=(-2, -1)) # Global Average Pooling -> (B, 1280)
            pooled = self.head_norm(pooled)
            features = pooled
            pooled = self.dropout(pooled)
            logits = self.head(pooled)              # (B, num_classes)
        else:
            # AST ViT pathway
            tokens = self.encoder(x)                # (B, 2 + N, tok_dim) or (B, N, tok_dim)
            has_cls_dist = getattr(self.encoder, "has_cls_dist", True)
            if self.use_cls_dist and has_cls_dist and tokens.shape[1] >= 2:
                # Dual-token average from AST (CLS + DIST)
                pooled = (tokens[:, 0] + tokens[:, 1]) / 2.0
            else:
                pooled = tokens[:, 2:].mean(dim=1) if (has_cls_dist and tokens.shape[1] > 2) else tokens.mean(dim=1)
            pooled = self.head_norm(pooled)
            features = pooled
            pooled = self.dropout(pooled)
            logits = self.head(pooled)              # (B, num_classes)

        if return_features:
            return logits, features
        return logits


Classifer = Classifier


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand([1, 1, 128, 500], device=device)
    model_ast = Classifier(encoder_type="ast").to(device=device)
    print("AST Output shape:", model_ast(x).shape)
    model_eff = Classifier(encoder_type="efficientnet").to(device=device)
    print("EfficientNet Output shape:", model_eff(x).shape)