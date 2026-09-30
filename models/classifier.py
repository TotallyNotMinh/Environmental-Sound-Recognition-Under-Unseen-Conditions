
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from models.encoder import ASTEncoder
try:
    from torchinfo import summary
except ImportError:
    summary = None


class AttentionPool(nn.Module):
    """Learned softmax-weighted sum over patch tokens (excludes CLS/DIST)."""
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):                     # x: (B, N, D) patch tokens only
        w = self.score(x).softmax(dim=1)      # (B, N, 1)
        return (w * x).sum(dim=1)             # (B, D)


class Classifer(nn.Module):
    """
    AST Classifier for FSD50K multi-label sound-event classification.
    pooling: None -> baseline (CLS+DIST)/2 average (unchanged)
             "gap" | "gap_max" | "attention" -> ablation variants over the 1188 patch tokens
    """
    def __init__(
        self,
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
        pooling=None,
    ):
        super().__init__()
        self.use_cls_dist = use_cls_dist
        self.pooling = pooling
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

        if pooling == "attention":
            self.pool = AttentionPool(tok_dim)

        head_in = 2 * tok_dim if pooling == "gap_max" else tok_dim
        self.head_norm = nn.LayerNorm(head_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(head_in, num_classes)

    def forward(self, x):
        # x: (B, 1, 128, target_frames)
        tokens = self.encoder(x)               # (B, 2 + 1188, tok_dim)

        if self.pooling in ("gap", "gap_max", "attention"):
            patch_tokens = tokens[:, 2:]        # (B, 1188, tok_dim), CLS/DIST dropped
            if self.pooling == "gap":
                pooled = patch_tokens.mean(dim=1)
            elif self.pooling == "gap_max":
                pooled = torch.cat([patch_tokens.mean(dim=1), patch_tokens.amax(dim=1)], dim=-1)
            else:  # "attention"
                pooled = self.pool(patch_tokens)
        elif self.use_cls_dist and tokens.shape[1] >= 2:
            pooled = (tokens[:, 0] + tokens[:, 1]) / 2.0     # baseline, unchanged
        else:
            pooled = tokens[:, 2:].mean(dim=1) if tokens.shape[1] > 2 else tokens.mean(dim=1)

        pooled = self.head_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.head(pooled)             # (B, num_classes)
        return logits


Classifier = Classifer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand([1, 1, 128, 1000], device=device)
    for p in [None, "gap", "gap_max", "attention"]:
        m = Classifer(pooling=p, pretrained_dino=False).to(device)
        print(p, m(x).shape)
