import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b6, EfficientNet_B6_Weights,
)
try:
    from torchinfo import summary
except ImportError:
    summary = None


ARCH_CONFIGS = {
    "b0": {"builder": efficientnet_b0, "weights": EfficientNet_B0_Weights.IMAGENET1K_V1, "feature_dim": 1280, "name": "EfficientNet-B0"},
    "b2": {"builder": efficientnet_b2, "weights": EfficientNet_B2_Weights.IMAGENET1K_V1, "feature_dim": 1408, "name": "EfficientNet-B2"},
    "b6": {"builder": efficientnet_b6, "weights": EfficientNet_B6_Weights.IMAGENET1K_V1, "feature_dim": 2304, "name": "EfficientNet-B6"},
}


def _adapt_stem_to_single_channel(stem_conv: nn.Conv2d) -> nn.Conv2d:
    """
    Replaces a 3-input-channel stem conv with a 1-input-channel conv whose
    weights are the channel-SUM of the pretrained 3-channel weights.
    Same trick as models/encoder.py::DINOVisionTransformer.load_pretrained_dino_weights,
    applied to a plain conv instead of a patch-embed conv. (CMKD/AST average the
    channels instead; summing is a deliberate repo choice for consistency with
    the AST encoder -- see docs/cmkd-paper-notes.md section 9.2.)
    """
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=stem_conv.out_channels,
        kernel_size=stem_conv.kernel_size,
        stride=stem_conv.stride,
        padding=stem_conv.padding,
        bias=stem_conv.bias is not None,
    )
    with torch.no_grad():
        summed_weight = stem_conv.weight.sum(dim=1, keepdim=True)
        assert summed_weight.shape == new_conv.weight.shape, (
            f"Channel-summed stem weight shape {summed_weight.shape} "
            f"does not match new 1-channel conv shape {new_conv.weight.shape}"
        )
        new_conv.weight.copy_(summed_weight)
        if stem_conv.bias is not None:
            new_conv.bias.copy_(stem_conv.bias)
    return new_conv


class CNNClassifier(nn.Module):
    """
    CMKD-style CNN classifier: an ImageNet-pretrained EfficientNet backbone
    (1-channel-adapted) with plain time+frequency mean pooling and a linear
    head -- NOT PSLA's 4-headed attention pooling. See docs/cmkd-paper-notes.md
    section 9.2 for the paper source of this simplification.
    """
    def __init__(self, arch="b0", num_classes=200, pretrained=True, dropout=0.0):
        super().__init__()
        if arch not in ARCH_CONFIGS:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(ARCH_CONFIGS)}.")
        cfg = ARCH_CONFIGS[arch]
        self.arch = arch
        self.feature_dim = cfg["feature_dim"]

        weights = cfg["weights"] if pretrained else None
        backbone = cfg["builder"](weights=weights)

        # torchvision EfficientNet stem: features[0] is a Conv2dNormActivation
        # container whose index 0 is the Conv2d itself.
        stem_block = backbone.features[0]
        stem_block[0] = _adapt_stem_to_single_channel(stem_block[0])

        self.features = backbone.features  # drop torchvision's own avgpool + classifier
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 128, target_frames)
        feat = self.features(x)               # (B, feature_dim, H', W')
        pooled = feat.mean(dim=(2, 3))         # time+frequency mean pooling -> (B, feature_dim)
        pooled = self.dropout(pooled)
        logits = self.head(pooled)             # (B, num_classes)
        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand([2, 1, 128, 1000], device=device)
    model = CNNClassifier(arch="b0").to(device=device)
    out = model(x)
    print("Output shape:", out.shape)
    assert out.shape == (2, 200), f"Expected (2, 200), got {tuple(out.shape)}"
    print("Shape check passed.")
    if summary is not None:
        summary(model, input_size=(2, 1, 128, 1000))
