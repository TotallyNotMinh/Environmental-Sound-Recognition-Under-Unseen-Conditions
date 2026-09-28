import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from models.encoder import ASTEncoder
from models.decoder import DenoisingDecoder
from torchinfo import summary


class Reconstructer(nn.Module):
    """
    AST Audio Spectrogram Reconstructer (SSAST-style Masked Patch Reconstruction).
    Encodes mel-spectrogram patches with ViT, applies random patch zeroing,
    and reconstructs the full clean mel-spectrogram using a convolutional upsampling decoder.
    """
    def __init__(
        self,
        tok_dim=192,
        c_in=1,
        overlap=6,
        patch_size=16,
        size=(128, 500),
        num_head=3,
        num_layer=12,
        use_dino=True,
        pretrained_dino=True,
    ):
        super().__init__()

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
        )
        self.decoder = DenoisingDecoder(tok_dim, patch_size, overlap, size=size)

    def forward(self, x, mask_ratio: float = 0.0):
        features = self.encoder(x, mask_ratio=mask_ratio)
        output = self.decoder(features)
        return output


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand([2, 1, 128, 500], device=device)
    model = Reconstructer(pretrained_dino=False).to(device=device)
    out = model(x, mask_ratio=0.6)
    print("Reconstructed shape:", out.shape)
    summary(model, [2, 1, 128, 500])
