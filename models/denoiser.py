import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from models.encoder import ASTEncoder
from models.decoder import DenoisingDecoder
from torchinfo import summary

class Denoiser(nn.Module):
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

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

if __name__ == "__main__":
    x = torch.rand([1, 1, 128, 500], device="cuda")
    model = Denoiser().to(device="cuda")
    print(model(x).shape)
    summary(model, [1, 1, 128, 500])