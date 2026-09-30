import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from models.encoder import EfficientNetEncoder
from models.decoder import DenoisingDecoder
try:
    from torchinfo import summary
except ImportError:
    summary = None

class Denoiser(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.encoder = EfficientNetEncoder()
        self.decoder = DenoisingDecoder()

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand([1, 1, 128, 1000], device=device)
    model = Denoiser().to(device=device)
    print(model(x).shape)
    summary(model, [1, 1, 128, 1000])