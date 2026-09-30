import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
import torch.nn.functional as F

class DenoisingDecoder(nn.Module):
    def __init__(self, size=(128, 1000)):
        super().__init__()

        (self.H, self.W) = size
        
        self.dec1= nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dec2= nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.dec3= nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec4= nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor):
        dec1 = self.dec1(features)
        dec1 = F.interpolate(dec1, scale_factor=2, mode="bilinear", align_corners=False)

        dec2 = self.dec2(dec1)
        dec2 = F.interpolate(dec2, scale_factor=2, mode="bilinear", align_corners=False)

        dec3 = self.dec3(dec2)
        dec3 = F.interpolate(dec3, scale_factor=2, mode="bilinear", align_corners=False)

        dec4 = self.dec4(dec3)
        dec4 = F.interpolate(dec4, scale_factor=2, mode="bilinear", align_corners=False)

        output = F.interpolate(dec4, (self.H, self.W), mode="bilinear", align_corners=False)

        return output