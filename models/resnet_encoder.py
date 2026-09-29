import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet152_Weights
)

class ResNetEncoder(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True, in_channels=1):
        super().__init__()
        
        # 1. Khởi tạo backbone tương ứng
        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            self.out_dim = 512
        elif model_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
            self.out_dim = 512
        elif model_name == "resnet152":
            weights = ResNet152_Weights.DEFAULT if pretrained else None
            backbone = resnet152(weights=weights)
            self.out_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose from ['resnet18', 'resnet34', 'resnet152']")

        # 2. Sửa conv1 từ 3 channels (RGB) về 1 channel (Spectrogram)
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # Giữ lại tri thức tiền huấn luyện (pretrained weights) bằng trung bình cộng 3 channels
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = new_conv

        # 3. Trích xuất các tầng convolution
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

    def forward(self, x):
        # Input x: (B, 1, 128, 500)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)            # (B, out_dim, 1, 1)
        feat = torch.flatten(x, 1)     # (B, out_dim)
        return feat