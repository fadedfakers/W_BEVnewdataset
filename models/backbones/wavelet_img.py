import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import List

class DWTDownsampling(nn.Module):
    """
    Discrete Wavelet Transform (DWT) Downsampling using Haar Wavelets.
    Replaces standard Pooling to preserve high-frequency details.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        # Haar Wavelet Kernels
        # LL: Low-pass (Average)
        # LH: Vertical edges
        # HL: Horizontal edges
        # HH: Diagonal edges
        kernel = torch.tensor([
            [[1, 1], [1, 1]],
            [[1, 1], [-1, -1]],
            [[1, -1], [1, -1]],
            [[1, -1], [-1, 1]]
        ]).float() / 2.0
        
        # Reshape to (4, 1, 2, 2) for depthwise convolution
        kernel = kernel.unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer('weight', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, H, W)
        Output: (B, C*4, H/2, W/2)
        """
        B, C, H, W = x.shape
        # Use groups=C to perform depthwise wavelet transform
        out = F.conv2d(x, self.weight, stride=2, groups=C)
        return out

class WaveletResNet(nn.Module):
    """
    ResNet-50 variant that uses DWTDownsampling instead of MaxPooling.
    Extracts multi-scale features for FPN/BEV fusion.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = resnet50(weights=weights)
        
        # Root layers
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        
        # Replace MaxPool2d with DWT + 1x1 Conv
        # ResNet conv1 output is 64 channels. DWT makes it 256.
        self.dwt_pool = DWTDownsampling(64)
        self.pool_reduction = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        
        # ResNet Layers
        self.layer1 = base_resnet.layer1 # 256 output
        self.layer2 = base_resnet.layer2 # 512 output
        self.layer3 = base_resnet.layer3 # 1024 output
        self.layer4 = base_resnet.layer4 # 2048 output

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning intermediate features.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        
        # DWT instead of MaxPool
        x = self.dwt_pool(x)
        x = self.pool_reduction(x)
        
        f1 = self.layer1(x)  # 1/4 res
        f2 = self.layer2(f1) # 1/8 res
        f3 = self.layer3(f2) # 1/16 res
        f4 = self.layer4(f3) # 1/32 res
        
        return [f2, f3, f4] # Return multi-scale features for fusion
