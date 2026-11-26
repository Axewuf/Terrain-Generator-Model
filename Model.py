import torch
import torch.nn as nn
import torch.nn.functional as F

class TerrainSR(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        # Upscale input 2x
        upscaled = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        # Learn residual corrections
        y = F.relu(self.conv1(upscaled))
        y = F.relu(self.conv2(y))
        residual = self.conv3(y)

        return upscaled + residual