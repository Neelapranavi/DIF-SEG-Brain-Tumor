import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DIF_SEG(nn.Module):
    """
    Fallback U-Net Architecture since the original DIF-SEG is unavailable.
    Uses the mateuszbuda/brain-segmentation-pytorch architecture.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Load the architecture from PyTorch Hub without pretrained weights
        # We will load the pretrained weights from our local checkpoint
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', 
                                  in_channels=in_channels, out_channels=out_channels, 
                                  init_features=32, pretrained=False)

    def forward(self, x):
        # Convert 1-channel to 3-channel if necessary
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)
