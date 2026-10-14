import torch
import torch.nn as nn
from torch.nn import functional as F

class Up_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x

    
class Down_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.conv(x)
        return x

def zero_(m):
    for p in m.parameters():
        nn.init.zeros_(p)
    return m

class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, E, dropout=0.0, groups=32):
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.embedding = nn.Linear(E, 2 * out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, e):
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        s, b = self.embedding(F.silu(e))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + s) + b
        h = self.dropout(F.silu(h))
        h = self.conv2(h)
        return self.skip(x) + h

    

