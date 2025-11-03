import torch.nn as nn
class SpectralTokenizer(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x): return self.bn(self.proj(x))
