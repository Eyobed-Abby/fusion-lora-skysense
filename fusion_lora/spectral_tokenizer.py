import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralTokenizer(nn.Module):
    def __init__(self, in_channels = 6, out_dim = 2816, spatial_downsample=32):
        """
        Maps 6-band multispectral input to visual encoder-compatible tokens.
        Args:
            in_channels: Number of input spectral bands (e.g. 6 for RGB + NIR + 2 SWIR)
            out_dim: Output embedding dimension (match visual encoder stage3, e.g. 96)
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_dim, kernel_size=1),
        )

        self.spatial_downsample = spatial_downsample
        
    def forward(self, x, target_size=None):
        """
        x: Tensor of shape [B, in_channels, H, W]
        returns: Tensor of shape [B, out_dim, H, W]
        """
        x = self.proj(x)
        if target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


#Testing
"""
if __name__ == "__main__":
    model = SpectralTokenizer(in_channels=6, out_dim=96)
    dummy_input = torch.randn(4, 6, 120, 120)  # batch of 4 patches
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: [4, 96, 120, 120]
    print(output)
"""