import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLateFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()  # Outputs gate ∈ (0,1)
        )

    def forward(self, fused_feat, visual_feat):
        """
        fused_feat:  [B, C, H, W]  (e.g. CAF Stage4)
        visual_feat: [B, C, H, W]  (original visual Stage4)
        """
        x = torch.cat([fused_feat, visual_feat], dim=1)  # [B, 2C, H, W]
        gate = self.gate_conv(x)                         # [B, C, H, W]
        return gate * fused_feat + (1 - gate) * visual_feat
