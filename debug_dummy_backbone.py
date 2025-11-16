# debug_dummy_backbone.py
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

from fusion_lora.spectral_tokenizer import SpectralTokenizer

class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [B, 64, H/2, W/2]
        x = F.relu(self.conv2(x))   # [B,128,H/4,W/4]
        return x

def main():
    sample_path = Path("datasets/bigearthnet_s2/train_tensors/sample_000003.pt")
    x = torch.load(sample_path).unsqueeze(0)  # [1, 6, 256, 256]

    tokenizer = SpectralTokenizer(in_channels=6, out_dim=3)
    pseudo_rgb = tokenizer(x, target_size=(384, 384))

    backbone = DummyBackbone()
    out = backbone(pseudo_rgb)

    print("Backbone output:", out.shape)

if __name__ == "__main__":
    main()
