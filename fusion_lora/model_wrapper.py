import torch.nn as nn
from fusion_lora.spectral_tokenizer import SpectralTokenizer
from fusion_lora.lora_layers import LoRALinear

class FusionLoRAWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = SpectralTokenizer(6, 3)
    def forward(self, x6):
        x3 = self.tokenizer(x6)
        return x3
