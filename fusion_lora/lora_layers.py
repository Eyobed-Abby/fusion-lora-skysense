import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r = 8, alpha = 1.0, dropout=0.0, bias=True):
        """
        LoRA layer for adapting pre-trained models with low-rank updates.
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            r: Rank of the LoRA update
            alpha: Scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # LoRA adapters
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        self.dropout = nn.Dropout(dropout)

        # Freeze base weight
        self.weight.requires_grad = False

    def forward(self, x):
        lora_update = self.dropout(x) @ self.A.T
        lora_update = lora_update @ self.B.T
        out = F.linear(x, self.weight, self.bias)
        return out + self.scale * lora_update