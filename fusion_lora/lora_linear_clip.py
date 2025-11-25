import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Frozen pretrained weight & bias
        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        # LoRA A, B (trainable)
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = self.dropout(x) @ self.A.T @ self.B.T
        return base + self.scaling * lora
