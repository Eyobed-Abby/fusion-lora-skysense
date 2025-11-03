import torch, torch.nn as nn
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.zeros(base.out_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.scale = 1.0
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.B.t() @ self.A.t())
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
