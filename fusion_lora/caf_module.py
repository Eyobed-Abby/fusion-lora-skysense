import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_lora.lora_layers import LoRALinear

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, lora_rank=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim ** -0.5

        # LoRA-enabled projections
        self.q_proj = LoRALinear(embed_dim, embed_dim, r=lora_rank)
        self.k_proj = LoRALinear(embed_dim, embed_dim, r=lora_rank)
        self.v_proj = LoRALinear(embed_dim, embed_dim, r=lora_rank)
        self.out_proj = LoRALinear(embed_dim, embed_dim, r=lora_rank)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, visual_feat, spectral_feat):
        B, C, H, W = visual_feat.shape

        # Flatten to sequences
        Q = visual_feat.flatten(2).transpose(1, 2)  # [B, N, C]
        K = spectral_feat.flatten(2).transpose(1, 2)
        V = spectral_feat.flatten(2).transpose(1, 2)

        # Normalize
        Q = self.norm1(Q)
        K = self.norm1(K)
        V = self.norm1(V)

        # Project
        Q_proj = self.q_proj(Q)
        K_proj = self.k_proj(K)
        V_proj = self.v_proj(V)

        # Reshape for attention
        B, N, _ = Q_proj.shape
        Q_proj = Q_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        K_proj = K_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V_proj = V_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) * self.scale  # [B, h, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V_proj)  # [B, h, N, d]

        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # Residual + FFN
        out = Q + attn_output
        out = out + self.ffn(self.norm2(out))
        out = out.transpose(1, 2).view(B, C, H, W)

        return out
