import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_lora.spectral_tokenizer import SpectralTokenizer
from fusion_lora.caf_module import CrossAttentionFusion

class FusionLoRAWrapper(nn.Module):
    def __init__(self, visual_encoder):
        super().__init__()
        self.visual_encoder = visual_encoder  # e.g. CLIP or SkySense-O encoder (frozen)

        self.spectral_tokenizer = SpectralTokenizer(in_channels=6, out_dim=2816)
        self.caf_stage3 = CrossAttentionFusion(embed_dim=1408, num_heads=8, lora_rank=8)
        self.caf_stage4 = CrossAttentionFusion(embed_dim=2816, num_heads=8, lora_rank=8)

    def forward(self, rgb_images, ms_images):
        """
        rgb_images: [B, 3, H, W] for SkySense-O encoder
        ms_images:  [B, 6, H, W] for SpectralTokenizer
        """
        # Step 1: RGB → SkySense visual features
        with torch.no_grad():
            clip_features = self.visual_encoder(rgb_images, dense=True)
            feat_stage3 = clip_features[-2]  # [B, 1408, 14, 14]
            feat_stage4 = clip_features[-1]  # [B, 2816, 7, 7]

        # Step 2: Spectral features
        spec_tokens = self.spectral_tokenizer(ms_images)  # [B, 2816, H, W]
        spec_s3 = F.interpolate(spec_tokens, size=(14, 14), mode='bilinear', align_corners=False)
        spec_s4 = F.interpolate(spec_tokens, size=(7, 7), mode='bilinear', align_corners=False)

        # Step 3: Cross-attention fusion
        fused_s3 = self.caf_stage3(feat_stage3, spec_s3)
        fused_s4 = self.caf_stage4(feat_stage4, spec_s4)

        return {
            "fused_stage3": fused_s3,  # Optional: pass to visual_guidance conv
            "fused_stage4": fused_s4   # Decoder input (and GLF target)
        }
