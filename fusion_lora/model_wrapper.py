import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_lora.spectral_tokenizer import SpectralTokenizer


class FusionLoRAWrapper(nn.Module):
    """
    wrapper:
    - Takes 6-band multispectral images
    - Uses SpectralTokenizer to map 6 -> 3 channels
    - Feeds pseudo-RGB into SkySense-CLIP visual encoder
    - (LoRA / CAF / GLF can be added on top later)
    """
    def __init__(self, visual_encoder, clip_resolution=(384, 384)):
        """
        Args:
            visual_encoder: callable like clip_model.encode_image(img, dense=True)
            clip_resolution: spatial size expected by CLIP/SkySense-O (384, 384)
        """
        super().__init__()
        self.visual_encoder = visual_encoder
        self.clip_resolution = clip_resolution

        # 6 -> 3 pseudo-RGB
        self.spectral_tokenizer = SpectralTokenizer(in_channels=6, out_dim=3)

    def forward(self, ms_images):
        """
        ms_images: [B, 6, H, W] multispectral patch
        Returns:
            clip_features: list of feature maps as returned by encode_image(dense=True)
                           e.g. [res2, res3, res4, res5_projected]
        """
        B, C, H, W = ms_images.shape
        assert C == 6, f"Expected 6 spectral channels, got {C}"

        # Step 1: 6-band -> 3-channel pseudo RGB
        pseudo_rgb = self.spectral_tokenizer(ms_images, target_size=self.clip_resolution)
        # pseudo_rgb: [B, 3, clip_H, clip_W]

        # Step 2: CLIP visual encoder (no change)
        clip_features = self.visual_encoder(pseudo_rgb, dense=True)

        return clip_features
