from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import get_cfg

from fusion_lora.spectral_tokenizer import SpectralTokenizer
from fusion_lora.caf_module import CrossAttentionFusion
from fusion_lora.glf_module import GatedLateFusion

# NEW: import our CLIP LoRA injector utilities
from fusion_lora.clip_lora_injector import (
    add_lora_to_skysense_visual,
    freeze_clip_except_lora,
    LoRALinear,  # we may want this for introspection
)

# -------------------------------------------------------------------------
# Path setup: add SkySense-O repo root so `skysense_o` package is visible
# -------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

SKYSENSE_REPO_ROOT = ROOT / "external" / "skysense_o"
if str(SKYSENSE_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SKYSENSE_REPO_ROOT))

from skysense_o.modeling.backbone.skysense_clip import SkySenseCLIP


# -------------------------------------------------------------------------
# Build SkySense-CLIP model + config
# (same as before, but factored out for reuse)
# -------------------------------------------------------------------------
def build_skysense_clip():
    """
    Build SkySenseCLIP with config from external/skysense_o/configs/skysense_o.yaml.
    The config internally includes base.yaml and the CLIP yaml.
    """
    cfg = get_cfg()

    # Allow new keys before merging skysense_o.yaml (and thus base.yaml)
    cfg.set_new_allowed(True)
    cfg.DATASETS.set_new_allowed(True)
    cfg.MODEL.set_new_allowed(True)
    cfg.INPUT.set_new_allowed(True)

    # Avoid type mismatch for MIN_SIZE_TRAIN when merging
    cfg.INPUT.MIN_SIZE_TRAIN = 384

    # Merge SkySense-O config (which internally includes base.yaml)
    cfg.merge_from_file(str(SKYSENSE_REPO_ROOT / "configs" / "skysense_o.yaml"))

    # Ensure we have a CLIP checkpoint path (relative to external/skysense_o)
    if not hasattr(cfg.MODEL, "CLIP_CKPT_PATH") or cfg.MODEL.CLIP_CKPT_PATH is None:
        cfg.MODEL.CLIP_CKPT_PATH = "pretrain/skysense_clip.pth"

    cfg.freeze()

    # SkySenseCLIP reads CLIP_CFG_PATH + CLIP_CKPT_PATH from cfg
    model = SkySenseCLIP(cfg)
    return model, cfg


# -------------------------------------------------------------------------
# LoRA-based SkySense-CLIP classifier for BigEarthNet
# -------------------------------------------------------------------------
class EarthGPTFuseClassifierClipLoRA(nn.Module):
    """
    Spectral → pseudo-RGB → SkySense-CLIP (with LoRA inside Swin) →
    CAF + GLF → linear classifier.

    This is a scene classification model for BigEarthNet using *true* CLIP
    fine-tuning (via LoRA) on Swin stages 2 and 3.
    """

    def __init__(self, num_classes: int, lora_rank: int = 8):
        super().__init__()

        # --------------- 1) Build SkySense-CLIP ---------------
        clip_model, cfg = build_skysense_clip()
        self.clip_model = clip_model  # keep full CLIP (visual + text), but we only use visual

        # --------------- 2) Inject LoRA into visual Swin ---------------
        # We adapt Swin stages 2 and 3 (0-based indexing), i.e., the two highest-level stages.
        add_lora_to_skysense_visual(
            self.clip_model,
            r=lora_rank,
            alpha=16.0,
            target_stages=(2, 3),
        )

        # --------------- 3) Freeze CLIP except LoRA params ---------------
        freeze_clip_except_lora(self.clip_model)

        # --------------- 4) Spectral tokenizer (6 -> 3) for CLIP input ---------------
        self.spectral_tokenizer = SpectralTokenizer(in_channels=6, out_dim=3)
        self.clip_resolution = (384, 384)

        # --------------- 5) CLIP normalization stats ---------------
        pixel_mean = torch.tensor(cfg.MODEL.CLIP_PIXEL_MEAN).view(1, 3, 1, 1)
        pixel_std = torch.tensor(cfg.MODEL.CLIP_PIXEL_STD).view(1, 3, 1, 1)
        self.register_buffer("clip_pixel_mean", pixel_mean, persistent=False)
        self.register_buffer("clip_pixel_std", pixel_std, persistent=False)

        # --------------- 6) Extra spectral branch for fusion ---------------
        # Project original 6-band tensor to 1024 channels, then downsample to CLIP feature size
        self.spectral_proj = nn.Conv2d(6, 1024, kernel_size=1)

        # --------------- 7) CAF + GLF for feature-level fusion ---------------
        self.caf = CrossAttentionFusion(embed_dim=1024, num_heads=8, lora_rank=lora_rank)
        self.glf = GatedLateFusion(channels=1024)

        # --------------- 8) Simple classifier on top of pooled features ---------------
        # Now that CLIP is truly adapted, we can use a standard linear head here.
        self.classifier = nn.Linear(1024, num_classes)

    def encode_visual(self, x_6band: torch.Tensor) -> torch.Tensor:
        """
        Convenience function: 6-band tensor -> CLIP last visual feature map [B, 1024, h, w]
        """
        # Spectral -> pseudo-RGB, resize to CLIP resolution, and normalize
        pseudo_rgb = self.spectral_tokenizer(x_6band, target_size=self.clip_resolution)
        x_norm = (pseudo_rgb - self.clip_pixel_mean) / self.clip_pixel_std

        # encode_image(..., dense=True) returns multi-scale features; last one is [B, 1024, h, w]
        clip_features = self.clip_model.encode_image(x_norm, dense=True)
        image_feats = clip_features[-1]  # [B, 1024, h, w]
        return image_feats

    def forward(self, batch):
        """
        batch: dict with keys:
          - "image": [B, 6, H, W]
          - "labels": [B, C] (optional, used for loss in training loop)
        """
        x = batch["image"]           # [B, 6, H, W]
        labels = batch.get("labels")

        # ----------------- 1) Visual features from CLIP + LoRA -----------------
        image_feats = self.encode_visual(x)  # [B, 1024, h, w]

        # ----------------- 2) Spectral feature map for fusion -------------------
        spectral_feats = self.spectral_proj(x)  # [B, 1024, H, W]
        spectral_feats = F.interpolate(
            spectral_feats,
            size=image_feats.shape[-2:],       # (h, w)
            mode="bilinear",
            align_corners=False,
        )

        # ----------------- 3) CAF (LoRA attention in fusion module) -------------
        fused = self.caf(visual_feat=image_feats, spectral_feat=spectral_feats)

        # ----------------- 4) GLF (gated late fusion) ---------------------------
        fused = self.glf(fused_feat=fused, visual_feat=image_feats)

        # ----------------- 5) Global pooling + classifier ----------------------
        pooled = fused.mean(dim=(2, 3))        # [B, 1024]
        logits = self.classifier(pooled)       # [B, num_classes]

        if labels is None:
            return logits

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return {"loss_cls": loss, "logits": logits}

    # ---------------------------------------------------------------------
    # Helper methods for introspection (optional, handy for debugging)
    # ---------------------------------------------------------------------
    def get_lora_modules(self):
        """Return a list of (name, module) for all LoRALinear modules in clip_model.visual."""
        lora_modules = []
        for name, module in self.clip_model.visual.named_modules():
            if isinstance(module, LoRALinear):
                lora_modules.append((name, module))
        return lora_modules

    def count_parameters(self):
        """Return (total_params, trainable_params) counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
