from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import get_cfg

from fusion_lora.spectral_tokenizer import SpectralTokenizer
from fusion_lora.caf_module import CrossAttentionFusion
from fusion_lora.glf_module import GatedLateFusion
from fusion_lora.lora_layers import LoRALayer

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
# -------------------------------------------------------------------------

def build_skysense_clip():
    """
    SkySenseCLIP can auto-load the pretrained weights.
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

    # Make sure we have a ckpt path defined (relative to external/skysense_o)
    if not hasattr(cfg.MODEL, "CLIP_CKPT_PATH") or cfg.MODEL.CLIP_CKPT_PATH is None:
        cfg.MODEL.CLIP_CKPT_PATH = "pretrain/skysense_clip.pth"

    cfg.freeze()

    # NEW: pass the cfg directly — SkySenseCLIP will read CLIP_CFG_PATH & CLIP_CKPT_PATH
    model = SkySenseCLIP(cfg)

    return model, cfg

#old build_skysense_clip()
# def build_skysense_clip():
#     cfg = get_cfg()

#     # Allow new keys before merging skysense_o.yaml (and thus base.yaml)
#     cfg.set_new_allowed(True)
#     cfg.DATASETS.set_new_allowed(True)
#     cfg.MODEL.set_new_allowed(True)
#     cfg.INPUT.set_new_allowed(True)

#     # Avoid type mismatch for MIN_SIZE_TRAIN when merging
#     cfg.INPUT.MIN_SIZE_TRAIN = 384

#     # Merge SkySense-O config (which internally includes base.yaml)
#     cfg.merge_from_file(str(SKYSENSE_REPO_ROOT / "configs" / "skysense_o.yaml"))
#     cfg.freeze()

#     # Build absolute path to the CLIP config YAML
#     # base.yaml: MODEL.CLIP_CFG_PATH: "skysense_o/modeling/backbone/clip_config.yml"
#     clip_cfg_rel = cfg.MODEL.CLIP_CFG_PATH
#     clip_cfg_path = SKYSENSE_REPO_ROOT / clip_cfg_rel

#     # SkySenseCLIP expects a path string to its YAML
#     model = SkySenseCLIP(str(clip_cfg_path))

#     return model, cfg


# -------------------------------------------------------------------------
# LoRA-based SkySense-CLIP classifier for BigEarthNet
# -------------------------------------------------------------------------
class EarthGPTFuseClassifier(nn.Module):
    """
    Spectral → pseudo-RGB → SkySense-CLIP → CAF+GLF (LoRA) → classifier (LoRA).

    This is a scene classification model for BigEarthNet.
    """

    def __init__(self, num_classes: int, lora_rank: int = 8):
        super().__init__()
        self.clip_model, cfg = build_skysense_clip()

        # --------------- Freeze CLIP backbone ---------------
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        # --------------- Spectral tokenizer (6 -> 3) for CLIP input ---------------
        self.spectral_tokenizer = SpectralTokenizer(in_channels=6, out_dim=3)
        self.clip_resolution = (384, 384)

        # --------------- CLIP normalization stats ---------------
        pixel_mean = torch.tensor(cfg.MODEL.CLIP_PIXEL_MEAN).view(1, 3, 1, 1)
        pixel_std = torch.tensor(cfg.MODEL.CLIP_PIXEL_STD).view(1, 3, 1, 1)
        self.register_buffer("clip_pixel_mean", pixel_mean, False)
        self.register_buffer("clip_pixel_std", pixel_std, False)

        # --------------- Extra spectral branch for fusion ---------------
        # Project original 6-band tensor to 1024 channels, then downsample
        self.spectral_proj = nn.Conv2d(6, 1024, kernel_size=1)

        # --------------- CAF + GLF for feature-level LoRA fusion ---------------
        self.caf = CrossAttentionFusion(embed_dim=1024, num_heads=8, lora_rank=lora_rank)
        self.glf = GatedLateFusion(channels=1024)

        # --------------- LoRA classifier on top of pooled features ---------------
        self.classifier = LoRALayer(
            in_features=1024,
            out_features=num_classes,
            r=lora_rank,
            alpha=1.0,
            dropout=0.0,
            bias=True,
        )

    def forward(self, batch):
        """
        batch: dict with keys:
          - "image": [B, 6, H, W]
          - "labels": [B, C] (optional, used for loss in training loop)
        """
        x = batch["image"]              # [B, 6, H, W]
        labels = batch.get("labels")    # [B, C] or None

        # ----------------- 1) Spectral -> pseudo-RGB -----------------
        pseudo_rgb = self.spectral_tokenizer(x, target_size=self.clip_resolution)
        x_norm = (pseudo_rgb - self.clip_pixel_mean) / self.clip_pixel_std

        # ----------------- 2) CLIP visual features -------------------
        # encode_image(..., dense=True) returns multi-scale features; last one is [B, 1024, h, w]
        clip_features = self.clip_model.encode_image(x_norm, dense=True)
        image_feats = clip_features[-1]          # [B, 1024, h, w]

        # ----------------- 3) Spectral feature map for fusion --------
        # Project 6 bands -> 1024 channels and match spatial size of CLIP features
        spectral_feats = self.spectral_proj(x)   # [B, 1024, H, W]
        spectral_feats = F.interpolate(
            spectral_feats,
            size=image_feats.shape[-2:],         # (h, w)
            mode="bilinear",
            align_corners=False,
        )

        # ----------------- 4) CAF (LoRA attention) -------------------
        fused = self.caf(visual_feat=image_feats, spectral_feat=spectral_feats)

        # ----------------- 5) GLF (gated late fusion) ----------------
        fused = self.glf(fused_feat=fused, visual_feat=image_feats)

        # ----------------- 6) Global pooling + LoRA classifier -------
        pooled = fused.mean(dim=(2, 3))          # [B, 1024]
        logits = self.classifier(pooled)         # [B, num_classes]

        if labels is None:
            return logits

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return {"loss_cls": loss, "logits": logits}
