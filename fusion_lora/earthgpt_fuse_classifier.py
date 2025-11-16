from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import get_cfg

# Path to external/skysense_o
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
EXTERNAL_DIR = ROOT / "external" / "skysense_o"
sys.path.insert(0, str(EXTERNAL_DIR / "skysense_o"))

from skysense_o.modeling.backbone.skysense_clip import SkySenseCLIP
from fusion_lora.spectral_tokenizer import SpectralTokenizer


def build_skysense_clip():
    cfg = get_cfg()
    cfg.merge_from_file(str(EXTERNAL_DIR / "configs" / "skysense_o.yaml"))
    cfg.freeze()
    model = SkySenseCLIP(cfg)
    return model, cfg


class EarthGPTFuseClassifier(nn.Module):
    """
    Spectral → pseudo-RGB → SkySense-CLIP → pooled features → classifier.

    This is a scene classification model for BigEarthNet.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.clip_model, cfg = build_skysense_clip()

        # spectral → pseudo-RGB
        self.spectral_tokenizer = SpectralTokenizer(in_channels=6, out_dim=3)
        self.clip_resolution = (384, 384)

        # use same normalization as SkySense-O
        pixel_mean = torch.tensor(cfg.MODEL.CLIP_PIXEL_MEAN).view(1, 3, 1, 1)
        pixel_std = torch.tensor(cfg.MODEL.CLIP_PIXEL_STD).view(1, 3, 1, 1)
        self.register_buffer("clip_pixel_mean", pixel_mean, False)
        self.register_buffer("clip_pixel_std", pixel_std, False)

        # simple classifier on top of global pooled visual features
        # clip_features[-1] is [B, 1024, H', W']
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, batch):
        """
        batch: dict with keys:
          - "image": [B, 6, H, W]
          - "labels": [B, C] (optional, used for loss in training loop)
        """
        x = batch["image"]              # [B, 6, H, W]
        labels = batch.get("labels")    # [B, C] or None

        # spectral → pseudo-RGB
        pseudo_rgb = self.spectral_tokenizer(x, target_size=self.clip_resolution)
        x_norm = (pseudo_rgb - self.clip_pixel_mean) / self.clip_pixel_std

        # visual features from SkySense-CLIP
        clip_features = self.clip_model.encode_image(x_norm, dense=True)
        image_feats = clip_features[-1]          # [B, 1024, h, w]

        # global average pool
        pooled = image_feats.mean(dim=(2, 3))   # [B, 1024]

        logits = self.classifier(pooled)        # [B, num_classes]

        if labels is None:
            return logits

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return {"loss_cls": loss, "logits": logits}
