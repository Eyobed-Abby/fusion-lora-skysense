# train_scripts/demo_inference_bigearthnet.py

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
# Make repo root importable so `fusion_lora.*` works when run as a script
# -------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent  # fusion-lora-skysense/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_lora.bigearthnet_dataset import BigEarthNetSpectralDataset
from fusion_lora.earthgpt_fuse_classifier_LoRA import EarthGPTFuseClassifier


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_root = ROOT / "datasets" / "bigearthnet_s2"
    ckpt_path = ROOT / "checkpoints" / "debug_run" / "best_epoch_2_lora_only.pth"

    # 1) Dataset & loader (val split for demo)
    ds = BigEarthNetSpectralDataset(data_root, split="val")
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

    print(f"Val samples: {len(ds)}")
    print(f"Num classes: {ds.num_classes}")

    # 2) Build model (loads SkySense-CLIP pretrained internally)
    print("\n=== Building model (this loads SkySense-CLIP pretrained) ===")
    model = EarthGPTFuseClassifier(
        num_classes=ds.num_classes,
        lora_rank=8,
    ).to(device)

    # 3) Load LoRA + fusion checkpoint only
    print(f"Loading LoRA+fusion weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_state = ckpt.get("model", ckpt)

    # strict=False because we only stored LoRA+fusion params, not full CLIP
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print("Loaded LoRA/fusion state dict.")
    print(f"  Missing keys   : {len(missing)} (expected, these are frozen parts)")
    print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()

    # 4) Take one batch for demo inference
    batch = next(iter(loader))
    images = batch["image"].to(device)   # [B, 6, H, W]
    labels = batch["labels"]             # [B, C]

    with torch.no_grad():
        out = model({"image": images, "labels": None})
        logits = out                      # forward() returns logits when labels=None
        probs = torch.sigmoid(logits).cpu()

    # 5) Use class_names from dataset if available
    class_names = getattr(ds, "class_names", None)
    threshold = 0.5

    print("\n=== Sample predictions ===")
    for i in range(images.size(0)):
        true_idx = labels[i].nonzero(as_tuple=True)[0].tolist()
        pred_idx = (probs[i] > threshold).nonzero(as_tuple=True)[0].tolist()

        if class_names is not None:
            true_labels = [class_names[j] for j in true_idx]
            pred_labels = [class_names[j] for j in pred_idx]
        else:
            true_labels = true_idx
            pred_labels = pred_idx

        print(f"\n  Sample {i}:")
        print(f"    True labels : {true_labels}")
        print(f"    Pred labels : {pred_labels}")


if __name__ == "__main__":
    main()
