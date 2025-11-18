# train_scripts/demo_inference_bigearthnet.py

from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

# -------------------------------------------------------------------
# Make repo importable when running as a script
# -------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fusion_lora.bigearthnet_dataset import BigEarthNetSpectralDataset
from fusion_lora.earthgpt_fuse_classifier_LoRA import EarthGPTFuseClassifier


def main():
    data_root = Path("datasets/bigearthnet_s2")
    ckpt_path = Path("checkpoints/debug_run/best_epoch_2_lora_only.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----------------- 1) Dataset & loader -----------------
    ds = BigEarthNetSpectralDataset(data_root, split="val")
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    # ----------------- 2) Build base model -----------------
    print("\n=== Building model (this loads SkySense-CLIP pretrained) ===")
    model = EarthGPTFuseClassifier(num_classes=ds.num_classes, lora_rank=8).to(device)
    model.eval()

    # ----------------- 3) Load LoRA+fusion weights only -----------------
    print(f"Loading LoRA+fusion weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    lora_state = ckpt["lora_fusion_state_dict"]

    # We only overwrite matching keys (LoRA + fusion). Other weights
    # remain as initialized and loaded from skysense_clip.pth.
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    print(f"Loaded LoRA/fusion state dict.")
    print(f"  Missing keys   : {len(missing)} (expected, these are frozen parts)")
    print(f"  Unexpected keys: {len(unexpected)}")

    # ----------------- 4) Run one batch for demo -----------------
    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["labels"]

    with torch.no_grad():
        out = model({"image": images, "labels": None})
        logits = out  # forward returns logits when labels=None
        probs = torch.sigmoid(logits).cpu()

    print("\n=== Sample predictions (indices) ===")
    for i in range(images.size(0)):
        true_idx = labels[i].nonzero(as_tuple=True)[0].tolist()
        pred_idx = (probs[i] > 0.5).nonzero(as_tuple=True)[0].tolist()
        print(f"  Sample {i}:")
        print(f"    True labels idx : {true_idx}")
        print(f"    Predicted idx   : {pred_idx}")


if __name__ == "__main__":
    main()
