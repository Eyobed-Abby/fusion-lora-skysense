# train_scripts/demo_inference_bigearthnet.py
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

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

    ds = BigEarthNetSpectralDataset(data_root, split="val")
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    # build model
    model = EarthGPTFuseClassifier(num_classes=ds.num_classes, lora_rank=8).to(device)

    # load LoRA + fusion only
    print(f"Loading LoRA+fusion weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    lora_state = ckpt["lora_fusion_state_dict"]
    model_state = model.state_dict()
    model_state.update(lora_state)
    model.load_state_dict(model_state)
    model.eval()

    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["labels"]

    with torch.no_grad():
        out = model({"image": images, "labels": None})
        logits = out  # because labels=None, forward returns logits directly
        probs = torch.sigmoid(logits).cpu()

    print("Sample predictions:")
    for i in range(images.size(0)):
        true_indices = labels[i].nonzero(as_tuple=True)[0].tolist()
        pred_indices = (probs[i] > 0.5).nonzero(as_tuple=True)[0].tolist()
        print(f"  Sample {i}:")
        print(f"    True labels idx : {true_indices}")
        print(f"    Predicted idx   : {pred_indices}")


if __name__ == "__main__":
    main()
