# train_scripts/eval_bigearthnet_clip_lora.py

import sys
from pathlib import Path
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_lora.bigearthnet_dataset import BigEarthNetSpectralDataset
from fusion_lora.earthgpt_fuse_classifier_clip_lora import EarthGPTFuseClassifierClipLoRA


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(ROOT / "datasets" / "bigearthnet_s2"),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(
            ROOT
            / "checkpoints"
            / "benet_clip_visual_lora_e15_bs8"
            / "last.pth"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="benet_clip_visual_lora_e15_bs8",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--print-samples", type=int, default=4)
    parser.add_argument("--save-final", action="store_true",
                    help="Save full predictions (y_true, y_pred) into results/final/")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ================================================================
    # Dataset
    # ================================================================
    base_ds = BigEarthNetSpectralDataset(args.data_root, split=args.split)
    if args.num_samples is not None:
        ds = Subset(base_ds, range(args.num_samples))
    else:
        ds = base_ds

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Dataset split={args.split}, samples={len(ds)}")
    class_names = base_ds.class_names
    num_classes = base_ds.num_classes
    print(f"Num classes: {num_classes}")

    # ================================================================
    # Model
    # ================================================================
    print("\n=== Building CLIP+LoRA model (EarthGPTFuseClassifierClipLoRA) ===")
    model = EarthGPTFuseClassifierClipLoRA(
        num_classes=num_classes,
        # keep default lora_rank used in training
    ).to(device)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model_state = ckpt.get("model", ckpt)

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"Loaded state dict with strict=False")
    print(f"  Missing keys   : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()

    # ================================================================
    # Evaluation loop
    # ================================================================
    total_loss = 0.0
    total_correct_bits = 0
    total_bits = 0

    all_true = []
    all_pred = []
    saved_samples = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"].cpu()  # [B, C]

        out = model(batch)
        loss = out["loss_cls"]
        logits = out["logits"]

        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu()
        preds = (probs > args.threshold).float()

        total_correct_bits += (preds == labels).sum().item()
        total_bits += labels.numel()

        all_true.append(labels)
        all_pred.append(preds)

        # Sample examples for inspection
        if len(saved_samples) < args.print_samples:
            for i in range(labels.size(0)):
                if len(saved_samples) < args.print_samples:
                    saved_samples.append(
                        {
                            "true": labels[i].cpu().tolist(),
                            "pred": preds[i].tolist(),
                            "probs": probs[i].tolist(),
                        }
                    )

    # Stack
    y_true = torch.vstack(all_true).numpy()
    y_pred = torch.vstack(all_pred).numpy()

    # ================================================================
    # Compute metrics
    # ================================================================
    avg_loss = total_loss / len(loader)
    bit_acc = total_correct_bits / total_bits

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    results = {
        "exp_name": args.exp_name,
        "split": args.split,
        "avg_loss": avg_loss,
        "bit_accuracy": bit_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(class_names, per_class_f1)),
    }

    # ================================================================
    # Save results
    # ================================================================
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    json_path = results_dir / f"{args.exp_name}_{args.split}_metrics.json"
    samples_path = results_dir / f"{args.exp_name}_{args.split}_samples.json"
    txt_path = results_dir / f"{args.exp_name}_{args.split}.txt"

    # ================================================================
    # Optional FINAL save (full ground-truth, predictions + duplicates)
    # ================================================================
    if args.save_final:
        final_dir = ROOT / "results" / "final" / args.exp_name / args.split
        final_dir.mkdir(parents=True, exist_ok=True)

        # ---- Save predictions ----
        np.save(final_dir / "y_true.npy", y_true)
        np.save(final_dir / "y_pred.npy", y_pred)

        # ---- Duplicate the main result files ----
        # 1. txt summary
        (final_dir / "summary.txt").write_text(
            f"Loss={avg_loss:.4f}, BitAcc={bit_acc:.4f}, "
            f"MicroF1={micro_f1:.4f}, MacroF1={macro_f1:.4f}\n"
        )

        # 2. metrics.json
        (final_dir / "metrics.json").write_text(
            json.dumps(results, indent=4)
        )

        # 3. samples.json
        (final_dir / "samples.json").write_text(
            json.dumps(saved_samples, indent=4)
        )

        print(f"\n[FINAL SAVE] Full results saved to: {final_dir}")


    json_path.write_text(json.dumps(results, indent=4))
    samples_path.write_text(json.dumps(saved_samples, indent=4))
    txt_path.write_text(
        f"Loss={avg_loss:.4f}, BitAcc={bit_acc:.4f}, "
        f"MicroF1={micro_f1:.4f}, MacroF1={macro_f1:.4f}\n"
    )

    print("\nSaved:")
    print(" -", json_path)
    print(" -", samples_path)
    print(" -", txt_path)


if __name__ == "__main__":
    main()
