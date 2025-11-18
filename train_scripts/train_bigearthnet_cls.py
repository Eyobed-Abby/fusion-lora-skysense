# train_bigearthnet_cls.py
#
# Usage examples:
#   # 1) Quick sanity check on small subset (CPU or GPU)
#   python train_scripts/train_bigearthnet_cls.py \
#       --data-root datasets/bigearthnet_s2 \
#       --batch-size 32 \
#       --epochs 2 \
#       --debug-small 256 \
#       --exp-name debug_run \
#       --save-lora-only
#
#   # 2) Full training (when ready)
#   python train_scripts/train_bigearthnet_cls.py \
#       --data-root datasets/bigearthnet_s2 \
#       --batch-size 64 \
#       --epochs 10 \
#       --exp-name full_run \
#       --save-lora-only
#
# Checkpoints will be saved under: <save-dir>/<exp-name>/

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

from fusion_lora.bigearthnet_dataset import BigEarthNetSpectralDataset
from fusion_lora.earthgpt_fuse_classifier_LoRA import EarthGPTFuseClassifier


# ------------------------- Utilities ------------------------- #

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility (useful for sanity checks)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    for k in batch:
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device, non_blocking=True)
    return batch


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(batch)
                loss = out["loss_cls"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(batch)
            loss = out["loss_cls"]
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(f"[train] iter {i+1}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_labels = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        loss = out["loss_cls"]
        total_loss += loss.item()

        logits = out["logits"]
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        labels = batch["labels"]

        # simple multi-label bit accuracy
        total_correct += (preds == labels).sum().item()
        total_labels += labels.numel()

    avg_loss = total_loss / max(1, len(loader))
    acc = total_correct / max(1, total_labels)
    return avg_loss, acc


# ------------------------- Main ------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/bigearthnet_s2",
        help="Root folder with *_tensors and *_labels.csv",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Base directory for saving checkpoints",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="bigearthnet_lora",
        help="Subfolder name for this run (useful to separate debug/full runs)",
    )
    parser.add_argument(
        "--debug-small",
        type=int,
        default=0,
        help="If >0, use only this many samples from train and val for sanity check.",
    )
    parser.add_argument(
        "--save-lora-only",
        action="store_true",
        help="Additionally save compact checkpoints with only trainable (LoRA+fusion) weights.",
    )
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Datasets & loaders
    root = Path(args.data_root)

    print("\n=== Building datasets ===")
    full_train_set = BigEarthNetSpectralDataset(root, split="train")
    full_val_set = BigEarthNetSpectralDataset(root, split="val")

    print("Full train samples:", len(full_train_set))
    print("Full val samples  :", len(full_val_set))
    print("Num classes       :", full_train_set.num_classes)

    # If debug_small > 0, restrict to small subset for quick sanity check
    if args.debug_small > 0:
        n_train = min(args.debug_small, len(full_train_set))
        n_val = min(max(args.debug_small // 4, 1), len(full_val_set))  # smaller val
        train_indices = list(range(n_train))
        val_indices = list(range(n_val))

        train_set = Subset(full_train_set, train_indices)
        val_set = Subset(full_val_set, val_indices)

        print(
            f"\n[DEBUG MODE] Using subsets: "
            f"{n_train} train samples, {n_val} val samples."
        )
    else:
        train_set = full_train_set
        val_set = full_val_set

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 2) Model (LoRA + spectral fusion)
    print("\n=== Building model (SkySense-CLIP + Spectral Fusion + LoRA) ===")
    model = EarthGPTFuseClassifier(
        num_classes=full_train_set.num_classes,
        lora_rank=args.lora_rank,
    ).to(device)

    # 3) Trainable params (only LoRA + fusion modules should be requires_grad=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M")
    print("\nTrainable parameter names (for LoRA+fusion):")
    for name in trainable_names:
        print("  ", name)

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # Optional AMP
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # 4) Checkpoint directory
    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints will be saved to: {save_dir}")

    best_val_loss = float("inf")

    # 5) Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        print(f"[epoch {epoch}] train loss: {train_loss:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] val loss: {val_loss:.4f}, bit-accuracy: {val_acc:.4f}")

        # Always save "last" checkpoint (for debugging / resume)
        last_ckpt_path = save_dir / "last.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            },
            last_ckpt_path,
        )
        print(f"  → Saved last checkpoint to {last_ckpt_path}")

        # Save best checkpoint (full model state)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = save_dir / f"best_epoch_{epoch}.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
            print(f"  → Saved new BEST checkpoint to {best_ckpt_path}")

            # Optionally save a compact LoRA+fusion-only checkpoint
            if args.save_lora_only:
                full_state = model.state_dict()
                trainable_name_set = set(trainable_names)
                lora_fusion_state = {
                    k: v for k, v in full_state.items() if k in trainable_name_set
                }
                lora_ckpt_path = save_dir / f"best_epoch_{epoch}_lora_only.pth"
                torch.save(
                    {
                        "lora_fusion_state_dict": lora_fusion_state,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "num_classes": full_train_set.num_classes,
                    },
                    lora_ckpt_path,
                )
                print(f"  → Saved compact LoRA+fusion checkpoint to {lora_ckpt_path}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
