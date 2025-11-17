# train_bigearthnet_cls.py

import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from fusion_lora.bigearthnet_dataset import BigEarthNetSpectralDataset
from fusion_lora.earthgpt_fuse_classifier_LoRA import EarthGPTFuseClassifier


def move_batch_to_device(batch, device):
    for k in batch:
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device)
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
            scaler.scale(loss).step(optimizer)
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

        # simple multi-label accuracy: how many label bits correct
        total_correct += (preds == labels).sum().item()
        total_labels += labels.numel()

    avg_loss = total_loss / max(1, len(loader))
    acc = total_correct / max(1, total_labels)
    return avg_loss, acc


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
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Datasets & loaders
    root = Path(args.data_root)

    train_set = BigEarthNetSpectralDataset(root, split="train")
    val_set = BigEarthNetSpectralDataset(root, split="val")

    print("Train samples:", len(train_set))
    print("Val samples:", len(val_set))
    print("Num classes:", train_set.num_classes)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 2) Model (LoRA + spectral fusion)
    model = EarthGPTFuseClassifier(
        num_classes=train_set.num_classes,
        lora_rank=args.lora_rank,
    ).to(device)

    # 3) Optimizer (only LoRA + fusion params have requires_grad=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameters:", sum(p.numel() for p in trainable_params))

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # Optional AMP
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        print(f"[epoch {epoch}] train loss: {train_loss:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] val loss: {val_loss:.4f}, bit-accuracy: {val_acc:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / f"best_epoch_{epoch}.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"  → Saved new best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
