import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from fusion_lora.bigearthnet_dataset import BigEarthNetSpectralDataset
from fusion_lora.earthgpt_fuse_classifier_LoRA import EarthGPTFuseClassifier


def collate_fn(batch):
    # simple batcher: stack tensors
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {"image": images, "labels": labels}


def main():
    root = "datasets/bigearthnet_s2"
    train_ds = BigEarthNetSpectralDataset(root, split="train")
    val_ds = BigEarthNetSpectralDataset(root, split="val")

    num_classes = train_ds.num_classes
    model = EarthGPTFuseClassifier(num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss = out["loss_cls"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: train loss = {total_loss / len(train_loader):.4f}")

        # TODO: add validation loop (compute mAP, F1, etc.)

if __name__ == "__main__":
    main()
