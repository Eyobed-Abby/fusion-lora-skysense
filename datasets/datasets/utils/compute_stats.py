#!/usr/bin/env python3
import json
from pathlib import Path
import torch

SECTION = "eurosat_ms"
CHANNEL_ORDER = ["B04","B03","B02","B08","B11","B12"]  # R,G,B,NIR,SWIR1,SWIR2

def compute_stats(pt_dir: Path, limit: int = 0):
    pts = sorted(pt_dir.glob("*.pt"))
    if not pts:
        raise SystemExit(f"No .pt tensors found in {pt_dir}")
    if limit > 0:
        pts = pts[:limit]

    m1 = torch.zeros(6, dtype=torch.float64)
    m2 = torch.zeros(6, dtype=torch.float64)
    total_pix = 0

    for i, p in enumerate(pts, 1):
        X = torch.load(p, map_location="cpu")  # [6,H,W], float32
        C, H, W = X.shape
        x = X.reshape(C, -1).double()
        m1 += x.sum(dim=1)
        m2 += (x * x).sum(dim=1)
        total_pix += H * W

    mean = (m1 / total_pix)
    var  = (m2 / total_pix) - mean * mean
    std  = torch.sqrt(torch.clamp(var, min=1e-12))

    return mean.tolist(), std.tolist(), H, W

def main():
    project_root = Path(__file__).resolve().parents[2]
    ds_root = project_root / "datasets" / "eurosat_ms"
    train_pt = ds_root / "train_tensors"

    mean, std, h, w = compute_stats(train_pt, limit=0)

    cfg_path = project_root / "datasets" / "config.json"
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}

    cfg[SECTION] = {
        "tile_size": [h, w],
        "channel_order": CHANNEL_ORDER,
        "mean": mean,
        "std": std
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"✓ Wrote {cfg_path} with mean/std for {SECTION}")
    print(f"mean: {mean}\nstd : {std}")

if __name__ == "__main__":
    main()
