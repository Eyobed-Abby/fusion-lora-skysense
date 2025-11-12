#!/usr/bin/env python3
"""
compute_stats_cli.py  for cli use

- Computes per-channel mean/std from prepared .pt tensors
- Accepts --pt-dir explicitly (or --root + --split)
- Writes/updates JSON config with section key and channel order
"""
import json, argparse
from pathlib import Path
import torch

def compute_stats(pt_dir: Path, limit: int = 0):
    pts = sorted(pt_dir.glob("*.pt"))
    if not pts:
        raise SystemExit(f"No .pt tensors found in {pt_dir}")
    if limit > 0:
        pts = pts[:limit]

    m1 = None
    m2 = None
    total_pix = 0
    H = W = None

    for i, p in enumerate(pts, 1):
        X = torch.load(p, map_location="cpu")  # [C,H,W]
        C, H, W = X.shape
        x = X.reshape(C, -1).double()
        if m1 is None:
            m1 = torch.zeros(C, dtype=torch.float64)
            m2 = torch.zeros(C, dtype=torch.float64)
        m1 += x.sum(dim=1)
        m2 += (x * x).sum(dim=1)
        total_pix += x.shape[1]

    mean = (m1 / total_pix)
    var  = (m2 / total_pix) - mean * mean
    std  = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean.tolist(), std.tolist(), H, W

def main():
    ap = argparse.ArgumentParser(description="Compute per-channel mean/std for tensors.")
    ap.add_argument("--pt-dir", help="Directory containing *.pt tensors")
    ap.add_argument("--root", help="If --pt-dir not given, use ROOT/<split>_tensors")
    ap.add_argument("--split", default="train", help="Split tensors to read (default train)")
    ap.add_argument("--limit", type=int, default=0, help="Use first N tensors (0=all)")
    ap.add_argument("--out-config", default="config.json", help="JSON file to write/update")
    ap.add_argument("--section", default="bigearthnet_s2", help="Section key in the JSON")
    ap.add_argument("--channel-order", nargs="+",
                    default=["B04","B03","B02","B08","B11","B12"],
                    help="Channel order corresponding to stats")
    args = ap.parse_args()

    if args.pt_dir:
        pt_dir = Path(args.pt_dir).resolve()
    else:
        if not args.root:
            raise SystemExit("Provide --pt-dir or (--root and --split)")
        pt_dir = Path(args.root).resolve() / f"{args.split}_tensors"

    print(f"Reading tensors from: {pt_dir}")
    mean, std, h, w = compute_stats(pt_dir, limit=args.limit)

    cfg_path = Path(args.out_config).resolve()
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}

    cfg[args.section] = {
        "tile_size": [h, w],
        "channel_order": args.channel_order,
        "mean": mean,
        "std": std
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"✓ Wrote {cfg_path} with mean/std for {args.section}")
    print(f"mean: {mean}\nstd : {std}")

if __name__ == "__main__":
    main()
