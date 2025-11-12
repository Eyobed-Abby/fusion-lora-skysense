#!/usr/bin/env python3
"""
prepare_spectral_data_cli.py  to be used by ci

- Converts sample folders containing {B02,B03,B04,B08,B11,B12}.tif to tensors [6,S,S]
- Accepts explicit --src (subset root) and --dst (tensor root)
- Supports --splits to choose which splits to process
- Resizes to --size (default 256), normalizes reflectance to [0,1]
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

EXPECTED = {"B02.tif","B03.tif","B04.tif","B08.tif","B11.tif","B12.tif"}
ORDER    = ["B04.tif","B03.tif","B02.tif","B08.tif","B11.tif","B12.tif"]  # R,G,B,NIR,SWIR1,SWIR2
REF_BAND = "B04.tif"  # 10 m Red

def upsample_to_ref(arr, src_transform, crs, ref_shape, ref_transform):
    out = np.empty(ref_shape, dtype=arr.dtype)
    reproject(
        source=arr,
        destination=out,
        src_transform=src_transform,
        src_crs=crs,
        dst_transform=ref_transform,
        dst_crs=crs,
        resampling=Resampling.bilinear
    )
    return out

def load_sample_dir(sample_dir: Path, target_size: int):
    with rasterio.open(sample_dir / REF_BAND) as ref_ds:
        ref = ref_ds.read(1)
        ref_shape = ref.shape
        ref_transform = ref_ds.transform
        ref_crs = ref_ds.crs

    chips = []
    for fname in ORDER:
        with rasterio.open(sample_dir / fname) as ds:
            band = ds.read(1)
            if fname in ("B11.tif","B12.tif") and band.shape != ref_shape:
                band = upsample_to_ref(band, ds.transform, ds.crs, ref_shape, ref_transform)
            chips.append(band.astype(np.float32))

    x = np.stack(chips, axis=0)  # [6,H,W]
    if x.max() > 1.5:
        x = x / 10000.0
    x = np.clip(x, 0.0, 1.0)

    X = torch.from_numpy(x).unsqueeze(0)
    X = F.interpolate(X, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return X.squeeze(0).contiguous().float()

def process_split(split_dir: Path, out_dir: Path, size: int, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
    for sd in sample_dirs:
        band_files = {p.name for p in sd.glob("*.tif")}
        if not EXPECTED.issubset(band_files):
            print(f"⚠️  Skipping {sd.name}: missing {sorted(EXPECTED - band_files)}")
            continue
        out_pt = out_dir / f"{sd.name}.pt"
        if out_pt.exists() and not overwrite:
            try:
                test = torch.load(out_pt, map_location="cpu")
                if test.shape[0] == 6:
                    print(f"✓ {out_pt.name} (exists)")
                    continue
            except Exception:
                out_pt.unlink(missing_ok=True)
        X = load_sample_dir(sd, target_size=size)
        torch.save(X, out_pt)
        print(f"✓ {out_pt.name}  shape={tuple(X.shape)}  range=({X.min().item():.3f},{X.max().item():.3f})")

def main():
    ap = argparse.ArgumentParser(description="Prepare tensors for BigEarthNet-S2 subset.")
    ap.add_argument("--src", required=True, help="Subset root with train/validation/test")
    ap.add_argument("--dst", required=True, help="Output root for *_tensors folders")
    ap.add_argument("--splits", default="train,validation,test",
                    help="Comma list of splits to process")
    ap.add_argument("--size", type=int, default=256, help="Output tile size (default 256)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing tensors")
    args = ap.parse_args()

    SRC = Path(args.src).resolve()
    DST = Path(args.dst).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"SRC={SRC}\nDST={DST}\nSplits={splits}\nSize={args.size}")
    for s in splits:
        in_dir  = SRC / s
        out_dir = DST / f"{s}_tensors"
        if in_dir.exists():
            print(f"\n-- {s} --")
            process_split(in_dir, out_dir, size=args.size, overwrite=args.overwrite)
        else:
            print(f"(i) Split not found: {in_dir}")

if __name__ == "__main__":
    main()
