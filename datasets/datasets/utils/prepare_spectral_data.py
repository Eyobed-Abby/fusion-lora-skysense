#!/usr/bin/env python3
"""
BigEarthNet-S2 Spectral Data Preparation

Processes BigEarthNet-S2 satellite imagery:
- Loads 6 spectral bands (B02, B03, B04, B08, B11, B12)
- Stacks to [6, H, W] tensor
- Normalizes reflectance: img / 10000.0
- Resizes to 256x256 (default)
- Saves as .pt tensors
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
REF_BAND = "B04.tif"  # use 10 m Red band as reference for shape

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
    # Open reference band (B04) to get reference shape/grid
    with rasterio.open(sample_dir / REF_BAND) as ref_ds:
        ref = ref_ds.read(1)
        ref_shape = ref.shape
        ref_transform = ref_ds.transform
        ref_crs = ref_ds.crs

    chips = []
    for fname in ORDER:
        path = sample_dir / fname
        with rasterio.open(path) as ds:
            band = ds.read(1)
            # Upsample SWIRs (B11,B12) if needed
            if fname in ("B11.tif","B12.tif") and band.shape != ref_shape:
                band = upsample_to_ref(band, ds.transform, ds.crs, ref_shape, ref_transform)
            chips.append(band.astype(np.float32))

    x = np.stack(chips, axis=0)  # [6,H,W]

    # Reflectance scaling to [0,1]
    if x.max() > 1.5:
        x = x / 10000.0
    x = np.clip(x, 0.0, 1.0)

    # Resize to target_size using bilinear on tensors
    X = torch.from_numpy(x).unsqueeze(0)  # [1,6,H,W]
    X = F.interpolate(X, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return X.squeeze(0).contiguous().float()  # [6,S,S]

def process_split(split_dir: Path, out_dir: Path, size: int, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sample folders live directly under split_dir
    sample_dirs = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
    if not sample_dirs:
        print(f"(i) No sample folders found in {split_dir}")
        return

    for sd in sample_dirs:
        # quick integrity check
        band_files = {p.name for p in sd.glob("*.tif")}
        if not EXPECTED.issubset(band_files):
            print(f"⚠️  Skipping {sd.name}: missing bands {sorted(EXPECTED - band_files)}")
            continue

        out_pt = out_dir / f"{sd.name}.pt"   # sample_xxxxx.pt
        if out_pt.exists() and not overwrite:
            try:
                test_load = torch.load(out_pt, map_location="cpu")
                if test_load.shape[0] == 6:
                    print(f"✓ {out_pt.name} (exists)")
                    continue
            except Exception:
                print(f"⚠️  {out_pt.name} corrupted, recreating...")
                out_pt.unlink()

        X = load_sample_dir(sd, target_size=size)   # [6,S,S]
        torch.save(X, out_pt)
        print(f"✓ Saved {out_pt.name}  shape={tuple(X.shape)}  range=({X.min().item():.3f},{X.max().item():.3f})")

def main():
    ap = argparse.ArgumentParser(description="Prepare BigEarthNet-S2 per-sample tensors [6,H,W].")
    ap.add_argument("--size", type=int, default=256, help="Output tile size (default: 256)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    ds_root = project_root / "datasets" / "bigearthnet_s2"

    train_in = ds_root / "train"
    val_in = ds_root / "validation"
    test_in = ds_root / "test"
    
    train_out = ds_root / "train_tensors"
    val_out = ds_root / "validation_tensors"
    test_out = ds_root / "test_tensors"

    print(f"Project root: {project_root}")
    print(f"Dataset: BigEarthNet-S2")
    print(f"Preparing tensors at size {args.size}x{args.size}")
    print(f"Band order: R,G,B,NIR,SWIR1,SWIR2")
    print(f"Normalization: reflectance / 10000.0")

    if train_in.exists():
        print(f"\n-- Train --")
        process_split(train_in, train_out, size=args.size, overwrite=args.overwrite)
    
    if val_in.exists():
        print(f"\n-- Validation --")
        process_split(val_in, val_out, size=args.size, overwrite=args.overwrite)
    
    if test_in.exists():
        print(f"\n-- Test --")
        process_split(test_in, test_out, size=args.size, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
