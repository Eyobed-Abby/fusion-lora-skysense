#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# Sentinel-2 canonical order often used in EuroSAT multiband files
S2_ORDER = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]
# What we want to extract (and how to name files)
TARGET_BANDS = [("B02","B2"), ("B03","B3"), ("B04","B4"), ("B08","B8"), ("B11","B11"), ("B12","B12")]

def guess_band_map(ds: rasterio.io.DatasetReader):
    """
    Try to map physical Sentinel-2 band names to 1-based indices in this dataset.
    We look for band name hints in descriptions/tags; else assume canonical S2 order.
    Returns dict like {"B2": idx, "B3": idx, ...} with 1-based indices.
    """
    name_hints = []
    for i in range(1, ds.count + 1):
        desc = ds.descriptions[i-1] or ""
        tags = ds.tags(i)
        cand = (tags.get("BAND_NAME") or tags.get("NAME") or desc or "").upper()
        # Normalize some variants
        cand = cand.replace("BAND_", "B").replace("B0", "B")  # e.g., BAND_02 -> B2, B02 -> B2
        # Extract the first S2 token we recognize
        found = None
        for s2 in S2_ORDER:
            if s2 in cand:
                found = s2
                break
        name_hints.append(found)

    if all(n is not None for n in name_hints):
        # Build map from S2 name -> index
        name2idx = {}
        for i, n in enumerate(name_hints, start=1):
            # prefer first occurrence
            if n not in name2idx:
                name2idx[n] = i
        return name2idx

    # Fallback: assume canonical ordering across first ds.count bands
    name2idx = {}
    for i in range(min(ds.count, len(S2_ORDER))):
        name2idx[S2_ORDER[i]] = i + 1  # 1-based
    return name2idx

def read_and_maybe_resample(ds, band_index, ref_shape, ref_transform, do_resample: bool):
    """
    Read a band by 1-based index. If do_resample=True, resample to ref_shape/ref_transform.
    Returns (array, transform).
    """
    arr = ds.read(band_index)  # 2D
    transform = ds.transform

    if not do_resample:
        return arr, transform

    # If same shape, skip
    if arr.shape == ref_shape:
        return arr, transform

    dst = rasterio.io.MemoryFile().open(
        driver="GTiff",
        height=ref_shape[0],
        width=ref_shape[1],
        count=1,
        dtype=arr.dtype
    )  # temporary holder just to allocate shape; we only need the array

    out = rasterio.numpy.empty(ref_shape, dtype=arr.dtype)
    reproject(
        source=arr,
        destination=out,
        src_transform=transform,
        src_crs=ds.crs,
        dst_transform=ref_transform,
        dst_crs=ds.crs,
        resampling=Resampling.bilinear
    )
    return out, ref_transform

def write_single_band(out_path: Path, array, ref_profile, transform):
    profile = ref_profile.copy()
    profile.update({
        "count": 1,
        "height": array.shape[0],
        "width": array.shape[1],
        "transform": transform,
        "driver": "GTiff"
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array, 1)

def process_split(split_dir: Path, resample_swirs: bool, overwrite: bool):
    """
    For each multiband .tif in split_dir, create:
    split_dir/sample_xxxxx/
      B02.tif B03.tif B04.tif B08.tif B11.tif B12.tif
    """
    tifs = sorted(split_dir.glob("*.tif"))
    if not tifs:
        print(f"(i) No .tif files found in {split_dir}")
        return

    for tif in tifs:
        sample_id = tif.stem  # e.g., sample_00001
        sample_dir = split_dir / sample_id
        # Skip if already exists (and not overwriting)
        if sample_dir.exists() and not overwrite:
            # quick check: if all six exist, continue
            if all((sample_dir / f"{name}.tif").exists() for name, _ in TARGET_BANDS):
                print(f"✓ Skipping existing {sample_dir}")
                continue

        with rasterio.open(tif) as ds:
            name2idx = guess_band_map(ds)

            # Choose a 10 m reference band for shape/transform (prefer B4 then B2)
            ref_idx = name2idx.get("B4") or name2idx.get("B2")
            if ref_idx is None:
                raise RuntimeError(f"Could not find a 10 m reference band (B4/B2) in {tif}")

            ref = ds.read(ref_idx)
            ref_shape = ref.shape
            ref_transform = ds.transform

            # Keep a reference profile for output files
            ref_profile = ds.profile.copy()
            ref_profile["compress"] = "deflate"  # small files

            # Extract targets
            for out_name, s2name in TARGET_BANDS:
                idx = name2idx.get(s2name)
                if idx is None or idx > ds.count:
                    raise RuntimeError(f"Band {s2name} not found in {tif}")

                do_resample = resample_swirs and s2name in ("B11", "B12")
                band_arr, band_transform = read_and_maybe_resample(
                    ds, idx, ref_shape, ref_transform, do_resample=do_resample
                )

                out_path = sample_dir / f"{out_name}.tif"
                if out_path.exists() and not overwrite:
                    continue

                write_single_band(out_path, band_arr, ref_profile, band_transform)

        print(f"✓ Wrote {sample_dir} (from {tif.name})")

def main():
    parser = argparse.ArgumentParser(description="Extract 6 Sentinel-2 bands per sample into per-band GeoTIFFs.")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Path to project root containing datasets/eurosat_ms/{train,test}. "
                             "Default: parent of this script's directory.")
    parser.add_argument("--no-resample-swirs", action="store_true",
                        help="Do NOT resample B11/B12 to match 10 m reference.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-band files.")
    args = parser.parse_args()

    # Resolve project root
    if args.project_root:
        root = Path(args.project_root).resolve()
    else:
        root = Path(__file__).resolve().parents[1]  # Project root = parent of scripts/

    eurosat_root = root / "datasets" / "eurosat_ms"
    train_dir = eurosat_root / "train"
    test_dir  = eurosat_root / "test"

    print(f"Project root: {root}")
    print(f"Processing train: {train_dir}")
    process_split(train_dir, resample_swirs=not args.no_resample_swirs, overwrite=args.overwrite)

    if test_dir.exists():
        print(f"\nProcessing test:  {test_dir}")
        process_split(test_dir,  resample_swirs=not args.no_resample_swirs, overwrite=args.overwrite)
    else:
        print("(i) No test directory found; skipping.")

if __name__ == "__main__":
    main()