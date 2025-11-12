#!/usr/bin/env python3
"""
prepare_bigearthnet_s2_cli.py  to work on cli

- Selects bands and builds train/validation/test subsets from extracted BigEarthNet-S2
- Supports caps per split (--max-train/--max-val/--max-test) for down-sampling
- Places files via --link {hard,soft,copy} (hardlink by default)
- Accepts explicit --src/--dst/--metadata paths (no hardcoded repo layout)
- Provides --dry-run mode
"""
import os, csv, json, argparse, shutil, random
from pathlib import Path
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Build BigEarthNet-S2 subset with CSV labels.")
    ap.add_argument("--src", required=True, help="Extracted BigEarthNet-S2 root")
    ap.add_argument("--dst", required=True, help="Output dataset root (will create)")
    ap.add_argument("--metadata", required=True, help="metadata.parquet path")
    ap.add_argument("--bands", nargs="+", default=["B02","B03","B04","B08","B11","B12"],
                    help="Band list to include (default: 6-band set)")
    ap.add_argument("--max-train", type=int, default=None, help="Cap train samples")
    ap.add_argument("--max-val",   type=int, default=None, help="Cap val samples")
    ap.add_argument("--max-test",  type=int, default=None, help="Cap test samples")
    ap.add_argument("--link", choices=["hard","soft","copy"], default="hard",
                    help="Placement method: hardlink, symlink, or copy")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for sampling")
    ap.add_argument("--dry-run", action="store_true", help="Print actions, do not write files")
    return ap.parse_args()

def link_or_copy(src: Path, dst: Path, mode: str):
    if mode == "hard":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)  # cross-filesystem fallback
            return
    if mode == "soft":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)

def row_to_patch_path(root: Path, patch_id: str) -> Path:
    # Many BEN releases: <tile>/<patch>/<patch>_Bxx.tif
    tile_name = "_".join(patch_id.split("_")[:7])
    return root / tile_name / patch_id

def cap_df(df, cap, seed):
    if cap is None or len(df) <= cap:
        return df
    idx = list(df.index)
    random.Random(seed).shuffle(idx)
    return df.loc[idx[:cap]]

def main():
    args = parse_args()
    SRC = Path(args.src).resolve()
    DST = Path(args.dst).resolve()
    META = Path(args.metadata).resolve()

    (DST / "train").mkdir(parents=True, exist_ok=True)
    (DST / "validation").mkdir(parents=True, exist_ok=True)
    (DST / "test").mkdir(parents=True, exist_ok=True)

    print("Loading metadata:", META)
    df = pd.read_parquet(META)
    # Normalize split names
    split = df["split"].astype(str).str.lower()
    df = df.assign(split=split)

    # Collect classes
    print("Extracting unique classes...")
    all_classes = sorted({c for row in df["labels"] for c in (row or [])})
    print(f"Classes: {len(all_classes)}")

    # Select splits
    splits = {
        "train": df[df["split"].isin(["train","training"])],
        "validation": df[df["split"].isin(["validation","val"])],
        "test": df[df["split"].isin(["test"])],
    }

    # Cap sizes
    splits["train"]      = cap_df(splits["train"], args.max_train, args.seed)
    splits["validation"] = cap_df(splits["validation"], args.max_val, args.seed)
    splits["test"]       = cap_df(splits["test"], args.max_test, args.seed)

    print("Planned split sizes:",
          {k: len(v) for k,v in splits.items()})

    header = ["filename"] + all_classes
    label_csvs = {
        "train": DST / "train_labels.csv",
        "validation": DST / "validation_labels.csv",
        "test": DST / "test_labels.csv",
    }
    rows_acc = {"train": [], "validation": [], "test": []}

    for name in ["train","validation","test"]:
        df_split = splits[name]
        out_dir = DST / name
        sample_count = 0
        for row in df_split.itertuples():
            patch_id = row.patch_id
            labels   = row.labels or []

            patch_dir = row_to_patch_path(SRC, patch_id)
            if not patch_dir.exists():
                continue

            # band files
            band_files = {b: patch_dir / f"{patch_id}_{b}.tif" for b in args.bands}
            if not all(p.exists() for p in band_files.values()):
                continue

            sample_count += 1
            sample = f"sample_{sample_count:06d}"
            sample_dir = out_dir / sample
            if not args.dry_run:
                sample_dir.mkdir(parents=True, exist_ok=True)

            for b, srcf in band_files.items():
                dstf = sample_dir / f"{b}.tif"
                if not args.dry_run:
                    link_or_copy(srcf, dstf, args.link)

            label_binary = [1 if cls in labels else 0 for cls in all_classes]
            rows_acc[name].append([sample] + label_binary)

            if sample_count % 1000 == 0:
                print(f"[{name}] {sample_count}")

        # write csv per split
        if not args.dry_run:
            with open(label_csvs[name], "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows_acc[name])

    print("\nDone.")
    for name in ["train","validation","test"]:
        print(f"{name}: {len(rows_acc[name])}")
    print(f"Output in: {DST}")
    if args.dry_run:
        print("(*) Dry run only; no files were written.")
if __name__ == "__main__":
    main()
