import os
import shutil
import csv
from pathlib import Path
import pandas as pd

script_dir = Path(__file__).parent
src_root = script_dir.parent / "BigEarthNet-S2"
dst_root = script_dir.parent / "datasets" / "bigearthnet_s2"
os.makedirs(dst_root, exist_ok=True)

metadata_path = script_dir.parent / "metadata.parquet"

train_dir = dst_root / "train"
val_dir = dst_root / "validation"
test_dir = dst_root / "test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

required_bands = ["B02", "B03", "B04", "B08", "B11", "B12"]

print("Loading metadata...")
df = pd.read_parquet(metadata_path)
print(f"Loaded {len(df)} patches from metadata")
print(f"Split distribution: train={len(df[df['split']=='train'])}, validation={len(df[df['split']=='validation'])}, test={len(df[df['split']=='test'])}")

print("\nExtracting all unique classes...")
all_classes = set()
for labels in df['labels']:
    all_classes.update(labels)
all_classes = sorted(all_classes)
print(f"Found {len(all_classes)} unique classes: {all_classes[:5]}...")

train_rows, val_rows, test_rows = [], [], []

for split_name, split_df, target_dir, rows_list in [
    ("train", df[df['split'] == 'train'], train_dir, train_rows),
    ("validation", df[df['split'] == 'validation'], val_dir, val_rows),
    ("test", df[df['split'] == 'test'], test_dir, test_rows)
]:
    print(f"\nProcessing {split_name} split ({len(split_df)} patches)...")
    
    sample_count = 0
    for row in split_df.itertuples():
        patch_id = row.patch_id
        labels = row.labels
        
        tile_name = "_".join(patch_id.split("_")[:7])
        patch_dir = src_root / tile_name / patch_id
        
        if not patch_dir.exists():
            continue
        
        band_files = {band: patch_dir / f"{patch_id}_{band}.tif" for band in required_bands}
        if not all(f.exists() for f in band_files.values()):
            continue
        
        sample_count += 1
        sample_name = f"sample_{sample_count:06d}"
        sample_dir = target_dir / sample_name
        os.makedirs(sample_dir, exist_ok=True)
        
        for band, src_file in band_files.items():
            dst_file = sample_dir / f"{band}.tif"
            shutil.copy2(src_file, dst_file)
        
        label_binary = [1 if cls in labels else 0 for cls in all_classes]
        rows_list.append([sample_name] + label_binary)
        
        if sample_count % 1000 == 0:
            print(f"  Processed {sample_count} samples")

header = ["filename"] + all_classes

with open(dst_root / "train_labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(train_rows)

with open(dst_root / "validation_labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(val_rows)

with open(dst_root / "test_labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(test_rows)

print("\n✅ Dataset preparation complete!")
print(f"Train samples: {len(train_rows)}")
print(f"Validation samples: {len(val_rows)}")
print(f"Test samples: {len(test_rows)}")
print(f"\nNumber of classes: {len(all_classes)}")
print(f"CSV files saved in: {dst_root}")
print("Format: Binary one-hot encoding (1=class present, 0=class absent)")
