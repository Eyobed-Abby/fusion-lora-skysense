import os, shutil, random, csv
from pathlib import Path

# Paths
root = Path(" /path/to/your/dataset")  # Change this to your dataset root
src_root = root / "EuroSAT_MS"
dst_root = root / "datasets/eurosat_ms"

train_dir = dst_root / "train"
test_dir = dst_root / "test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split ratio (80% train, 20% test)
split_ratio = 0.8
random_seed = 42
random.seed(random_seed)

# Prepare CSV manifests
train_csv = dst_root / "train_labels.csv"
test_csv  = dst_root / "test_labels.csv"

train_rows, test_rows = [], []

# Collect all files across class folders
for class_name in sorted(os.listdir(src_root)):
    class_folder = src_root / class_name
    if not class_folder.is_dir():
        continue
    files = list(class_folder.glob("*.tif"))
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * split_ratio)
    train_files = files[:n_train]
    test_files = files[n_train:]

    print(f"{class_name}: {n_train} train, {len(test_files)} test")

    # Copy to train/test folders with unified names
    for i, f in enumerate(train_files):
        new_name = f"sample_{len(train_rows)+1:05d}.tif"
        dst = train_dir / new_name
        shutil.copy2(f, dst)
        train_rows.append([new_name, class_name])

    for i, f in enumerate(test_files):
        new_name = f"sample_{len(test_rows)+1:05d}.tif"
        dst = test_dir / new_name
        shutil.copy2(f, dst)
        test_rows.append([new_name, class_name])

# Write CSVs
with open(train_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(train_rows)

with open(test_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(test_rows)

print("✅ Split complete!")
print(f"Train samples: {len(train_rows)}, Test samples: {len(test_rows)}")
print(f"Labels saved to: {train_csv} and {test_csv}")