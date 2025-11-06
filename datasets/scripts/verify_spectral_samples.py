#!/usr/bin/env python3
import os
from pathlib import Path
from datetime import datetime

EXPECTED_BANDS = {"B02.tif", "B03.tif", "B04.tif", "B08.tif", "B11.tif", "B12.tif"}

def check_split(split_dir: Path, log_file):
    total = 0
    complete = 0
    missing = []

    for sample_dir in sorted(split_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        total += 1
        band_files = {f.name for f in sample_dir.glob("*.tif")}
        if band_files == EXPECTED_BANDS:
            complete += 1
        else:
            missing_bands = EXPECTED_BANDS - band_files
            extra_bands = band_files - EXPECTED_BANDS
            missing.append((sample_dir.name, sorted(missing_bands), sorted(extra_bands)))

    with open(log_file, "a") as log:
        log.write(f"\nChecked: {split_dir}\n")
        log.write(f"Total samples:   {total}\n")
        log.write(f"Complete samples:{complete}\n")
        log.write(f"Incomplete:      {total - complete}\n")

        if missing:
            log.write("\n⚠️  Incomplete or incorrect samples:\n")
            for name, miss, extra in missing:
                if miss:
                    log.write(f"  - {name}: missing {', '.join(miss)}\n")
                if extra:
                    log.write(f"    extra   {', '.join(extra)}\n")

        log.write("\n----------------------------------------\n")

    return total, complete, missing

def main():
    project_root = Path(__file__).resolve().parents[1]
    eurosat_root = project_root / "datasets" / "eurosat_ms"
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"verify_{timestamp}.log"

    print(f"📝 Logging verification to: {log_file}")

    splits = []
    train_dir = eurosat_root / "train"
    test_dir  = eurosat_root / "test"

    if train_dir.exists():
        splits.append(("train", train_dir))
    if test_dir.exists():
        splits.append(("test", test_dir))

    results = []
    for name, path in splits:
        total, complete, missing = check_split(path, log_file)
        results.append((name, total, complete, len(missing)))

    print("\n✅ Verification completed.")
    for name, total, complete, bad in results:
        print(f"{name:<5}: {complete}/{total} complete, {bad} incomplete.")
    print(f"\nFull details saved in: {log_file}\n")

if __name__ == "__main__":
    main()
