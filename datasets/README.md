# BigEarthNet-S2 Dataset Preparation

This directory contains scripts and instructions for preparing the BigEarthNet-S2

## Overview

The BigEarthNet v2.0 dataset consists of 549,488 Sentinel-2 image patches with multi-label land cover annotations. This preparation pipeline:
- Extracts 6 essential spectral bands (B02, B03, B04, B08, B11, B12)
- Organizes data into train/validation/test splits
- Generates binary one-hot encoded labels for multi-label classification

## Prerequisites

- Python 3.8+
- ~100 GB free disk space

## Download Dataset

### 1. Download BigEarthNet-S2 Archive

Visit [BigEarthNet Downloads](https://bigearth.net/#downloads) and download:

```bash
# Download BigEarthNet-S2.tar.zst 
https://bigearth.net/downloads/BigEarthNet-S2.tar.zst
```

### 2. Download Metadata

Download the metadata file from the same page:

```bash
# Download metadata.parquet 
https://bigearth.net/downloads/metadata.parquet
```

### 3. Extract Archive

Decompress the BigEarthNet-S2 archive using zstd:

```bash
# Install zstd if not available
sudo apt install zstd

# Extract the archive
tar --use-compress-program=unzstd -xvf BigEarthNet-S2.tar.zst
```

## Directory Structure

After downloading and extracting, your directory should look like this:

```
datasets/
├── BigEarthNet-S2/     
│   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP/
│   │   ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57/
│   │   │   ├── *_B01.tif
│   │   │   ├── *_B02.tif
│   │   │   ├── ...
│   │   │   └── *_B12.tif
│   │   └── ...
│   └── ...
├── BigEarthNet-S2.tar.zst               
├── metadata.parquet                     
├── scripts/
│   └── prepare_bigearthnet_s2.py       
├── requirements.txt                     
└── README.md                            
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Run Preparation Script

```bash
cd scripts
python prepare_bigearthnet_s2.py
```

### What the Script Does

1. **Loads metadata**: Reads patch IDs, labels, and split assignments
2. **Extracts unique classes**: Identifies all 19 land cover classes
3. **Processes each split**: 
   - Train: ~237,871 patches
   - Validation: ~122,342 patches
   - Test: ~119,825 patches
4. **Copies selected bands**: Only B02, B03, B04, B08, B11, B12 (RGB, NIR, SWIR)
5. **Generates CSV labels**: Binary one-hot encoded format

### Expected Output

After running the script successfully:

```
datasets/
└── datasets/
    └── bigearthnet_s2/
        ├── train/
        │   ├── sample_000001/
        │   │   ├── B02.tif    # Blue
        │   │   ├── B03.tif    # Green
        │   │   ├── B04.tif    # Red
        │   │   ├── B08.tif    # NIR
        │   │   ├── B11.tif    # SWIR1
        │   │   └── B12.tif    # SWIR2
        │   ├── sample_000002/
        │   └── ...
        ├── validation/
        │   └── ...
        ├── test/
        │   └── ...
        ├── train_labels.csv
        ├── validation_labels.csv
        └── test_labels.csv
```

## CSV Label Format

The CSV files use binary one-hot encoding for multi-label classification:

```csv
filename,Agro-forestry areas,Arable land,Beaches dunes sands,Broad-leaved forest,...
sample_000001,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0
sample_000002,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
```

- **Columns**: filename + 19 land cover classes
- **Values**: 1 = class present, 0 = class absent
- **Compatible with**: PyTorch BCEWithLogitsLoss, MultiLabelSoftMarginLoss

## Land Cover Classes (19 Classes)

Based on CORINE Land Cover (CLC) 2018:

1. Agro-forestry areas
2. Arable land
3. Beaches, dunes, sands
4. Broad-leaved forest
5. Coastal wetlands
6. Coniferous forest
7. Complex cultivation patterns
8. Industrial or commercial units
9. Inland waters
10. Inland wetlands
11. Land principally occupied by agriculture, with significant areas of natural vegetation
12. Marine waters
13. Mixed forest
14. Moors, heathland and sclerophyllous vegetation
15. Natural grassland and sparsely vegetated areas
16. Pastures
17. Permanent crops
18. Transitional woodland, shrub
19. Urban fabric

## Spectral Bands

| Band | Name  | Wavelength | Resolution | Purpose |
|------|-------|------------|------------|---------|
| B02  | Blue  | 490 nm     | 10m        | Visible |
| B03  | Green | 560 nm     | 10m        | Visible |
| B04  | Red   | 665 nm     | 10m        | Visible |
| B08  | NIR   | 842 nm     | 10m        | Vegetation analysis |
| B11  | SWIR1 | 1610 nm    | 20m        | Moisture content |
| B12  | SWIR2 | 2190 nm    | 20m        | Geology |

