#!/bin/bash
#SBATCH --job-name=prep_benet_s2
#SBATCH --account=kuin0137
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # use many CPUs for rasterio & tif loading
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/prep_benet_%j.out
#SBATCH --error=jobs/logs/prep_benet_%j.err

echo "======================================="
echo "   BigEarthNet-S2 Spectral Preparation"
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

# -----------------------------
# 1) Load Python environment
# -----------------------------
module purge
module load miniconda/3

# OPTIONAL: If you have a conda env for this project:
# conda activate fusionlora

# -----------------------------
# 2) Move to project directory
# -----------------------------
cd /dpc/kuin0137/code/fusion-lora-skysense

echo "Working directory: $(pwd)"
echo

# Create logs folder if not exists
mkdir -p jobs/logs

# -----------------------------
# 3) Run preprocessing
# -----------------------------
echo "Running spectral tensor preparation..."

python datasets/datasets/utils/prepare_spectral_data.py --size 256 --overwrite

echo
echo "Computing dataset statistics..."

python datasets/utils/compute_stats.py

echo
echo "Finish time: $(date)"
echo "======================================="
