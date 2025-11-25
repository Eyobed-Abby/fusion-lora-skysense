#!/bin/bash
#SBATCH --job-name=sanity_clip_lora
#SBATCH --account=kuin0137

## GPU + CPU RESOURCES
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

#SBATCH --output=jobs/logs/sanity_clip_lora/%j.out
#SBATCH --error=jobs/logs/sanity_clip_lora/%j.err


echo "======================================="
echo "     SkySense-CLIP LoRA Sanity Check"
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

# -----------------------------
# 1) Load Python
# -----------------------------
module purge
module load miniconda/3

echo "Python path: $(which python)"
python --version
echo

# -----------------------------
# 2) Move to project directory
# -----------------------------
cd /dpc/kuin0137/skysense_lora/fusion-lora-skysense

echo "Working directory: $(pwd)"
echo

# -----------------------------
# 3) Create output folders
# -----------------------------
mkdir -p jobs/logs/sanity_clip_lora
mkdir -p checkpoints/sanity_clip_lora

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Beginning 1-epoch sanity run..."
echo "======================================="
echo

# -----------------------------
# 4) Run sanity training
# -----------------------------
srun python train_scripts/train_bigearthnet_clip_lora.py \
    --data-root datasets/bigearthnet_s2 \
    --batch-size 4 \
    --epochs 1 \
    --lr 1e-4 \
    --num-workers 8 \
    --lora-rank 8 \
    --debug-small 32 \
    --exp-name sanity_clip_lora \
    --save-lora-only

echo
echo "======================================="
echo "End time: $(date)"
echo "      Sanity test finished!"
echo "======================================="
