#!/bin/bash
#SBATCH --job-name=benet_s2_train
#SBATCH --account=kuin0137

## 🔧 GPU + CPU resources (adjust partition/gpu type if needed)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

#SBATCH --output=jobs/logs/train_benet_%j.out
#SBATCH --error=jobs/logs/train_benet_%j.err

echo "======================================="
echo "   BigEarthNet-S2 LoRA Training"
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

# -----------------------------
# 1) Load Python environment
# -----------------------------
module purge
module load miniconda/3

# Your dedicated project env
PY=/dpc/kuin0137/envs/fusionlora/bin/python

# -----------------------------
# 2) Move to project directory
# -----------------------------
cd /dpc/kuin0137/skysense_lora/fusion-lora-skysense

echo "Working directory: $(pwd)"
echo

# Make sure log & checkpoint dirs exist
mkdir -p jobs/logs
mkdir -p checkpoints

# Optional: be nice to dataloaders
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -----------------------------
# 3) Run training
# -----------------------------
echo "Starting BigEarthNet-S2 training..."

$PY train_scripts/train_bigearthnet_cls.py \
    --data-root datasets/bigearthnet_s2 \
    --batch-size 64 \
    --epochs 15 \
    --lr 1e-4 \
    --num-workers 8 \
    --lora-rank 8 \
    --exp-name benet_full_e15_bs64 \
    --save-lora-only

echo
echo "Finish time: $(date)"
echo "======================================="
