#!/bin/bash
#SBATCH --job-name=benet_s2_train
#SBATCH --account=kuin0137

## 🔧 GPU + CPU resources
#SBATCH --partition=gpu          # GPU queue on Almesbar
#SBATCH --gres=gpu:1             # 1 V100 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # for dataloaders / BLAS
#SBATCH --mem=64G
#SBATCH --time=24:00:00          # 12 hours (max on gpu is 72h)

## 🔎 Logs (relative to project root)
#SBATCH --output=jobs/logs/train_benet_%j.out
#SBATCH --error=jobs/logs/train_benet_%j.err

echo "======================================="
echo "   BigEarthNet-S2 LoRA Training"
echo "======================================="
echo "Host       : $(hostname)"
echo "Start time : $(date)"
echo "SLURM_JOBID: ${SLURM_JOB_ID}"
echo

# -----------------------------
# 1) Load Python environment
# -----------------------------
module purge
module load miniconda/3

# Your dedicated project env (adjust if different)
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
echo "Using settings:"
echo "  batch-size : 64"
echo "  epochs     : 15"
echo "  lr         : 1e-4"
echo "  workers    : 8"
echo "  lora-rank  : 8"
echo "  exp-name   : benet_full_e15_bs64"
echo

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
