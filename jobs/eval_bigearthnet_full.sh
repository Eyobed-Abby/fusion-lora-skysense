#!/bin/bash
#SBATCH --job-name=benet_s2_eval
#SBATCH --account=kuin0137

## 🔧 GPU + CPU resources
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

#SBATCH --output=jobs/logs/eval_benet_%j.out
#SBATCH --error=jobs/logs/eval_benet_%j.err

echo "======================================="
echo "   BigEarthNet-S2 Evaluation (LoRA+Fusion)"
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

# -----------------------------
# 1) Load Python (same as login)
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

# Make sure log & results dirs exist
mkdir -p jobs/logs
mkdir -p results

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CKPT="checkpoints/benet_full_e15_bs8/best_epoch_15_lora_only.pth"
EXP_NAME="benet_full_e15_bs8_full"

echo "Checkpoint : $CKPT"
echo "Exp name   : $EXP_NAME"
echo

# -----------------------------
# 3a) Full VAL evaluation
# -----------------------------
echo ">>> Running FULL validation evaluation..."
srun python train_scripts/eval_bigearthnet.py \
    --split val \
    --batch-size 64 \
    --ckpt "$CKPT" \
    --exp-name "$EXP_NAME"

echo
echo "Validation eval done."
echo

# -----------------------------
# 3b) Full TEST evaluation
# -----------------------------
echo ">>> Running FULL test evaluation..."
srun python train_scripts/eval_bigearthnet.py \
    --split test \
    --batch-size 64 \
    --ckpt "$CKPT" \
    --exp-name "$EXP_NAME"

echo
echo "Test eval done."
echo

echo "Finish time: $(date)"
echo "======================================="
