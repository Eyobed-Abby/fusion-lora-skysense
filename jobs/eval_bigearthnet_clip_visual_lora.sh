#!/bin/bash
#SBATCH --job-name=benet_clip_eval
#SBATCH --account=kuin0137

## 🔧 GPU + CPU resources
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

#SBATCH --output=jobs/logs/eval_benet_clip_%j.out
#SBATCH --error=jobs/logs/eval_benet_clip_%j.err

echo "======================================="
echo "   BigEarthNet-S2 Evaluation (CLIP+LoRA)"
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

# 1) Load Python
module purge
module load miniconda/3

echo "Python path: $(which python)"
python --version
echo

# 2) Move to project directory
cd /dpc/kuin0137/skysense_lora/fusion-lora-skysense

echo "Working directory: $(pwd)"
echo

mkdir -p jobs/logs
mkdir -p results

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CKPT="checkpoints/benet_clip_visual_lora_e15_bs8/last.pth"
EXP_NAME="benet_clip_visual_lora_e15_bs8"

echo "Checkpoint : $CKPT"
echo "Exp name   : $EXP_NAME"
echo

# 3a) FULL VAL evaluation
echo ">>> Running FULL validation evaluation..."
srun python train_scripts/eval_bigearthnet_clip_lora.py \
    --split val \
    --batch-size 8 \
    --ckpt "$CKPT" \
    --exp-name "$EXP_NAME"

echo
echo "Validation eval done."
echo

# 3b) FULL TEST evaluation
echo ">>> Running FULL test evaluation..."
srun python train_scripts/eval_bigearthnet_clip_lora.py \
    --split test \
    --batch-size 8 \
    --ckpt "$CKPT" \
    --exp-name "$EXP_NAME"

echo
echo "Test eval done."
echo

echo "Finish time: $(date)"
echo "======================================="
