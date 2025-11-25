#!/bin/bash
#SBATCH --job-name=benet_clip_visual_lora
#SBATCH --account=kuin0137

## GPU + CPU RESOURCES
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00

#SBATCH --output=jobs/logs/benet_clip_visual_lora/%j.out
#SBATCH --error=jobs/logs/benet_clip_visual_lora/%j.err

echo "======================================="
echo "   BigEarthNet-S2 SkySense-CLIP LoRA"
echo "        FULL TRAINING RUN"
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
mkdir -p jobs/logs/benet_clip_visual_lora
mkdir -p checkpoints/benet_clip_visual_lora

# be nice to dataloaders
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting FULL BigEarthNet-S2 CLIP-LoRA training..."
echo "======================================="
echo

# -----------------------------
# 4) Run training
# -----------------------------
srun python train_scripts/train_bigearthnet_clip_lora.py \
    --data-root datasets/bigearthnet_s2 \
    --batch-size 8 \
    --epochs 15 \
    --lr 1e-4 \
    --num-workers 8 \
    --lora-rank 8 \
    --exp-name benet_clip_visual_lora_e15_bs8 \
    --save-lora-only

echo
echo "======================================="
echo "End time: $(date)"
echo "      Full training finished!"
echo "======================================="
