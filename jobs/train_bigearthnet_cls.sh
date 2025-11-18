#!/bin/bash
#SBATCH --job-name=benet_s2_train
#SBATCH --account=kuin0137
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dpc/kuin0137/skysense_lora/fusion-lora-skysense/jobs/logs/train_benet_%j.out
#SBATCH --error=/dpc/kuin0137/skysense_lora/fusion-lora-skysense/jobs/logs/train_benet_%j.err

echo "======================================="
echo "   BigEarthNet-S2 LoRA Training"
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

module purge
module load miniconda/3

PY=/dpc/kuin0137/envs/fusionlora/bin/python

cd /dpc/kuin0137/skysense_lora/fusion-lora-skysense
echo "Working directory: $(pwd)"
echo

mkdir -p checkpoints/full_run

$PY train_scripts/train_bigearthnet_cls.py \
    --data-root datasets/bigearthnet_s2 \
    --batch-size 64 \
    --epochs 15 \
    --lr 5e-5 \
    --num-workers 8 \
    --lora-rank 8 \
    --save-dir checkpoints/full_run \
    --exp-name full_run

echo
echo "Finish time: $(date)"
echo "======================================="
