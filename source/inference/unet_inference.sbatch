#!/bin/bash
#SBATCH -J unet-infer
#SBATCH -N 1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o unet-infer-%j.out
# atl1-1-03-010-30-0 (Oct 19) and atl1-1-03-011-3-0 (Oct 26) have H100s that CUDA cannot open. PACE's submit filter
# overwrites --exclude when no partition is given, so -p is required for it to stick.
#SBATCH -p coe-gpu
#SBATCH --exclude=atl1-1-03-010-30-0,atl1-1-03-011-3-0

cd ~/Code/NUE
module load anaconda3
conda activate nue

# 960x540 -> 4K super-resolution of 20 val frames spread over all clips, scored against the
# originals alongside bicubic. Results go to inference_best/ next to the checkpoint. Safe to
# run while training: train_unet.py replaces best.pt atomically.
RUN=/storage/ice1/4/8/jcochran66/runs/latent_unet
nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv

python source/inference/unet_inference.py \
  --input /storage/ice1/4/8/jcochran66/Inter4K_frames/val \
  --checkpoint "$RUN/best.pt" \
  --limit 20 --save-images 4 \
  --tile 256 --overlap 64 --tile-batch 16 --steps 50 \
  --amp
