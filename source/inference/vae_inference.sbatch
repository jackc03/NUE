#!/bin/bash
#SBATCH -J vae-infer
#SBATCH -N 1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o vae-infer-%j.out
# atl1-1-03-010-30-0 (Oct 19) and atl1-1-03-011-3-0 (Oct 26) have H100s that CUDA cannot open. PACE's submit filter
# overwrites --exclude when no partition is given, so -p is required for it to stick.
#SBATCH -p coe-gpu
#SBATCH --exclude=atl1-1-03-010-30-0,atl1-1-03-011-3-0

cd ~/Code/NUE
module load anaconda3
conda activate nue

# Tiled full-frame reconstruction metrics on 200 val frames spread over all clips.
# Results go to inference_best/ next to the checkpoint.
RUN=/storage/ice1/4/8/jcochran66/runs/mbconv_vae
nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv

python source/inference/vae_inference.py \
  --input /storage/ice1/4/8/jcochran66/Inter4K_frames/val \
  --checkpoint "$RUN/best.pt" \
  --limit 200 --save-images 4 \
  --tile 256 --overlap 64 --tile-batch 4 \
  --amp
