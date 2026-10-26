#!/bin/bash
#SBATCH -J vae
#SBATCH -N 1
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# coe-ice QOS caps a job at 960 GPU-minutes (GPUs x minutes <= 960)
#SBATCH -t 4:00:00
#SBATCH -o vae-%j.out
# atl1-1-03-010-30-0 (Oct 19) and atl1-1-03-011-3-0 (Oct 26) have H100s that CUDA cannot open. PACE's submit filter
# overwrites --exclude when no partition is given, so -p is required for it to stick.
#SBATCH -p coe-gpu
#SBATCH --exclude=atl1-1-03-010-30-0,atl1-1-03-011-3-0

cd ~/Code/NUE
# activate your PyTorch environment here
module load anaconda3
conda activate nue

# Resubmit this script to continue: it resumes from latest.pt and stops after --epochs total.
OUT=/storage/ice1/4/8/jcochran66/runs/mbconv_vae
RESUME=()
[ -f "$OUT/latest.pt" ] && RESUME=(--resume "$OUT/latest.pt")
NGPU=$(python -c 'import torch; print(torch.cuda.device_count())')
nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv

torchrun --standalone --nproc_per_node="$NGPU" source/training/train_VAE.py \
  --train-dir /storage/ice1/4/8/jcochran66/Inter4K_frames/train \
  --val-dir /storage/ice1/4/8/jcochran66/Inter4K_frames/val \
  --out-dir "$OUT" \
  --workers 8 --amp "${RESUME[@]}"
