#!/bin/bash
#SBATCH -J unet
#SBATCH -N 1
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# coe-ice QOS caps a job at 960 GPU-minutes (GPUs x minutes <= 960)
#SBATCH -t 4:00:00
#SBATCH -o unet-%j.out
# atl1-1-03-010-30-0 (Oct 19) and atl1-1-03-011-3-0 (Oct 26) have H100s that CUDA cannot open. PACE's submit filter
# overwrites --exclude when no partition is given, so -p is required for it to stick.
#SBATCH -p coe-gpu
#SBATCH --exclude=atl1-1-03-010-30-0,atl1-1-03-011-3-0

cd ~/Code/NUE
module load anaconda3
conda activate nue

# Resubmit this script to continue: it resumes from latest.pt and stops after --epochs total.
# A resume is refused if the VAE checkpoint below has been replaced since the run started.
RUNS=/storage/ice1/4/8/jcochran66/runs
OUT=$RUNS/latent_unet
EPOCHS=40
# Jobs still left in a submitted chain once every epoch is done stop here.
if [ -f "$OUT/latest.pt" ] && python -c "
import sys, torch
sys.exit(torch.load(sys.argv[1], map_location='cpu', weights_only=True, mmap=True)['epoch'] < int(sys.argv[2]))
" "$OUT/latest.pt" "$EPOCHS"; then
  echo "$OUT/latest.pt has finished all $EPOCHS epochs; nothing to do"
  exit 0
fi
RESUME=()
[ -f "$OUT/latest.pt" ] && RESUME=(--resume "$OUT/latest.pt")
NGPU=$(python -c 'import torch; print(torch.cuda.device_count())')
nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv

torchrun --standalone --nproc_per_node="$NGPU" source/training/train_unet.py \
  --train-dir /storage/ice1/4/8/jcochran66/Inter4K_frames/train \
  --val-dir /storage/ice1/4/8/jcochran66/Inter4K_frames/val \
  --vae "$RUNS/mbconv_vae/best.pt" \
  --out-dir "$OUT" --epochs "$EPOCHS" \
  --workers 8 --amp "${RESUME[@]}"
