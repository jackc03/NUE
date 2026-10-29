
#!/usr/bin/env bash
set -euo pipefail

# ImageNet Path
# /storage/ice1/shared/d-pace_community/makerspace-datasets/IMAGE/Imagenet2012
# /home/hice1/yyu496/scratch/NUE/inter4k_frames

# Use system GCC instead of Conda GCC
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Locate train_VAE.py relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_ARGS=(
    --train-dir "/home/hice1/yyu496/scratch/NUE/inter4k_frames_fps_60/train"
    --val-dir "/home/hice1/yyu496/scratch/NUE/inter4k_frames_fps_60/val"
    --out-dir "/home/hice1/yyu496/scratch/NUE/runs5_inter4k_encoder_3_decoder_4_test1"

    --workers 8
    --compile
    --amp
    --lr 1e-3
    --save_every_steps 200

    --imagenet

    --max_channels 512
    --latent-channels 16
    --blocks-per-stage 2

    --epochs 80
    --kl-warmup-steps 200
    --batch-size 32

    --train

    # --resume "/home/hice1/yyu496/scratch/NUE/runs1/step_level_latest.pt"
    # --resume "/home/hice1/yyu496/scratch/NUE/runs4_inter4k_small_bs_32_2_512/step_level_latest.pt"
)

torchrun --standalone --nproc-per-node=3 \
    "$SCRIPT_DIR/train_VAE.py" \
    "${TRAIN_ARGS[@]}"