#!/bin/bash
set -euo pipefail

echo "=== v6.3 Training Launch ==="
echo "Commit: a8113c5"
echo "HARMONIC_WEIGHT: 0.15 (masked unknown chords)"

python src/training/train_v6.py \
  --epochs 40 \
  --batch-size 512 \
  --lr 5e-4 \
  --resume checkpoints/v6.2.0_best.pt \
  --run-name note-executor-v6.3.0-runpod \
  --best-output checkpoints/v6.3.0_best.pt \
  --latest-output checkpoints/v6.3.0_latest.pt \
  --cache data/processed/notes_v6_cache.pt \
  --harmonic-weight 0.15 \
  --lambda-interval-min 0.01 \
  --lambda-interval-max 0.10

echo "Training completed."
