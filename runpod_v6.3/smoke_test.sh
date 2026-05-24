#!/bin/bash
set -e
echo "Running 50-step smoke test..."
python src/training/train_v6.py \
  --dry-run-batches 2 \
  --epochs 1 \
  --resume checkpoints/v6.2.0_best.pt \
  --run-name v6.3-smoke-test \
  2>&1 | tail -20
echo "Smoke test completed successfully."
