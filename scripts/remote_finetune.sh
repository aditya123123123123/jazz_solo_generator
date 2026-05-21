#!/usr/bin/env bash
set -euo pipefail

cd /workspace/jazz_solo_generator

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p checkpoints/v6.2.0 outputs/training_logs outputs/eval/conditioning_ablation/v6.2.0_runpod

# Confirm CUDA is visible. train_v6.py auto-selects CUDA when available.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))"

python -m src.training.train_v6 \
    --resume-from checkpoints/v6.1.0_best.pt \
    --lambda-interval 0.0475 \
    --output-dir checkpoints/v6.2.0/ \
    --best-output checkpoints/v6.2.0_best.pt \
    --latest-output checkpoints/v6.2.0/v6.2.0_latest.pt \
    --epochs 8 \
    --lr 1e-4 \
    --cache data/processed/notes_v6_cache.pt \
    --run-name note-executor-v6.2.0-runpod-4090 \
    2>&1 | tee outputs/training_logs/v6.2.0_finetune_runpod.log

python - <<'PY'
import torch
ckpt = torch.load('checkpoints/v6.2.0_best.pt', map_location='cpu', weights_only=False)
state = ckpt.get('model_state', ckpt)
w = state['phrase_pos_embed.weight']
print('phrase_pos_embed norm:', w.norm().item())
print('phrase_pos_embed min:', w.min().item())
print('phrase_pos_embed max:', w.max().item())
assert w.norm().item() > 0, 'phrase_pos_embed did not move from zero'
PY

python -m src.eval.conditioning_ablation \
    --checkpoint checkpoints/v6.2.0_best.pt \
    --baseline-checkpoint checkpoints/v6.1.0_best.pt \
    --cache data/processed/notes_v6_cache.pt \
    --subset-size 100000 \
    --batch-size 512 \
    --seed 42 \
    --out-dir outputs/eval/conditioning_ablation/v6.2.0_runpod \
    | tee outputs/v6.2.0_ablation.txt

if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 checkpoints/v6.2.0_best.pt
else
    sha256sum checkpoints/v6.2.0_best.pt
fi
