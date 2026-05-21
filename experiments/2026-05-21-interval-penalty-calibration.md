# Interval Penalty Calibration for v6.2.0

Date: 2026-05-21

## Goal

Calibrate `lambda_interval` before the v6.2.0 finetune so the interval
regularizer contributes roughly 5-15% of the start-of-finetune loss on real
WJazzD batches, rather than relying on the intentionally extreme synthetic
Phase 4 smoke batch.

## Setup

- Checkpoint: `checkpoints/v6.1.0_best.pt`
- Cache: `data/processed/notes_v6_cache.pt`
- Model: `NoteExecutor` with zero-initialized legacy `phrase_pos_embed.weight`
- Device: MPS
- Split and dataloader: same seed and first 10 shuffled training batches as `src/training/train_v6.py`
- Batch size: 512
- Base loss: pitch CE + duration CE + rest CE
- Raw interval penalty: `lambda_interval = 1.0`

The interval term uses differentiable expected MIDI pitch from the predicted
pitch distribution, appended to the context window. This gives the regularizer
a gradient; argmax would only measure behavior and would not train the model.

## Batch-Level Calibration

The lambda columns solve:

`ratio = lambda * raw_interval_penalty / (ce_loss + lambda * raw_interval_penalty)`

| batch | ce_loss | raw interval penalty | lambda @ 5% | lambda @ 10% | lambda @ 15% |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.0672 | 8.3655 | 0.0256 | 0.0540 | 0.0858 |
| 2 | 3.8881 | 7.7252 | 0.0265 | 0.0559 | 0.0888 |
| 3 | 3.8492 | 13.8723 | 0.0146 | 0.0308 | 0.0490 |
| 4 | 3.9286 | 7.9857 | 0.0259 | 0.0547 | 0.0868 |
| 5 | 3.9345 | 8.5433 | 0.0242 | 0.0512 | 0.0813 |
| 6 | 3.9140 | 11.6545 | 0.0177 | 0.0373 | 0.0593 |
| 7 | 3.9936 | 5.0977 | 0.0412 | 0.0870 | 0.1382 |
| 8 | 3.9786 | 10.2983 | 0.0203 | 0.0429 | 0.0682 |
| 9 | 3.8906 | 10.1231 | 0.0202 | 0.0427 | 0.0678 |
| 10 | 3.9157 | 8.6144 | 0.0239 | 0.0505 | 0.0802 |

## Summary

| metric | value |
|---|---:|
| mean CE loss | 3.9360 |
| mean raw interval penalty | 9.2280 |
| lambda for 5% contribution | 0.0224 |
| lambda for 10% contribution | 0.0474 |
| lambda for 15% contribution | 0.0753 |

## Recommendation

Use:

`lambda_interval = 0.0475`

At the 10-batch mean, this yields:

`0.0475 * 9.2280 / (3.9360 + 0.0475 * 9.2280) = 10.02%`

This keeps the interval regularizer audible enough to matter during the v6.2.0
finetune while avoiding the over-weighted synthetic Phase 4 setting.
