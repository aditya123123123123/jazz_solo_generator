# Conditioning Ablation — 2026-05-18

## Headline
NoteExecutor v6_best.pt is functionally ignoring its conditioning inputs (phrase_id, chord_ids, artist_id). The trained model solves the loss using only the local 8-note autoregressive context (ctx_pitch, ctx_dur, ctx_rest). The two-stage planner-executor architecture's central design claim — that the Planner shapes the Executor's output via phrase tokens — is not borne out by the v6 trained weights.

## Evidence — 100,000-window validation ablation

SEED=42 random_split val portion, v6_best.pt, batch 512.

| condition       | pitch_CE | dur_CE | rest_CE | total  | Δ pitch vs baseline |
|-----------------|----------|--------|---------|--------|---------------------|
| baseline        | 5.0017   | 2.8639 | 0.3868  | 8.2524 | —                   |
| phrase_zeroed   | 4.9698   | 2.8483 | 0.3816  | 8.1997 | -0.0319             |
| chord_zeroed    | 4.9375   | 2.8130 | 0.3701  | 8.1206 | -0.0642             |
| artist_zeroed   | 5.0021   | 2.8728 | 0.3762  | 8.2511 | +0.0004             |
| chord_shuffled  | 5.0014   | 2.8727 | 0.3851  | 8.2592 | -0.0003             |
| phrase_shuffled | 5.0022   | 2.8613 | 0.3885  | 8.2520 | +0.0005             |

Interpretation:
- Shuffle conditions (wrong chord/phrase) are statistically indistinguishable from baseline. The model treats correct and wrong conditioning identically.
- Zeroed conditions REDUCE pitch CE — the chord and phrase pathways are actively introducing noise the model has to filter out.
- artist_id is decorative (Δ within numerical noise).
- A model genuinely using conditioning would show shuffled > zeroed > baseline in CE. We see the opposite ordering, confirming collapse rather than under-use.

8,192-window pilot (run earlier): same direction, similar magnitudes (chord_zeroed Δ = -0.060, phrase_zeroed Δ = -0.029). Pattern reproduces and tightens at 100k.

## Wiring Audit
- chord_ids vocab utilization on subset: 146/196 unique
- phrase_id vocab utilization on subset: 64/67 unique
- artist_id vocab utilization on subset: 13/16 unique
- chord_mask alignment with chord_ids != 0: 100% (zero violations)
- chord_ids %zero on subset: 85.8% (within-batch padding to max_chord, expected)
- Embedding shapes (checkpoint vs instantiated model):
  - chord_embed.weight: (196, 256) == (196, 256) ✓
  - phrase_embed.weight: (67, 256) == (67, 256) ✓
  - artist_embed.weight: (16, 256) == (16, 256) ✓
  - pitch_embed.weight: (132, 256) == (132, 256) ✓
  - dur_embed.weight: (18, 256) == (18, 256) ✓
- load_state_dict(state, strict=False): 0 missing, 0 unexpected

Wiring is clean. The collapse is not a data path bug or a checkpoint mismatch.

## Root Cause — v5b → v6 archaeology

Commit 33971ee (2026-05-10): "Step 7: train_v6.py — full v6 training script with tempo conditioning, AMP, wandb, early stopping". Stat: +338 / -0. Created src/training/train_v6.py from a blank file rather than refactoring src/training/train_v5b.py.

In doing so, two elements of v5b's loss design were silently dropped without mention in the commit message:

1. The harmonic_penalty auxiliary loss (HARMONIC_WEIGHT = 0.15), applied at strong-beat positions {0, 4, 8, 12} (src/training/train_v5b.py:24, :33, :81-99, :113-134).
2. Label smoothing reduced from 0.1 to 0.0 (train_v5b.py vs train_v6.py:27).

v5b's harmonic_penalty (paraphrased):
- Computes -log P(chord_tone) over softmax(p_logits) at strong-beat positions
- Indexed against the first chord token of each window
- Added as auxiliary term: total_loss = pitch_CE + dur_CE + rest_CE + 0.15 * harmonic_penalty

v6's loss (src/training/train_v6.py:121-124):
    pitch_l = ce(p_logits, batch["target_pitch"])
    dur_l   = ce(d_logits, batch["target_dur"])
    rest_l  = ce(r_logits, batch["target_rest"])
    total   = pitch_l + dur_l + rest_l

No harmonic term. No chord_tone_tensor reference. Three uniform CEs, summed.

The harmonic auxiliary was the explicit pressure pushing the model to use chord_ids for pitch selection at the moments harmony matters most (downbeats). Without it, the v6 training loop has no signal that distinguishes "right chord" from "wrong chord" with respect to the loss the model actually optimizes — chord conditioning collapsed to noise.

train_v5b.py is still on disk and importable. The harmonic_penalty function is intact in v5b. v6_best.pt was trained from train_v6.py, the loss-stripped version.

## Recommended Fix — v6.1

Port v5b's harmonic_penalty into train_v6.py with HARMONIC_WEIGHT = 0.15. Finetune from v6_best.pt for 5-10 epochs (~30-60 min on existing hardware). Re-run this exact same conditioning ablation. Pre-committed success criteria:

- Strong success: Δ chord_shuffled ≥ +0.10 pitch CE (model meaningfully penalizes wrong chord input)
- Partial success: Δ chord_shuffled in [+0.05, +0.10] (model uses chord but weakly)
- No effect: Δ chord_shuffled < +0.05 (harmonic loss alone insufficient; deeper intervention needed)

## Open Implementation Questions for v6.1

1. Does v6's pos_in_phrase preserve v5b's strong-beat indexing semantics (positions {0, 4, 8, 12} as downbeats)? Verify before porting strong-beat constants.
2. Does v5b's "first chord token of each window" indexing apply directly to v6's chord_ids representation, given v6 uses (B, Lc) variable-length chord sequences with mask?
3. Restore LABEL_SMOOTHING = 0.1 simultaneously? Default recommendation: keep at 0.0 for first attempt to isolate harmonic loss effect, restore in a follow-up if needed.
4. Finetune from v6_best.pt or full retrain? Default recommendation: finetune first (fast, decisive). Full retrain only if finetune shows partial-or-better success and you want a cleaner final checkpoint.
5. Does v5b have its own conditioning-ablation result available for comparison? If v5b_best.pt exists, run the same ablation on it as a control — confirms whether the harmonic loss was sufficient in v5b.

## Status

Diagnostic complete. No source files modified. Sampling defaults committed in 052040e remain in effect. v6.1 fix to be implemented in next session, ideally in a separate tool (Codex) per the project's tool-switching plan.
