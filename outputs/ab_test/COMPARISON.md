# v6.1.0 vs v6.2.0 A/B MIDI Comparison
W&B finetune run: https://wandb.ai/adityajhalani43-trinity-school/jazz-solo-generator/runs/au7denbv
## Generation Setup
- Generator: `src.generation.generate_solo.generate_solo` via existing module functions.
- Rhythm section: `src.generation.rhythm_section.generate_rhythm_section`, style `swing`, seed `42`.
- Artist: `Charlie Parker`
- Progression: `autumn_leaves`
- Tempo: `120.0 BPM`
- Duration temperature: `1.6`
- Rest boost: `1.8`
- Final cadence boost: `3.0`
- Seeds and temperatures: `(0.9, 42)`, `(1.1, 1337)`

Chord progression used:

| # | chord | beats |
|---:|---|---:|
| 1 | Cm7 | 4 |
| 2 | F7 | 4 |
| 3 | Bbj7 | 4 |
| 4 | Ebj7 | 4 |
| 5 | Am7b5 | 4 |
| 6 | D7 | 4 |
| 7 | Gm7 | 8 |
| 8 | Gm7 | 8 |

## Equivalent Invocation

The public CLI does not expose a seed or single-progression selector, so generation was run by importing the existing entry-point functions and setting `torch.manual_seed(seed)` before each call. Rhythm-backed files reuse the JSON note events and merge in a deterministic swing rhythm section.

```bash
python - <<'PY'
from src.generation.generate_solo import PROGRESSIONS, generate_solo, load_models, export_midi
from src.generation.rhythm_section import generate_rhythm_section
# for checkpoint in checkpoints/v6.1.0_best.pt checkpoints/v6.2.0_best.pt
# for (temperature, seed) in [(0.9, 42), (1.1, 1337)]
# torch.manual_seed(seed); generate_solo(PROGRESSIONS["autumn_leaves"], temperature=temperature, ...)
# rhythm = generate_rhythm_section(PROGRESSIONS["autumn_leaves"], tempo_bpm=120.0, style="swing", seed=42)
PY
```

## Files And Metrics

| checkpoint | temp | seed | dry MIDI | with-rhythm MIDI | notes | IOI mean | IOI std | pitch range | first 16 pitches |
|---|---:|---:|---|---|---:|---:|---:|---|---|
| v6.1.0 | 0.9 | 42 | `outputs/ab_test/v6.1.0/temp_0.9_seed_42.mid` | `outputs/ab_test/v6.1.0/temp_0.9_seed_42_with_rhythm.mid` | 100 | 0.5474 | 0.7390 | 48-75 | 63, 61, 62, 60, 59, 55, 53, 51, 50, 48, 63, 60, 56, 51, 53, 54 |
| v6.1.0 | 1.1 | 1337 | `outputs/ab_test/v6.1.0/temp_1.1_seed_1337.mid` | `outputs/ab_test/v6.1.0/temp_1.1_seed_1337_with_rhythm.mid` | 123 | 0.3500 | 0.3452 | 48-84 | 57, 56, 55, 59, 63, 67, 65, 60, 56, 55, 62, 60, 56, 55, 58, 54 |
| v6.2.0 | 0.9 | 42 | `outputs/ab_test/v6.2.0/temp_0.9_seed_42.mid` | `outputs/ab_test/v6.2.0/temp_0.9_seed_42_with_rhythm.mid` | 113 | 0.4100 | 0.4428 | 48-75 | 48, 54, 56, 57, 55, 53, 54, 56, 55, 53, 50, 51, 48, 63, 58, 53 |
| v6.2.0 | 1.1 | 1337 | `outputs/ab_test/v6.2.0/temp_1.1_seed_1337.mid` | `outputs/ab_test/v6.2.0/temp_1.1_seed_1337_with_rhythm.mid` | 125 | 0.3148 | 0.3294 | 48-81 | 48, 51, 53, 57, 55, 56, 58, 59, 62, 60, 63, 61, 62, 60, 57, 59 |

## Sanity Notes

- `outputs/ab_test/v6.1.0/temp_0.9_seed_42.mid` / `outputs/ab_test/v6.1.0/temp_0.9_seed_42_with_rhythm.mid`: no obvious structural issue.
- `outputs/ab_test/v6.1.0/temp_1.1_seed_1337.mid` / `outputs/ab_test/v6.1.0/temp_1.1_seed_1337_with_rhythm.mid`: no obvious structural issue.
- `outputs/ab_test/v6.2.0/temp_0.9_seed_42.mid` / `outputs/ab_test/v6.2.0/temp_0.9_seed_42_with_rhythm.mid`: no obvious structural issue.
- `outputs/ab_test/v6.2.0/temp_1.1_seed_1337.mid` / `outputs/ab_test/v6.2.0/temp_1.1_seed_1337_with_rhythm.mid`: no obvious structural issue.
