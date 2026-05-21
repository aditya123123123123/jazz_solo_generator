# v6.1.0 qualitative sample pack

Date: 2026-05-21

Checkpoint: `checkpoints/v6.1.0_best.pt`

Checkpoint SHA-256:

```text
61a06d7ebbad5d2f2416345ea12569a6f357dcca9376164960ff5f54a97f9061  checkpoints/v6.1.0_best.pt
```

These samples were regenerated from the correct v6.1.0 checkpoint after discovering that the previous sample-pack handoff did not provide a traceable CLI path to `checkpoints/v6.1.0_best.pt`. The generation script now supports `--note-executor-checkpoint` for explicit checkpoint selection while preserving the old default behavior.

The public generation CLI still does not expose chord-zeroed/chord-shuffled perturbation flags, and `checkpoints/phrase_planner_best.pt` is not present on this Mac mini. For this pack, samples were generated through the existing `generate_solo.py` NoteExecutor loading/export helpers with fixed chord-family phrase IDs:

- minor family: `PHRASE_18`
- major family: `PHRASE_52`
- dominant family: `PHRASE_57`

Rhythm-section backing was added with the existing `src.generation.rhythm_section.generate_rhythm_section` API and `src.generation.midi_mixer.mix_to_midi`. No generation modules were duplicated.

WAV rendering was skipped because `fluidsynth` was not installed and `JAZZ_SOUNDFONT` was unset. The `_with_rhythm.mid` files are available for listening or later rendering.

## Files

| sample | JSON metadata | solo MIDI | chords used for backing | tempo | style | with-rhythm MIDI |
|---|---|---|---|---:|---|---|
| `baseline_ii_V_I_C_parker` | `baseline_ii_V_I_C_parker.json` | `baseline_ii_V_I_C_parker.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | `baseline_ii_V_I_C_parker_with_rhythm.mid` |
| `baseline_blues_F_parker` | `baseline_blues_F_parker.json` | `baseline_blues_F_parker.mid` | F7(4) Bb7(4) F7(4) F7(4) Bb7(4) Bb7(4) F7(4) D7(4) Gm7(4) C7(4) F7(4) C7(4) | 120.0 | swing | `baseline_blues_F_parker_with_rhythm.mid` |
| `artist_miles_ii_V_I_C` | `artist_miles_ii_V_I_C.json` | `artist_miles_ii_V_I_C.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | `artist_miles_ii_V_I_C_with_rhythm.mid` |
| `artist_coltrane_ii_V_I_C` | `artist_coltrane_ii_V_I_C.json` | `artist_coltrane_ii_V_I_C.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | `artist_coltrane_ii_V_I_C_with_rhythm.mid` |

## What to listen for

- Baseline samples should sound more harmonically grounded against the backing than the earlier v6 checkpoint, consistent with v6.1.0 recovering chord sensitivity.
- Chord-shuffled qualitative samples are not included in this regenerated pack because `generate_solo.py` does not expose chord perturbation at inference. The quantitative ablation remains the source of truth for the chord-shuffled result: `+0.1685` pitch CE.
- Artist-comparison samples may sound close to baseline. That is expected: artist conditioning remains weak in the ablation, with only `+0.0208` pitch CE when artist ID is zeroed.
- Phrase coherence should be more audible than in v6: phrase-zeroed and phrase-shuffled conditions both hurt v6.1.0 pitch CE.

## Backup

The previous active sample directory was moved to:

`outputs/samples/v6.1.0_WRONG_CHECKPOINT_BACKUP/`
