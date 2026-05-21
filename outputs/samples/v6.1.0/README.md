# v6.1.0 qualitative sample pack

Date: 2026-05-21

NoteExecutor checkpoint: `checkpoints/v6.1.0_best.pt`

```text
61a06d7ebbad5d2f2416345ea12569a6f357dcca9376164960ff5f54a97f9061  checkpoints/v6.1.0_best.pt
```

PhrasePlanner checkpoint: `checkpoints/phrase_planner_best.pt`

```text
e9e27ba1c922a8a94482771d76cce20cc8aa91c057a0e577f8df7782e88943d3  checkpoints/phrase_planner_best.pt
```

The phrase-planner checkpoint was recovered by retraining `src.training.train_phrase_planner` on this Mac mini, then the active v6.1.0 qualitative sample pack was regenerated so phrase selection no longer depends on the previous heuristic fallback planner.

`generate_solo.py` supports explicit NoteExecutor checkpoint selection with `--note-executor-checkpoint` and qualitative chord perturbation with `--chord-shuffled` / `--chord-zeroed`.

Rhythm-section backing was added with the existing `src.generation.rhythm_section.generate_rhythm_section` API and `src.generation.midi_mixer.mix_to_midi`. For perturbed samples, the solo model received shuffled or zeroed chord IDs, but the rhythm section always used the original unperturbed chord progression.

WAV rendering was skipped because `fluidsynth` was not installed and `JAZZ_SOUNDFONT` was unset. The `_with_rhythm.mid` files are available for listening or later rendering.

## Files

| sample | condition | JSON metadata | solo MIDI | chords used for backing | tempo | style | generation method | with-rhythm MIDI |
|---|---|---|---|---|---:|---|---|---|
| `baseline_ii_V_I_C_parker` | `normal` | `baseline_ii_V_I_C_parker.json` | `baseline_ii_V_I_C_parker.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | generate_solo.generate_solo with PhrasePlanner checkpoint | `baseline_ii_V_I_C_parker_with_rhythm.mid` |
| `baseline_blues_F_parker` | `normal` | `baseline_blues_F_parker.json` | `baseline_blues_F_parker.mid` | F7(4) Bb7(4) F7(4) F7(4) Bb7(4) Bb7(4) F7(4) D7(4) Gm7(4) C7(4) F7(4) C7(4) | 120.0 | swing | generate_solo.generate_solo with PhrasePlanner checkpoint | `baseline_blues_F_parker_with_rhythm.mid` |
| `artist_miles_ii_V_I_C` | `normal` | `artist_miles_ii_V_I_C.json` | `artist_miles_ii_V_I_C.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | generate_solo.generate_solo with PhrasePlanner checkpoint | `artist_miles_ii_V_I_C_with_rhythm.mid` |
| `artist_coltrane_ii_V_I_C` | `normal` | `artist_coltrane_ii_V_I_C.json` | `artist_coltrane_ii_V_I_C.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | generate_solo.generate_solo with PhrasePlanner checkpoint | `artist_coltrane_ii_V_I_C_with_rhythm.mid` |
| `chord_perturbed_ii_V_I_C_parker` | `chord_shuffled` | `chord_perturbed_ii_V_I_C_parker.json` | `chord_perturbed_ii_V_I_C_parker.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | generate_solo.generate_solo with PhrasePlanner checkpoint | `chord_perturbed_ii_V_I_C_parker_with_rhythm.mid` |
| `chord_perturbed_blues_F_parker` | `chord_shuffled` | `chord_perturbed_blues_F_parker.json` | `chord_perturbed_blues_F_parker.mid` | F7(4) Bb7(4) F7(4) F7(4) Bb7(4) Bb7(4) F7(4) D7(4) Gm7(4) C7(4) F7(4) C7(4) | 120.0 | swing | generate_solo.generate_solo with PhrasePlanner checkpoint | `chord_perturbed_blues_F_parker_with_rhythm.mid` |
| `chord_zeroed_ii_V_I_C_parker` | `chord_zeroed` | `chord_zeroed_ii_V_I_C_parker.json` | `chord_zeroed_ii_V_I_C_parker.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | generate_solo.generate_solo with PhrasePlanner checkpoint | `chord_zeroed_ii_V_I_C_parker_with_rhythm.mid` |

## Listening guide

- Baseline samples use normal chord conditioning and are the reference for v6.1.0 conditioned generation.
- `artist_miles_ii_V_I_C` and `artist_coltrane_ii_V_I_C` test artist conditioning. Expect subtle or no obvious difference because the ablation found artist conditioning remains weak (`+0.0208` pitch CE when artist ID is zeroed).
- `chord_perturbed_*` samples use `--chord-shuffled`: the model heard wrong chord IDs, while the rhythm section plays the correct chart. Listen for harmonic mismatch, weaker resolution, or tension landing in the wrong places.
- `chord_zeroed_ii_V_I_C_parker` uses `--chord-zeroed`: the model heard no usable chord context while the rhythm section plays the correct ii-V-I. It should sound more random harmonically than the baseline.
- The contrast between baseline and chord-perturbed samples is the qualitative version of the recovered chord sensitivity measured in the v6.1.0 ablation (`chord_shuffled` cost `+0.1685` pitch CE).

## Backup

The previous active sample directory was moved to:

`outputs/samples/v6.1.0_WRONG_CHECKPOINT_BACKUP/`
