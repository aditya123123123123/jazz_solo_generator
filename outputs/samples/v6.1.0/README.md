# v6.1.0 qualitative sample pack

Date: 2026-05-20

Checkpoint: `checkpoints/v6.1.0_best.pt`

These samples were generated after the successful v6.1.0 harmonic-loss finetune. Rhythm-section backing was added with the existing `src.generation.rhythm_section.generate_rhythm_section` API and `src.generation.midi_mixer.mix_to_midi`; no generation modules were rewritten or duplicated.

WAV rendering was skipped on this Mac mini because `fluidsynth` was not installed and `JAZZ_SOUNDFONT` was unset. The `_with_rhythm.mid` files are available for listening or later rendering.

## Files

| sample | solo MIDI | chords used for backing | tempo | style | dry render | with-rhythm MIDI | with-rhythm WAV |
|---|---|---|---:|---|---|---|---|
| `artist_coltrane_ii_V_I_C` | `artist_coltrane_ii_V_I_C.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | n/a | `artist_coltrane_ii_V_I_C_with_rhythm.mid` | skipped: fluidsynth or soundfont unavailable |
| `artist_miles_ii_V_I_C` | `artist_miles_ii_V_I_C.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | n/a | `artist_miles_ii_V_I_C_with_rhythm.mid` | skipped: fluidsynth or soundfont unavailable |
| `baseline_blues_F_parker` | `baseline_blues_F_parker.mid` | F7(4) Bb7(4) F7(4) F7(4) Bb7(4) Bb7(4) F7(4) D7(4) Gm7(4) C7(4) F7(4) C7(4) | 120.0 | swing | n/a | `baseline_blues_F_parker_with_rhythm.mid` | skipped: fluidsynth or soundfont unavailable |
| `baseline_ii_V_I_C_parker` | `baseline_ii_V_I_C_parker.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | n/a | `baseline_ii_V_I_C_parker_with_rhythm.mid` | skipped: fluidsynth or soundfont unavailable |
| `chord_perturbed_blues_F_parker` | `chord_perturbed_blues_F_parker.mid` | F7(4) Bb7(4) F7(4) F7(4) Bb7(4) Bb7(4) F7(4) D7(4) Gm7(4) C7(4) F7(4) C7(4) | 120.0 | swing | n/a | `chord_perturbed_blues_F_parker_with_rhythm.mid` | skipped: fluidsynth or soundfont unavailable |
| `chord_perturbed_ii_V_I_C_parker` | `chord_perturbed_ii_V_I_C_parker.mid` | Dm7(4) G7(4) Cj7(8) Cj7(8) | 120.0 | swing | n/a | `chord_perturbed_ii_V_I_C_parker_with_rhythm.mid` | skipped: fluidsynth or soundfont unavailable |

## What to listen for

- Baseline samples should sound more harmonically grounded against the backing than the earlier v6 checkpoint, consistent with v6.1.0 recovering chord sensitivity.
- Chord-perturbed samples use the intended chart changes in the backing while the generated solo was conditioned on intentionally wrong chord IDs. They should sound harmonically out of place against the rhythm section in places, matching the ablation result where `chord_shuffled` costs `+0.1685` pitch CE.
- Artist-comparison samples may sound close to baseline. That is expected: artist conditioning remains weak in the ablation, with only `+0.0208` pitch CE when artist ID is zeroed.
- Phrase coherence should be more audible than in v6: phrase-zeroed and phrase-shuffled conditions both hurt v6.1.0 pitch CE.

## Generation notes

- Backing style: `swing` for all samples.
- Rhythm seed: `42` for all samples.
- Backing chords come from each JSON sidecar's `intended_chord` and `beats` fields. This is deliberate for chord-perturbed samples so the wrong-chord solo can be heard against the intended form.
- Original dry MIDI files were not overwritten.
