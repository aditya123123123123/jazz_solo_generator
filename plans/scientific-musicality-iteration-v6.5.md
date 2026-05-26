# Scientific Musicality Iteration Plan v6.5

Goal: iterate from the current v6.4 phrase-shaped probe toward solos that sound musical, jazzy, and stay inside the chord changes.

## Method

Each experiment changes exactly one primary variable. For every iteration:

1. State the hypothesis before changing code/training.
2. Freeze the baseline artifacts and metrics.
3. Make one minimal change.
4. Run focused tests.
5. Generate the same matched probe set: `ii_V_I_C`, `blues_F`, `autumn_leaves`.
6. Add/enable a rhythm section for all audition/listening generations from now on, then render MP3s from the rhythm-section MIDI and listen.
7. Score objective metrics, including accompaniment verification counts when rhythm-section artifacts are produced.
8. Accept, reject, or keep-for-later based on the prewritten gate.
9. Log results here before choosing the next experiment.

## Evaluation probes

Standing artifact rule from Aditya (2026-05-24): all generated listening/audition outputs should include a rhythm section from now on. Direct generation should use `--with-rhythm-section`; post-processed outputs should merge piano, bass, and drums from explicit chord metadata before rendering. Solo-only generations are acceptable only for internal diagnostics and should not be the delivered listening artifacts.

- `ii_V_I_C`: coherence and clean voice-leading sanity check.
- `blues_F`: jazz vocabulary, repetition/diversity, blues language.
- `autumn_leaves`: longer-form harmonic continuity and phrase development.

## Core metrics

Primary objective metrics:

- Chord-faithfulness: pure chord tone %, chord+extension %, outside %.
- Voice-leading continuity: count and rate of melodic leaps >= 12 semitones, max leap, mean leap.
- Phrase faithfulness: contour match rate, note-count error, density error, rest-ratio error.
- Repetition: most repeated phrase ID / total phrase sections.
- Cadence resolution rate.

Subjective listening rubric:

- Does it sound like a coherent solo rather than disconnected note fragments?
- Does it sound jazzy rather than random diatonic/chord-tone picking?
- Does it outline the changes clearly?
- Does it have rests, phrasing, and motivic development?
- Are any artifacts obvious: teleporting register jumps, machine-gun density, phrase stamping, dead-air, wrong notes?

## Current baseline: v6.4 phrase-shaped probe

Artifacts:

- JSON/MIDI: `outputs/solos_v6.4_phrase_shaped_probe/`
- Rendered WAV: `outputs/rendered_audio_v64_phrase_shaped_probe/`
- MP3: `outputs/mp3_v64_phrase_shaped_probe/`
- Phrase metrics: `outputs/phrase_faithfulness_v64_phrase_shaped_probe.json`

Observed baseline metrics:

- Aggregate phrase contour match: 87.5%.
- Aggregate cadence resolution: 75.0%.
- Aggregate mean note-count error: 2.21.
- Aggregate outside rate from quick analysis: Autumn Leaves 6.9%, Blues F 11.1%, ii-V-I C 4.3%.
- Octave+ leaps: Autumn Leaves 14, Blues F 19, ii-V-I C 1.
- Phrase repetition: Blues F repeats `PHRASE_53` 6/12 sections.

Listening/analysis notes:

- Phrase control is real but not enough.
- Main musical failure is register discontinuity / teleporting leaps.
- Blues is contour-faithful but too repetitive.
- ii-V-I is the cleanest proof-of-concept.

## Experiment 1: Inference-time register continuity constraint

Hypothesis: If the sampler chooses the nearest octave-equivalent pitch to the prior sounding note, constrained to playable solo range, then large register teleports will drop sharply without retraining, while chord faithfulness and phrase contour remain roughly unchanged.

Primary variable changed: post-sampling octave placement only. No model architecture, dataset, loss, phrase planner, or temperature changes.

Acceptance gate:

- Reduce octave+ leaps by at least 50% on Autumn Leaves and Blues F.
- Do not increase outside-note rate by more than +2 percentage points on any probe.
- Do not reduce aggregate contour match below 80%.
- Listening: fewer obvious register teleports.

If accepted: keep the continuity constraint as default and move to Experiment 2.

If rejected: tune or remove it before trying any other change.

## Experiment results

### Experiment 1: register continuity

Result: accepted as a continuity fix.

Matched baseline vs Exp 1:

- Autumn octave+ leaps: 13 -> 0; max leap 27 -> 10.
- Blues octave+ leaps: 18 -> 0; max leap 31 -> 10.
- ii-V-I octave+ leaps: 1 -> 0; max leap 14 -> 8.
- Chord/outside rates unchanged versus the matched baseline, because the transform preserves pitch class.

Interpretation: this fixes register teleporting without changing harmony. Keep it on by default for probe experiments.

### Experiment 2: all-position chord-tone bias

Hypothesis: Enabling existing chord-tone bias on every generated position, while keeping register continuity, will improve chord faithfulness without reintroducing leaps.

Result: accepted as a harmony fix.

Exp 1 -> Exp 2:

- Autumn outside: 23.0% -> 5.9%; pure chord tones: 48.9% -> 76.5%; octave+ leaps remain 0.
- Blues outside: 19.1% -> 9.2%; pure chord tones: 49.6% -> 73.8%; octave+ leaps remain 0.
- ii-V-I outside: 26.1% -> 4.3%; pure chord tones: 54.3% -> 80.4%; octave+ leaps remain 0.

Interpretation: the model can be steered toward staying inside the changes at inference time. Remaining failure: Blues still repeats `PHRASE_53` 6/12 sections, so next single-variable experiment should target phrase diversity/repetition, not retraining yet.

### Experiment 3: phrase diversity constraint

Hypothesis: Limiting repeated phrase IDs after planning will reduce phrase-stamping in Blues without harming harmony, register continuity, or phrase faithfulness too much.

Primary variable changed: phrase-plan post-processing only. No retraining, no note model change, no harmony-bias change.

Result: accepted with `--phrase-max-uses 3`; rejected/aggressive at `--phrase-max-uses 2`.

Exp 2 -> Exp 3 max-uses=2:

- Blues max repeated phrase: 6/12 -> 2/12.
- Blues outside: 9.2% -> 10.3%.
- Blues cadence resolution: 83.3% -> 66.7%.

Interpretation: max-uses=2 fixes repetition but is too aggressive; it hurts cadence.

Exp 2 -> Exp 3b max-uses=3:

- Blues max repeated phrase: 6/12 -> 3/12.
- Blues remapped phrase sections: 3.
- Blues outside: 9.2% -> 7.8%.
- Blues pure chord tones: 73.8% -> 78.9%.
- Blues octave+ leaps remain 0; max leap remains 10.
- Blues cadence resolution: 83.3% -> 75.0%.
- Aggregate contour match unchanged at 41.7%.
- Aggregate pitch-range error improves 4.50 -> 3.83.

Interpretation: max-uses=3 is the better tradeoff. It substantially reduces phrase stamping and slightly improves harmonic purity/range metrics, with a modest cadence regression. Keep this as the current candidate pending listening.

Current best candidate settings:

```bash
--note-executor-checkpoint checkpoints/v6.3.2_best.pt \
--phrase-shaping \
--register-continuity \
--chord-tone-bias --all-beat-chord-tone-bias \
--phrase-diversity --phrase-max-uses 3
```

Latest best artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp3b_phrase_diversity_max3_probe/`
- MP3: `outputs/mp3_v65_exp3b_phrase_diversity_max3/`
- Metrics: `outputs/musicality_metrics_v65_exp3b_phrase_diversity_max3.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp3b_phrase_diversity_max3.json`

### Experiment 4: rhythm density/rest calibration

Hypothesis: If phrase-shaped generation keeps the current note sampler but reactivates sampled rest positions until each section reaches its phrase-cluster note-count target, then density/rest phrase faithfulness will improve without changing harmony controls, phrase planning, register continuity, checkpoint, training, or dataset.

Primary variable changed: post-decode rest flags only, behind `--rhythm-density-calibration`. Sampled pitches, durations, phrase IDs, chord-tone bias, and register-continuity logic were otherwise unchanged. I also fixed two artifact-requirement bugs discovered during verification: rhythm-section exports now name the source track `Solo` when piano/bass/drums are present, and direct generation repeats the explicit chord progression enough times for accompaniment to cover the full solo.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py tests/test_rhythm_section.py
# 15 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp4_rhythm_density_calibrated_probe/`
- MP3: `outputs/mp3_v65_exp4_rhythm_density_calibrated/`
- MP3 ZIP: `outputs/v65_exp4_rhythm_density_calibrated_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp4_rhythm_density_calibrated.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp4_rhythm_density_calibrated.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp4_rhythm_density_calibrated.json`

Exp 3b -> Exp 4 objective comparison:

- Aggregate mean note-count error improved: 1.875 -> 1.083.
- Aggregate contour match improved: 41.7% -> 45.8%.
- Aggregate cadence unchanged: 83.3% -> 83.3%.
- Aggregate density error regressed: 1.100 -> 1.263.
- Aggregate rest-ratio error regressed: 0.0799 -> 0.0872.
- Aggregate pitch-range error regressed: 3.83 -> 4.25.
- Register continuity stayed controlled: octave+ leaps remain 0 on all probes.
- Phrase repetition stayed controlled: Blues max repeated phrase remains 3/12.
- Harmony stayed broadly controlled in section-based metrics: Autumn outside 5.9% -> 5.8%, Blues outside 7.8% -> 7.5%, ii-V-I outside 4.3% -> 4.3%; Blues pure chord tones 78.9% -> 79.9%.

Accompaniment verification on generated MIDI:

- Autumn Leaves tracks: Solo 155, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0.
- Blues F tracks: Solo 134, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0.

Result: rejected as the next best musicality setting. The narrow rest-reactivation change did what it was told on note count, and it did not break register, repetition, or section-based harmony, but the actual primary target was density/rest feel. Those aggregate density and rest-ratio metrics regressed, especially on Autumn Leaves. Keep the artifact fixes, but do not add `--rhythm-density-calibration` to the current best candidate.

Interpretation: phrase-cluster `median_num_notes` and `median_density` are not internally equivalent under the generated duration grid; forcing note count alone can overfill some phrases and make breathing worse. The next single-variable experiment should target duration/rhythm directly, not rest flags: a small matched probe of lower `--duration-temperature` (or a deterministic duration quantile clamp) while leaving rest behavior unchanged.

### Experiment 5: lower duration-temperature rhythm-grid probe

Hypothesis: Reducing only the duration sampling temperature from 1.6 to 1.0 will make rhythmic durations less noisy and improve density/rest phrase faithfulness without changing phrase planning, note count targets, chord-tone bias, register continuity, rest behavior, checkpoint, training, or dataset.

Primary variable changed: `--duration-temperature 1.0` only, relative to the Exp 3b current-best candidate. No production code change was made for this probe.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py tests/test_rhythm_section.py
# 15 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp5_duration_temp_1p0_probe/`
- MP3: `outputs/mp3_v65_exp5_duration_temp_1p0/`
- MP3 ZIP: `outputs/v65_exp5_duration_temp_1p0_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp5_duration_temp_1p0.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp5_duration_temp_1p0.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp5_duration_temp_1p0.json`

Exp 3b -> Exp 5 objective comparison:

- Aggregate density error improved slightly: 1.100 -> 1.085.
- Aggregate contour match improved: 41.7% -> 45.8%.
- Aggregate note-count error regressed: 1.875 -> 2.042.
- Aggregate rest-ratio error regressed: 0.0799 -> 0.1107.
- Aggregate cadence resolution regressed: 83.3% -> 70.8%.
- Aggregate pitch-range error regressed: 3.83 -> 5.38.
- Register continuity stayed controlled: octave+ leaps remain 0 on all probes; max leap stayed <= 11.
- Phrase repetition stayed controlled: Blues max repeated phrase remains 3/12.
- Harmony regressed mildly: Autumn outside 5.9% -> 6.8%, Blues outside 7.8% -> 7.9%, ii-V-I outside 4.3% -> 6.5%; Blues pure chord tones 78.9% -> 76.4%.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 133, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0.
- Blues F tracks: Solo 127, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0.
- ii-V-I C tracks: Solo 46, Piano 36, Bass 24, Drums 66; bass boundaries 4/4; piano bad tones 0.

Result: rejected as the next best musicality setting. Lowering duration temperature produced a small density-error win, but it worsened the more audible breathing proxy (rest-ratio error), cadence, range faithfulness, and ii-V-I harmony. Keep Exp 3b as the current best candidate.

Interpretation: duration randomness is not the main density/rest failure. The next single-variable experiment should preserve the Exp 3b duration temperature and target cadence preservation in phrase-diverse replacements, because Exp 3b's biggest known tradeoff is Blues cadence regression from 83.3% to 75.0% while repetition is now controlled.

### Experiment 6: cadence-preserving phrase diversity replacement

Hypothesis: If phrase-diversity remaps prefer alternatives with similar training-set final-chord-tone cadence rates, then the Exp 3b Blues cadence regression can recover without changing checkpoint, note sampling, chord-tone bias, register continuity, phrase max-uses, duration/rest behavior, training, or dataset.

Primary variable changed: phrase-diversity replacement tie-breaker only, behind `--phrase-diversity-preserve-cadence`. I added a nondefault phrase-feature field, `final_chord_tone_rate`, estimated from `phrases_all.json` by carrying forward the active chord inside each training phrase and checking whether the final note is a shell chord tone. Current-best generation without the new flag remains the matched control.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_diversity.py tests/test_phrase_shaping_inference.py tests/test_rhythm_section.py
# 18 passed
```

Generated artifacts:

- Matched current-best JSON/MIDI: `outputs/solos_v6.5_exp6_matched_current_best_probe/`
- Exp 6 JSON/MIDI: `outputs/solos_v6.5_exp6_cadence_preserving_phrase_diversity_probe/`
- MP3: `outputs/mp3_v65_exp6_cadence_preserving_phrase_diversity/`
- MP3 ZIP: `outputs/v65_exp6_cadence_preserving_phrase_diversity_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp6_cadence_preserving_phrase_diversity.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp6_cadence_preserving_phrase_diversity.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp6_cadence_preserving_phrase_diversity.json`

Matched current-best -> Exp 6 objective comparison:

- Aggregate outside improved: 6.0% -> 4.67%.
- Aggregate pure chord tones slightly regressed: 78.6% -> 77.83%.
- Aggregate cadence resolution regressed: 83.3% -> 70.8%.
- Aggregate contour match regressed: 41.7% -> 33.3%.
- Aggregate note-count error improved: 1.875 -> 1.708.
- Aggregate density error improved: 1.100 -> 0.983.
- Aggregate rest-ratio error regressed: 0.0799 -> 0.0964.
- Aggregate pitch-range error regressed: 3.83 -> 4.17.
- Register continuity stayed controlled: octave+ leaps remain 0 on all probes.
- Phrase repetition stayed controlled: max repeated phrase remains <= 3.
- Blues outside improved 7.8% -> 5.1%, but Blues cadence fell 75.0% -> 66.7%.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 108, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0.
- Blues F tracks: Solo 137, Piano 216, Bass 144, Drums 396; bass boundaries 36/36; piano bad tones 0.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0.

Result: rejected as the next best musicality setting. The learned final-chord-tone-rate proxy is too coarse: PHRASE_53, PHRASE_22, PHRASE_42, and PHRASE_40 all have similar corpus cadence rates (~0.66-0.70), but swapping among them still changes local generated cadences through note-sampler context. The metric helped outside-note rate, but it failed the primary cadence-preservation goal and also hurt contour/rest/range proxies. Keep Exp 3b as the current best candidate; do not add `--phrase-diversity-preserve-cadence` to the recommended settings.

Interpretation: phrase-level cadence metadata alone is not a reliable control unless generation is seeded independently per section or cadence is enforced at the note decoder. The next single-variable experiment should target the actual failure surface directly: a lightweight per-section final-note chord-tone enforcement for every chord section, not just the final segment, while leaving phrase plan/diversity and other sampling settings unchanged.

### Experiment 7: per-section cadence enforcement

Hypothesis: If each generated section's final sounding note is post-processed to the nearest active chord tone only when it ends outside the shell, then cadence resolution will improve without changing note counts, durations, rests, phrase planning, checkpoint, training, dataset, register-continuity logic, or the body of the phrase.

Primary variable changed: final sounding pitch of each section only, behind `--section-cadence-enforcement`. The method preserves duration/rest grids and all non-final notes. I also added `scripts/render_midi_dir_to_mp3.py` so rhythm-section MIDI directories can be rendered directly to MP3 previews without leaving WAV artifacts behind.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py tests/test_phrase_diversity.py tests/test_rhythm_section.py
# 20 passed

PYTHONPATH=. .venv/bin/python -m pytest tests/
# 64 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp7_section_cadence_probe/`
- MP3: `outputs/mp3_v65_exp7_section_cadence/`
- MP3 ZIP: `outputs/v65_exp7_section_cadence_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp7_section_cadence.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp7_section_cadence.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp7_section_cadence.json`

Matched current-best -> Exp 7 objective comparison:

- Aggregate cadence resolution improved: 83.3% -> 100.0%.
- Aggregate outside improved: 6.45% -> 5.79%.
- Aggregate pure chord tones improved: 78.07% -> 79.35%.
- Blues outside improved: 7.8% -> 6.2%; Blues pure chord tones improved: 78.9% -> 81.2%.
- ii-V-I pure chord tones improved: 80.4% -> 82.6%; outside unchanged at 4.3%.
- Autumn was unchanged by the cadence post-process: already 8/8 cadences.
- Aggregate contour match regressed: 41.7% -> 33.3%.
- Aggregate density error unchanged: 1.100 -> 1.100.
- Aggregate note-count error unchanged: 1.875 -> 1.875.
- Aggregate rest-ratio error unchanged: 0.0799 -> 0.0799.
- Aggregate pitch-range error slightly improved: 3.83 -> 3.79.
- Register continuity stayed controlled: octave+ leaps remain 0 on all probes; max leap stayed <= 10.
- Phrase repetition stayed controlled: Blues max repeated phrase remains 3/12.
- Cadence edits made: Autumn 0/8 sections, Blues 3/12, ii-V-I 1/4.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 136, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0.
- Blues F tracks: Solo 128, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0.

Result: provisional / not yet promoted as the default current best. The experiment cleanly fixed the direct cadence target and improved strict harmony without altering rhythm/density/repetition, but the blunt nearest-tone final-pitch edit flipped enough section contour labels to create an objective contour regression. This may still sound better because only final notes changed, but without human listening I should not promote it over Exp 3b yet.

Interpretation: final-note chord-tone enforcement is promising, but the next single-variable experiment should make the cadence edit contour-aware: when multiple chord tones are available, prefer a final pitch that preserves the pre-edit generated contour if possible; otherwise fall back to nearest chord tone.

### Experiment 8: contour-aware per-section cadence enforcement

Hypothesis: If cadence enforcement chooses a chord-tone final pitch that preserves the pre-edit generated section contour when possible, then it will keep Exp 7's cadence/harmony benefit while recovering Exp 3b's contour-match rate.

Primary variable changed: cadence-enforcement candidate selection only, behind `--section-cadence-preserve-contour` and only active with `--section-cadence-enforcement`. It searches in-range active chord tones and prefers candidates whose edited section contour matches the pre-edit generated contour before falling back to nearest chord tone. No model, phrase plan, duration/rest, chord-bias, register-continuity, dataset, or training changes.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py tests/test_phrase_diversity.py tests/test_rhythm_section.py
# 21 passed

PYTHONPATH=. .venv/bin/python -m pytest tests/
# 65 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp8_contour_aware_section_cadence_probe/`
- MP3: `outputs/mp3_v65_exp8_contour_aware_section_cadence/`
- MP3 ZIP: `outputs/v65_exp8_contour_aware_section_cadence_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp8_contour_aware_section_cadence.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp8_contour_aware_section_cadence.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp8_contour_aware_section_cadence.json`

Exp 3b current-best -> Exp 8 objective comparison:

- Aggregate cadence resolution improved: 83.3% -> 100.0%.
- Aggregate outside improved across scored probes: Autumn unchanged 5.9%, Blues 7.8% -> 6.2%, ii-V-I unchanged 4.3%.
- Aggregate pure chord tones improved on edited probes: Blues 78.9% -> 81.2%, ii-V-I 80.4% -> 82.6%; Autumn unchanged 76.5%.
- Aggregate contour match still regressed: 41.7% -> 37.5%.
- Blues contour match regressed: 41.7% -> 33.3%; ii-V-I stayed 50.0%; Autumn stayed 37.5%.
- Density, note-count, and rest-ratio errors were unchanged: density 1.100 -> 1.100, note-count 1.875 -> 1.875, rest-ratio 0.0799 -> 0.0799.
- Pitch-range error slightly improved: 3.83 -> 3.79.
- Register continuity stayed controlled: octave+ leaps remain 0 on all probes; max leap stayed <= 10.
- Phrase repetition stayed controlled: Blues max repeated phrase remains 3/12.
- Cadence edits made: Autumn 0/8 sections, Blues 3/12, ii-V-I 1/4.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 136, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0/180.
- Blues F tracks: Solo 128, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0/144.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0/72.

Result: objective contour gate failed, but after A/B review Aditya accepted Exp 8 as the more musical direction and provisional human-approved baseline. It kept the cadence/harmony benefits of Exp 7 and did not damage rhythm or repetition metrics; the remaining flaw was contour target mismatch.

Interpretation: compare future musicality probes against Exp 8, not Exp 3b, unless the new probe loses the cadence/harmony gains that made Exp 8 preferable by ear. The next single-variable experiment should target the actual phrase target rather than the generated contour: target-contour-aware cadence enforcement, choosing a chord-tone final pitch that improves or preserves match to `phrase_features.contour` when available.

### Experiment 9: target-contour-aware per-section cadence enforcement

Hypothesis: If cadence enforcement chooses a chord-tone final pitch that matches the phrase-cluster target contour when possible, then it will keep Exp 8's 100% cadence/harmony gains while recovering contour match toward Exp 3b.

Primary variable changed: cadence-enforcement candidate selection only, behind `--section-cadence-target-contour` and active with `--section-cadence-enforcement`. It prefers in-range active chord-tone candidates whose edited section contour matches `phrase_features.contour`; if no target match exists, it falls back to nearest chord tone. No model, phrase plan, duration/rest, chord-bias, register-continuity, dataset, or training changes.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py tests/test_phrase_diversity.py tests/test_register_continuity.py tests/test_rhythm_section.py tests/test_phrase_plan_trace.py tests/test_phrase_position_embed.py -q
# 30 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp9_target_contour_section_cadence_probe/`
- MP3: `outputs/mp3_v65_exp9_target_contour_section_cadence/`
- MP3 ZIP: `outputs/v65_exp9_target_contour_section_cadence_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp9_target_contour_section_cadence.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp9_target_contour_section_cadence.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp9_target_contour_section_cadence.json`

Exp 8 human-approved baseline -> Exp 9 objective comparison:

- Aggregate cadence resolution stayed perfect: 100.0% -> 100.0%.
- Aggregate contour match improved: 37.5% -> 41.7%, recovering to the Exp 3b level.
- Harmony metrics were unchanged on the scored probes: Autumn outside 5.9%, Blues outside 6.2%, ii-V-I outside 4.3%; Blues pure chord tones 81.2%, ii-V-I pure chord tones 82.6%.
- Density, note-count, rest-ratio, and pitch-range errors were unchanged: density 1.100, note-count 1.875, rest-ratio 0.0799, pitch-range 3.79.
- Register continuity stayed controlled: octave+ leaps remain 0 on all probes; max leap stayed <= 10.
- Phrase repetition stayed controlled: Blues max repeated phrase remains 3/12.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 136, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0/180.
- Blues F tracks: Solo 128, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0/144.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0/72.

Result: Aditya listened and said Exp 9 sounds better, but still not jazzy. Treat Exp 9 as the best correctness/musical-shape baseline, but not the final musical target.

Interpretation: cadence, contour, register, repetition, and harmonic safety are no longer the main blocker. The next single-variable experiment should target idiomatic jazz language directly: weak-beat chromatic approach notes into strong-beat chord tones, while keeping Exp 9's phrase plan, cadence, register, and rhythm-section settings intact.

### Experiment 10: weak-beat bebop approach notes

Hypothesis: If existing weak-beat notes immediately before strong-beat chord tones are rewritten as chromatic approach tones, then the generated solos will sound more idiomatic/jazzy while preserving Exp 9's cadence, contour, note count, register continuity, repetition, and rhythm section behavior.

Primary variable changed: a narrow post-process behind `--bebop-approach-notes`. It only changes an existing sounding weak-beat note immediately before an integer-beat chord-tone target. Durations, rests, note count, phrase plan, cadence targets, model, dataset, and training are unchanged. Cadence enforcement still runs after this layer.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py tests/test_phrase_diversity.py tests/test_register_continuity.py tests/test_rhythm_section.py tests/test_phrase_plan_trace.py tests/test_phrase_position_embed.py -q
# 32 passed

PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
# 68 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp10_bebop_approach_probe/`
- MP3: `outputs/mp3_v65_exp10_bebop_approach/`
- MP3 ZIP: `outputs/v65_exp10_bebop_approach_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp10_bebop_approach.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp10_bebop_approach.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp10_bebop_approach.json`

Exp 9 correctness baseline -> Exp 10 jazz-vocabulary probe objective comparison:

- Aggregate cadence resolution stayed perfect: 100.0% -> 100.0%.
- Aggregate contour match stayed unchanged: 41.7% -> 41.7%.
- Density, note-count, rest-ratio, and pitch-range errors stayed unchanged: density 1.100, note-count 1.875, rest-ratio 0.0799, pitch-range 3.79.
- Register continuity stayed controlled: octave+ leaps remain 0; max leap stayed <= 10.
- Phrase repetition stayed controlled: Blues max repeated phrase remains 3/12.
- Controlled chromaticism increased outside-note rate where approach notes were added:
  - Autumn outside 5.9% -> 8.8% with 5 bebop approach edits.
  - Blues outside 6.2% -> 8.6% with 3 bebop approach edits.
  - ii-V-I unchanged at 4.3% with 0 edits.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 136, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0/180.
- Blues F tracks: Solo 128, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0/144.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0/72.

Result: human listening accepted. Aditya said Exp 10 is "definitely a lot better." Treat Exp 10 as the current best musical baseline for the jazzy-feeling goal: it preserves Exp 9's correctness gates while moving the output toward intentional jazz vocabulary.

Interpretation: controlled chromatic approach notes are the right direction. The next experiment should expand vocabulary one step, not return to generic correctness tuning: add constrained two-note enclosures around important guide tones, especially 3rds/7ths on dominant chords, while preserving Exp 10's cadence, register, repetition, phrase plan, and accompaniment behavior.

## Candidate next experiments, one at a time

1. Add a second vocabulary layer for constrained two-note enclosures around 3rds/7ths on dominant chords, comparing against Exp 10.
2. Add blues-specific dominant vocabulary for Blues F only if enclosures improve the jazz feeling without sounding forced.
3. Run a training-data vocabulary diagnostic for blues/bebop devices if inference-time vocabulary layers plateau.
4. Chord-tone/extension bias strength grid only if listening says Exp 10 or enclosure probes are too boxed-in after vocabulary decisions.
5. Deterministic duration quantile clamp only if listening says rhythm is still too unstable after jazz-language work.

### Training-set jazz vocabulary audit v1

Question: should we expand the model's training set before adding more inference-time vocabulary layers?

Implemented `scripts/audit_jazz_vocabulary.py` to measure whether the current processed training corpus already contains learnable jazz vocabulary devices:

- beat-boundary chromatic approaches into chord tones
- two-note enclosures into guide tones
- guide-tone landings, especially 3rds/7ths
- dominant/blues color tones: b3, b5, b7 over dominant harmony

TDD/verification:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_jazz_vocabulary_audit.py -q
# 4 passed

PYTHONPATH=. .venv/bin/python scripts/audit_jazz_vocabulary.py \
  --phrases data/processed/phrases_all_with_tempo.json \
  --json-out outputs/jazz_vocabulary_audit_v1.json \
  --md-out outputs/jazz_vocabulary_audit_v1.md
```

Audit artifacts:

- `outputs/jazz_vocabulary_audit_v1.json`
- `outputs/jazz_vocabulary_audit_v1.md`

Overall corpus result from `data/processed/phrases_all_with_tempo.json`:

- Phrases: 10,163
- Analyzable notes: 126,625
- Beat-boundary approach candidates: 53,094, or 41.9% of analyzable notes
- Weak→strong chromatic approach rate: 14.1%, or 7,493/53,094 candidate pairs
- Guide-tone enclosure rate: 10.7%, or 2,908/27,061 guide-tone targets
- Guide-tone landing rate: 22.4%
- Chord-tone landing rate: 50.1%
- Dominant blues-color note rate: 21.3%, or 13,347/62,664 dominant-chord notes

Interpretation: the current corpus is not empty of jazz vocabulary. It contains measurable approach-note, enclosure, guide-tone, and blues-color material. That means the next issue is probably not only "we need any jazz data at all"; it is that the current note model/training objective/conditioning is not making those devices salient enough at generation time. Expanding the dataset can still help, especially if we add more targeted bebop/blues/transcription-heavy material, but the expansion should be paired with explicit vocabulary-aware training targets or sampling/evaluation metrics. Otherwise the model may continue averaging the language into safe but bland chord-tone output.

Recommended next move: build v6.6 training-data expansion/fine-tuning around these labels. Use the audit to create auxiliary targets for chromatic approach, enclosure, guide-tone landing, and blues-color events, then train/fine-tune and compare the raw model output against Exp 10 before applying post-processing.

### v6.6 vocabulary-aware training label plumbing

Implemented the first concrete step toward training the model to learn the Exp 10 jazz-language behavior natively instead of relying only on inference post-processing.

Changes:

- Added `src/data/jazz_vocab_labels.py` with target-aligned scalar labels per note:
  - `chromatic_approach_label`
  - `enclosure_label`
  - `guide_tone_label`
  - `blues_color_label`
  - aggregate `jazz_vocab_label`
- Updated `NoteWindowDataset` so every one-note training sample carries those labels.
- Updated `scripts/build_v6_cache.py` so rebuilt caches store the label tensors plus `target_chord_id` and `phrase_position` explicitly.
- Updated `collate_note_window` to preserve the labels in training batches while remaining backward-compatible with older caches.
- Added `--jazz-vocab-sample-weight` to `src/training/train_v6.py`; it upweights pitch loss on labeled vocabulary examples without changing the model architecture yet. This is deliberately conservative: it makes existing jazz events more salient before adding new heads or changing generation.

TDD/verification:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_jazz_vocab_training_labels.py -q
# 3 passed

PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
# 75 passed
```

Cache rebuilt locally at `data/processed/notes_v6_cache.pt` and verified:

- Samples: 2,216,748
- Cache size: 948.6 MB
- `jazz_vocab_label`: 462,936 samples, 20.88%
- `chromatic_approach_label`: 99,156 samples, 4.47%
- `enclosure_label`: 34,896 samples, 1.57%
- `guide_tone_label`: 340,860 samples, 15.38%
- `blues_color_label`: 160,164 samples, 7.23%

Dry-run training check:

```bash
WANDB_MODE=disabled PYTHONPATH=. .venv/bin/python src/training/train_v6.py \
  --epochs 1 \
  --dry-run-batches 1 \
  --jazz-vocab-sample-weight 2.0 \
  --output-dir outputs/tmp_vocab_dryrun_ckpt \
  --best-output outputs/tmp_vocab_dryrun_ckpt/best.pt \
  --latest-output outputs/tmp_vocab_dryrun_ckpt/latest.pt
```

Result: dry run completed; train and validation both consumed the rebuilt cache with vocabulary-weighted pitch loss. Temporary dry-run checkpoints were removed afterward.

Recommended v6.6 training command on the Windows GPU machine after pulling this commit and rebuilding/syncing the cache:

```bash
WANDB_MODE=online PYTHONPATH=. python src/training/train_v6.py \
  --resume checkpoints/v6.3.2_best.pt \
  --run-name note-executor-v6.6-vocab-weighted-ft \
  --jazz-vocab-sample-weight 2.0 \
  --epochs 20 \
  --lr 1e-4 \
  --best-output checkpoints/v6.6_vocab_weighted_best.pt \
  --latest-output checkpoints/v6.6_vocab_weighted_latest.pt
```

Evaluate by generating the usual Blues F / Autumn Leaves / ii-V-I probes from the raw v6.6 checkpoint first, then compare against Exp 10 before enabling post-processing. Acceptance requires the raw model to show more intentional chromatic approaches/blues language while preserving Exp 10-level cadence/harmony after the existing safe inference settings are applied.

### Experiment 11: constrained two-note guide-tone enclosures

Hypothesis: If Exp 10's accepted weak-beat chromatic approach layer is expanded by rewriting two existing pickup notes into chromatic enclosures around strong-beat guide tones (3rds/7ths), then the solos will sound more idiomatic/jazzy while preserving Exp 10's cadence, phrase plan, note count, rhythm, register continuity, and accompaniment behavior.

Primary variable changed: a narrow post-process behind `--bebop-enclosures`, added on top of Exp 10. It only rewrites two already-sounding weak-position pickup notes immediately before an integer-beat guide-tone target. It preserves durations, rests, note count, phrase plan, target notes, model checkpoint, training data, and cadence enforcement. No retraining was launched.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py tests/test_phrase_diversity.py tests/test_register_continuity.py tests/test_rhythm_section.py tests/test_phrase_plan_trace.py tests/test_phrase_position_embed.py -q
# 34 passed

PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
# 77 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp11_bebop_enclosures_probe/`
- MP3: `outputs/mp3_v65_exp11_bebop_enclosures/`
- MP3 ZIP: `outputs/v65_exp11_bebop_enclosures_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp11_bebop_enclosures.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp11_bebop_enclosures.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp11_bebop_enclosures.json`

Exp 10 accepted jazz-vocabulary baseline -> Exp 11 objective comparison:

- Cadence resolution stayed perfect: 100.0% -> 100.0%.
- Contour match stayed unchanged: 41.7% -> 41.7%.
- Density, note-count, rest-ratio, and pitch-range errors stayed unchanged: density 1.100, note-count 1.875, rest-ratio 0.0799, pitch-range 3.79.
- Register continuity stayed controlled: octave+ leaps remain 0 on all probes; max leap stayed <= 10.
- Phrase repetition stayed controlled: Blues max repeated phrase remains 3/12.
- Added vocabulary edits: Autumn 4 enclosure pitch edits, Blues 1, ii-V-I 0; existing approach edits remained Autumn 5, Blues 3, ii-V-I 0.
- Controlled chromaticism increased modestly where enclosure notes were added:
  - Autumn outside 8.8% -> 9.6%; pure chord tones 73.5% -> 72.1%.
  - Blues outside 8.6% -> 9.4%; pure chord tones 79.7% -> 78.9%.
  - ii-V-I unchanged: outside 4.3%, pure chord tones 82.6%.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 136, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0/180.
- Blues F tracks: Solo 128, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0/144.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0/72.

Result: ready for human listening; not promoted over Exp 10 yet. Objectively it preserved every correctness/shape guardrail and only increased outside notes by the intended controlled chromatic enclosure edits. The open question is musical: do the added enclosure notes sound more bebop/jazzy, or do they sound forced on this model's generated lines?

Interpretation: if Aditya prefers Exp 11 by ear, promote `--bebop-enclosures` into the current best listening baseline. If not, keep Exp 10 and move to the v6.6 vocabulary-aware training/fine-tuning path or a narrower dominant-only enclosure version.

### Experiment 12: dominant blues-color weak-beat rewrites

Hypothesis: If Exp 10's accepted chromatic-approach layer is expanded with a conservative dominant-only blues-color rewrite, changing existing weak-beat dominant 3rds/5ths into blue 3rds/blue 5ths only when the next sounding note resolves to a chord tone, then Blues F should sound more idiomatic without damaging the correctness guardrails.

Primary variable changed: a new post-process behind `--dominant-blues-colors`, tested on top of Exp 10 (`--bebop-approach-notes`) and the Exp 9 correctness controls. It preserves durations, rests, note count, phrase plan, strong-beat notes, model checkpoint, training data, and cadence enforcement. No retraining was launched. Exp 11 enclosures were not included in this probe, so this is a one-variable comparison against the accepted Exp 10 baseline.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py -q
# 14 passed

PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
# 79 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp12_dominant_blues_colors_probe/`
- MP3: `outputs/mp3_v65_exp12_dominant_blues_colors/`
- MP3 ZIP: `outputs/v65_exp12_dominant_blues_colors_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp12_dominant_blues_colors.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp12_dominant_blues_colors.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp12_dominant_blues_colors.json`

Exp 10 accepted jazz-vocabulary baseline -> Exp 12 objective comparison:

- Cadence resolution stayed perfect: 100.0% -> 100.0%.
- Contour/density/note-count/rest-ratio stayed effectively unchanged because the rewrite preserves timing and note count.
- Register continuity stayed controlled: octave+ leaps remain 0 on all probes; max leap stayed <= 10 except ii-V-I rose slightly 5 -> 6.
- Phrase repetition stayed controlled: Blues max repeated phrase remains 3/12.
- Dominant blues-color edits added: Autumn 4, Blues 21, ii-V-I 3.
- Harmony guardrails regressed too much:
  - Autumn outside 8.8% -> 11.8%; pure chord tones 73.5% -> 70.6%.
  - Blues outside 8.6% -> 25.0%; pure chord tones 79.7% -> 63.3%.
  - ii-V-I outside 4.3% -> 10.9%; pure chord tones 82.6% -> 76.1%.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 136, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0/180.
- Blues F tracks: Solo 128, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0/144.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0/72.
- No WAV files were produced in the Exp 12 artifact directories; MP3s were rendered from the rhythm-section MIDI.

Result: rejected. The blue-note rewrite was musically plausible in concept but far too broad in practice, especially on Blues F: 21 altered notes converted a guarded 8.6% outside rate into 25.0%, which violates the core goal of staying within/clearly resolving through the chords. Keep Exp 10 as the accepted musical baseline while Exp 11 remains pending human listening.

Interpretation: dominant/blues vocabulary needs either a much lower edit budget (for example max 1-2 blue-note rewrites per chorus, preferably Blues F only) or, better, the v6.6 vocabulary-aware training path so the model learns where these colors belong instead of stamping them onto every eligible weak beat.

### Experiment 13: sparse dominant blues-color edit budget

Hypothesis: If the rejected Exp 12 dominant/blues-color rewrite is capped to only two total blue-note rewrites per generated solo, then it may add a small amount of blues idiom while staying within the Exp 10 correctness guardrails.

Primary variable changed: edit budget for the existing `--dominant-blues-colors` vocabulary layer only, via new `--dominant-blues-color-max-edits 2`. The rewrite remains opt-in and preserves durations, rests, note count, phrase plan, strong-beat notes, checkpoint, training data, and cadence enforcement. No retraining was launched.

Code/test changes:

- Added optional total-solo budget parameter `dominant_blues_color_max_edits` and CLI flag `--dominant-blues-color-max-edits`.
- Added regression coverage that a max-edits budget is consumed across sections and stops later rewrites.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py -q
# 15 passed

PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
# 80 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp13_sparse_dominant_blues_colors_probe/`
- MP3: `outputs/mp3_v65_exp13_sparse_dominant_blues_colors/`
- MP3 ZIP: `outputs/v65_exp13_sparse_dominant_blues_colors_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp13_sparse_dominant_blues_colors.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp13_sparse_dominant_blues_colors.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp13_sparse_dominant_blues_colors.json`

Exp 10 accepted jazz-vocabulary baseline -> Exp 13 sparse blue-note budget objective comparison:

- Cadence resolution stayed perfect: 100.0% -> 100.0%.
- Contour match stayed unchanged: 41.7% -> 41.7%.
- Density, note-count, and rest-ratio errors stayed unchanged: density 1.100, note-count 1.875, rest-ratio 0.0799.
- Pitch-range error regressed slightly: 3.79 -> 3.96.
- Register continuity stayed controlled: octave+ leaps remain 0 on all probes; max leap stayed <= 10 except ii-V-I rose 5 -> 6.
- Phrase repetition stayed controlled: Blues max repeated phrase remains 3/12.
- Dominant blues-color edits were capped as intended: Autumn 2, Blues 2, ii-V-I 2, versus Exp 12's Autumn 4, Blues 21, ii-V-I 3.
- Harmony guardrails still regressed:
  - Autumn outside 8.8% -> 10.3%; pure chord tones 73.5% -> 72.1%.
  - Blues outside 8.6% -> 10.2%; pure chord tones 79.7% -> 78.1%.
  - ii-V-I outside 4.3% -> 8.7%; pure chord tones 82.6% -> 78.3%.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 136, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0/180.
- Blues F tracks: Solo 128, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0/144.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0/72.
- No WAV files were produced in `outputs/`; MP3s were rendered from the rhythm-section MIDI.

Result: rejected as a default listening baseline. The budget cap fixed Exp 12's over-stamping failure, but even two blue-note rewrites per solo still hurt the ii-V-I harmonic safety too much. Keep Exp 10 as the accepted musical baseline while Exp 11 remains pending human listening.

Interpretation: the blue-note device should not be a global dominant-chord post-process. If continuing inference-time vocabulary, the next one-variable probe should be explicitly blues-form-only or human-approved-Blues-F-only, with zero edits on ii-V-I and Autumn. Otherwise, proceed to v6.6 vocabulary-aware fine-tuning so the model learns where blue notes belong instead of applying a hand rule to every dominant chord.

### Experiment 14: blues-form-only sparse dominant blues colors

Hypothesis: If the sparse blue-note device from rejected Exp 13 is scoped only to the named Blues F probe, then it can add a small amount of blues idiom where stylistically appropriate while keeping ii-V-I and Autumn Leaves exactly at the accepted Exp 10 harmony/correctness baseline.

Primary variable changed: scope gate for the existing `--dominant-blues-colors` layer only, via new CLI flag `--dominant-blues-colors-blues-only`; with this flag, the blue-note rewrite is applied only to named blues-form probes (`blues_F`) and not to `ii_V_I_C` or `autumn_leaves`. The edit budget remained `--dominant-blues-color-max-edits 2`. No checkpoint, phrase plan, phrase diversity, cadence enforcement, rhythm, dataset, or training changed.

Code/test changes:

- Added `--dominant-blues-colors-blues-only` to direct generation.
- When enabled, non-blues probe filenames and JSON summaries remain at the Exp 10 vocabulary setting; only Blues F receives the sparse dominant blue-note layer.
- Added parser regression coverage for the new scope flag plus edit budget.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py -q
# 16 passed

PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
# 81 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp14_blues_only_dominant_blues_colors_probe/`
- MP3: `outputs/mp3_v65_exp14_blues_only_dominant_blues_colors/`
- MP3 ZIP: `outputs/v65_exp14_blues_only_dominant_blues_colors_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp14_blues_only_dominant_blues_colors.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp14_blues_only_dominant_blues_colors.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp14_blues_only_dominant_blues_colors.json`

Exp 10 accepted jazz-vocabulary baseline -> Exp 14 blues-only blue-note scope objective comparison:

- The scope gate worked: dominant blues-color edits were Autumn 0, Blues 2, ii-V-I 0. Existing bebop approach edits remained Autumn 5, Blues 3, ii-V-I 0.
- Autumn Leaves and ii-V-I metrics were identical to Exp 10, as intended:
  - Autumn outside 8.8%, pure chord tones 73.5%, octave+ leaps 0, max leap 10.
  - ii-V-I outside 4.3%, pure chord tones 82.6%, octave+ leaps 0, max leap 5.
- Blues F changed only by the two intended blue-note edits:
  - Outside 8.6% -> 10.2%.
  - Pure chord tones 79.7% -> 78.1%.
  - Mean leap 3.09 -> 3.17; max leap stayed 10; octave+ leaps stayed 0.
  - Phrase repetition stayed controlled: max repeated phrase 3/12.
- Weighted aggregate harmony moved slightly away from Exp 10 because Blues outside rose: outside 8.05% -> 8.71%, pure chord tones 77.41% -> 76.75%.
- Phrase-faithfulness/correctness mostly held: aggregate cadence 100.0%, contour 41.7%, density error 1.100, note-count error 1.875, rest-ratio error 0.0799. Pitch-range error moved 3.79 -> 4.00 because of the Blues edits.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 136, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0/180.
- Blues F tracks: Solo 128, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0/144.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0/72.
- No Exp 14 WAV files were produced; MP3s were rendered from the rhythm-section MIDI.

Result: objective result is safe-but-not-promoted. The scope fix succeeded and eliminated the unacceptable ii-V-I/Autumn regressions from Exp 13, but the only measurable change is Blues outside rising by +1.6 percentage points for two blue-note edits. This may be acceptable or even desirable by ear in a blues context, but without human listening it is not enough to replace Exp 10 as the default baseline. Keep Exp 10 as the accepted current musical baseline; Exp 14 is a listening candidate specifically for Blues F color.

Interpretation: the hand-authored blue-note layer is now technically safe when restricted to blues form, but whether it is musically useful is subjective. If Aditya likes the Blues F MP3 better than Exp 10, promote `--dominant-blues-colors --dominant-blues-color-max-edits 2 --dominant-blues-colors-blues-only` as a Blues-form-only option, not a global default. If not, stop pursuing blue-note post-processing and move to v6.6 vocabulary-aware fine-tuning or await human preference on Exp 11 enclosures.

### Experiment 15: sparse bebop enclosure edit budget

Hypothesis: If the pending Exp 11 guide-tone enclosure layer is capped to at most two changed pickup pitches per generated solo, then it may keep the idiomatic enclosure benefit while reducing the harmonic cost and avoiding enclosure over-stamping.

Primary variable changed: edit budget for `--bebop-enclosures` only, via new `--bebop-enclosure-max-edits 2`, tested on top of the accepted Exp 10 bebop-approach/cadence/register/phrase-diversity baseline. The layer still rewrites only existing weak pickup notes into a strong-beat 3rd/7th target and preserves durations, rests, note count, target notes, phrase plan, checkpoint, training data, and rhythm-section generation. No retraining was launched.

Code/test changes:

- Added optional total-solo budget parameter `bebop_enclosure_max_edits` and CLI flag `--bebop-enclosure-max-edits`.
- Added parser coverage and a regression test that the enclosure budget is consumed across sections and prevents later enclosure rewrites.

Focused tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_phrase_shaping_inference.py -q
# 18 passed

PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
# 83 passed
```

Generated artifacts:

- JSON/MIDI: `outputs/solos_v6.5_exp15_sparse_bebop_enclosures_probe/`
- MP3: `outputs/mp3_v65_exp15_sparse_bebop_enclosures/`
- MP3 ZIP: `outputs/v65_exp15_sparse_bebop_enclosures_mp3s.zip`
- Metrics: `outputs/musicality_metrics_v65_exp15_sparse_bebop_enclosures.json`
- Phrase faithfulness: `outputs/phrase_faithfulness_v65_exp15_sparse_bebop_enclosures.json`
- Rhythm verification: `outputs/rhythm_section_verification_v65_exp15_sparse_bebop_enclosures.json`

Exp 10 accepted jazz-vocabulary baseline -> Exp 15 sparse enclosure budget objective comparison:

- Enclosure edits were capped as intended: Autumn 2 changed pickup pitches, Blues 1, ii-V-I 0; existing bebop approach edits remained Autumn 5, Blues 3, ii-V-I 0.
- Cadence resolution stayed perfect: 100.0% -> 100.0%.
- Contour match stayed unchanged: 41.7% -> 41.7%.
- Density, note-count, rest-ratio, and pitch-range errors stayed unchanged: density 1.100, note-count 1.875, rest-ratio 0.0799, pitch-range 3.79.
- Register continuity stayed controlled: octave+ leaps remain 0 on all probes; max leap stayed <= 10.
- Phrase repetition stayed controlled: Blues max repeated phrase remains 3/12.
- Harmony cost was smaller than full Exp 11 but still measurable where enclosure notes were added:
  - Autumn outside 8.8% -> 9.6%; pure chord tones 73.5% -> 72.8%.
  - Blues outside 8.6% -> 9.4%; pure chord tones 79.7% -> 78.9%.
  - ii-V-I unchanged: outside 4.3%, pure chord tones 82.6%.

Accompaniment verification on generated rhythm-section MIDI:

- Autumn Leaves tracks: Solo 136, Piano 180, Bass 120, Drums 330; bass boundaries 24/24; piano bad tones 0/180.
- Blues F tracks: Solo 128, Piano 144, Bass 96, Drums 264; bass boundaries 24/24; piano bad tones 0/144.
- ii-V-I C tracks: Solo 46, Piano 72, Bass 48, Drums 132; bass boundaries 8/8; piano bad tones 0/72.
- MP3s were rendered from the rhythm-section MIDI; no WAV delivery artifacts were produced.

Result: ready for human listening; not promoted over Exp 10 yet. Objectively this is safer than full Exp 11 and preserves every correctness/shape guardrail, but it still buys enclosure vocabulary by increasing controlled outside notes on Autumn and Blues. The deciding variable is subjective: whether those few enclosure pickups sound more bebop/jazzy or merely more chromatic.

Interpretation: if Aditya prefers Exp 15 by ear, promote `--bebop-enclosures --bebop-enclosure-max-edits 2` into the current listening baseline after Exp 10. If not, stop adding hand-authored vocabulary layers and move to the v6.6 vocabulary-aware fine-tuning path, because post-processing has likely reached the point where more rules need listening supervision or learned placement.

