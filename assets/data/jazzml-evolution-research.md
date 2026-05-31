# JazzMLProject Evolution Research Brief

Purpose: deep background for a future website / visual representation of how JazzMLProject evolved. This is a synthesis of git history, repo docs, audition artifacts, and prior session decisions. It is intentionally narrative + technical, not just a changelog.

## Core thesis

JazzMLProject evolved from a working symbolic jazz solo generator into an increasingly evidence-driven research project about musicality. The arc is not simply “newer checkpoint = better.” The real story is:

1. Build a hierarchical phrase-planner + note-executor jazz generator.
2. Discover that the v6 model was technically generating notes but had partly lost meaningful conditioning.
3. Restore harmonic/phrase sensitivity.
4. Add phrase-aware planning and evaluation.
5. Run one-variable musicality experiments to fix register, chord safety, repetition, cadence, and contour.
6. Realize correctness is not the same as jazz.
7. Add controlled bebop vocabulary; Exp 10 becomes the first clearly accepted “more jazzy” direction.
8. Try to teach vocabulary through fine-tuning; v6.6 becomes safer but less colorful.
9. Move to v6.8/v6.8.1 with broader auditions and inference controls as the current promising line.

## Architecture baseline

The project is a two-stage symbolic jazz solo generator trained on Weimar Jazz Database material.

- Stage 1: PhrasePlanner
  - Transformer encoder-decoder.
  - Input: chord sequence + artist ID.
  - Output: phrase-cluster tokens.
  - Phrase clusters: 64 K-means clusters plus special tokens.

- Stage 2: NoteExecutor
  - Transformer encoder-decoder.
  - Input: chord context, phrase cluster, artist, note history, tempo/position features.
  - Output heads: pitch, duration, rest flag.

Important supporting pieces:
- chord, duration, phrase, and artist tokenizers
- v6 pretokenized cache
- harmonic auxiliary loss
- phrase-position embedding
- interval regularization
- rhythm section generation
- JSON/MIDI/audio audition pipeline
- musicality / phrase-faithfulness metrics

## Timeline with anchors

### 1. Project bootstrap and phrase-data pipeline

Key commits:
- `ec48b846` — Add README
- `9bf62400` — Phase 1 phrase extraction
- `d11dd30b` — Phase 2 phrase features
- `cd4cb999` — Phase 3 phrase clustering
- `242ec1c5` — Phase 4 tokenizers
- `375920a3` — Phase 5 dataset/collate/stub training

Narrative:
The project first built the symbolic data pipeline: extract phrases from WJazzD, compute features, cluster them, tokenize chords/pitches/durations/artists/phrases, and validate dataset shapes.

Website representation:
- “From corpus to phrase tokens” pipeline graphic.
- Cluster map showing Parker/Miles stylistic separation.

### 2. PhrasePlanner and NoteExecutor become a real hierarchy

Key commits:
- `961a4949` — PhrasePlanner introduced
- `b31d2681` — expanded to 15 artists
- `eaad0ce1` — reclustering over 3,693 phrases
- `a00cabcf` — NoteExecutor + full note dataset
- `8ff14512` — working end-to-end jazz solo pipeline
- `99b26c00` — v4 cross-chord context + harmonic loss

Narrative:
This is when the project became a real hierarchical generator rather than just data processing. PhrasePlanner predicted phrase categories; NoteExecutor rendered them into notes.

Website representation:
- Two-stage model diagram.
- “Phrase intention → note realization” animation.

### 3. v5/v6 data-engineering and tokenizer redesign

Key commits:
- `affd06dc` — v5 with 12-key augmentation and rhythm helpers
- `3d4de863` — v5b training script; later important because it contained harmonic auxiliary loss
- `d9e30e47` — musical-duration tokenization
- `0365c41e` — v6 pretokenized cache
- `93827e1e` — v6 NoteExecutor update with tempo conditioning
- `847866ab` — tiny-overfit test
- `7e48704a` — canonical v6 training loop

Narrative:
v6 made the system faster and more musically structured, with beat-relative duration tokens and a large cache. But the new v6 training loop accidentally dropped important harmonic-loss behavior from v5b, which later caused a major diagnosis.

Website representation:
- “The fast cache unlocked experiments, but hid a regression” section.

### 4. Conditioning collapse diagnosed and fixed

Key docs:
- `experiments/2026-05-18-conditioning-ablation.md`
- `experiments/2026-05-20-v6.1.0-harmonic-loss-finetune.md`

Key commits:
- `ec69342c` — conditioning ablation diagnosis
- `4150ed42` — restore harmonic loss in v6 training
- `30cdec74` — v6.1.0 finetune results

Finding:
The v6 model was not strongly using phrase/chord/artist conditioning. Ablations showed chord-shuffled and phrase-shuffled inputs were too close to baseline; in some cases zeroing conditioning could even reduce CE.

Root cause:
`train_v6.py` had been rewritten and omitted the older v5b harmonic auxiliary loss.

Result:
v6.1.0 restored harmonic loss and recovered chord/phrase sensitivity. Artist conditioning remained weak.

Website representation:
- Before/after ablation table.
- “The model was playing, but not listening” section.

### 5. v6.2/v6.3 harmonic and melodic control

Key commits/docs:
- `a8803113` — phrase-position embedding
- `d57179b3` — batches provide phrase_position
- `bb6bcd81`, `dc2ad507`, `84010da7` — interval penalty and calibration
- `ed7444ad` — v6.3 stronger harmonic weight + interval curriculum
- `180e591a` — chord-aware evaluation / multiseed / decoder bias infra
- `outputs/solo_eval_report.md`
- `outputs/solo_eval_multiseed_report.md`
- `outputs/v631_decoder_bias_grid.md`
- `outputs/v632_allbeat_musical_audit.md`

Narrative:
This phase focused on making generated solos fit chord changes and move melodically without ugly leaps. The raw model improved, but decisive harmonic gains came from inference-time chord steering.

Important subjective framing:
- Chord-correct outputs risked sounding “correct but stiff / exercise-like.”
- The model was learning chord correctness more than jazz language.

Website representation:
- Safety metrics over versions.
- Chord-tone vs chord+approach charts.

### 6. v6.4 phrase-aware generation

Key doc:
- `docs/plans/2026-05-24-v64-phrase-aware-generation.md`

Key commit:
- `0af99db9` — phrase-aware generation probe

Changes:
- Whole-progression phrase planning instead of per-chord planning.
- Phrase-plan tracing in JSON.
- Phrase-faithfulness metrics.
- `--phrase-shaping` inference control.

Outcome:
- Contour match improved significantly.
- Note-count error improved.
- Strict chord-tone improved slightly.
- Main issue: register/pitch-range behavior worsened; phrase-shaped but jumpy.

Website representation:
- Before/after phrase contour example.
- Phrase plan overlay on generated solo.

### 7. v6.5 scientific musicality iteration

Primary source:
- `plans/scientific-musicality-iteration-v6.5.md`

Anchor commit:
- `50b5c535` — v6.5 musicality experiments

This is the core narrative section. It is a sequence of one-variable experiments with hypotheses, artifacts, metrics, and accept/reject decisions.

#### Exp 1 — Register continuity
Accepted.
- Fixed octave teleports.
- Kept pitch-class harmony intact.

#### Exp 2 — All-position chord-tone bias
Accepted.
- Large outside-note reduction.
- Moved toward harmonic safety.

#### Exp 3 / 3b — Phrase diversity
Exp 3b max-uses=3 accepted.
- Reduced phrase stamping/repetition.
- Better than stricter max-uses=2.

#### Exp 4 — Rhythm density/rest calibration
Rejected.
- Helped some note-count metrics but hurt density/rest feel.
- Kept separate export/rhythm-section fixes.

#### Exp 5 — Lower duration temperature
Rejected.
- Small density win, worse cadence/rest/range.

#### Exp 6 — Cadence-preserving phrase diversity proxy
Rejected.
- Proxy too coarse; cadence did not actually preserve reliably.

#### Exp 7 — Per-section cadence enforcement
Not promoted.
- Cadence improved to 100%.
- Contour regressed.

#### Exp 8 — Contour-aware section cadence
Human-approved provisional baseline.
- Objective contour gate still failed.
- But listening/human judgment preferred the direction because cadence/harmony gains mattered.

Important website moment:
Objective best and musically preferred best diverged.

#### Exp 9 — Target-contour-aware cadence
Best correctness / musical-shape baseline.
- Cadence 100%.
- Contour recovered.
- Register/repetition/harmony safe.

Important judgment:
“Correctness is not the same thing as jazz.”

#### Exp 10 — Weak-beat bebop approach notes
Accepted musical baseline.
- Added controlled chromatic approach notes on weak beats into strong-beat chord tones.
- Preserved cadence, contour, register, rhythm section behavior.
- User judgment: “definitely a lot better.”

Important website moment:
This was the pivot from making the model correct to making it sound jazzy.

#### Exp 11 / 15 / 16 — Enclosure variants
Listening candidates, not promoted.
- Technically safe.
- Open question: more bebop/jazzy or forced?

#### Exp 12 / 13 / 14 — Blues-color variants
Mostly rejected or not promoted.
- Global blue-note rewrite over-stamped and hurt harmony.
- Sparse / blues-only variants were safer but not enough to replace Exp 10.

Website representation:
- Experiment timeline with accepted/rejected/listening-candidate tags.
- Safety-vs-jazziness plot.
- Audio comparison: Exp 9 vs Exp 10.

### 8. Training-set vocabulary audit and v6.6

Key commits:
- `7e5308f5` — training corpus jazz vocabulary audit
- `a7e5cdb8` — vocabulary-aware training labels

Key files:
- `src/data/jazz_vocab_labels.py`
- `scripts/build_v6_cache.py`
- `src/training/train_v6.py`
- `outputs/jazz_vocabulary_audit_v1.md`
- `outputs/jazz_vocabulary_audit_v1.json`

Rationale:
Exp 10 proved jazz vocabulary helped. The next question was whether the model could learn vocabulary placement instead of relying on more hand-authored post-processing.

v6.6 / Exp 17 setup:
- Fine-tune from `checkpoints/v6.3.2_best.pt`.
- Expanded cache oversampled jazz-vocabulary positives 2x.
- Expanded cache: 2,679,684 samples.
- Jazz-vocab positives: 925,872 / 34.55%.
- Best checkpoint: `checkpoints/v6.6_expanded_cache_x2_best.pt`.

Interim findings:
- Epoch 1: safer but bland / weak phrase behavior.
- Epoch 8: better than epoch 1; still raised safety-vs-color concerns.
- v6.6+Exp10 initially exposed a late-postprocess register safety bug.

Final v6.6 evaluation:
- Review: `outputs/_LISTENING_v6.6_final_review/REVIEW.md`
- Metric comparison: `outputs/musicality_metrics_v66_exp17_final_compare.json`
- Listening bundle: `outputs/_LISTENING_v6.6_final_review/`

Bug fix:
- Cadence/vocabulary edits could happen after register continuity and reintroduce octave jumps.
- Fix: if register continuity is enabled, re-run section register continuity after cadence/vocabulary edits.

Verdict:
Do not promote v6.6.
- It improved harmonic safety.
- But color-tone usage dropped.
- It sounded/profiled like “less wrong” rather than genuinely more jazzy.

Website representation:
- Exp 10 vs v6.6 raw vs v6.6+Exp10 audio triptych.
- “Inside is not always better” section.

### 9. v6.8 / v6.8.1 current candidate line

Key commits:
- `fc50291c` — v6.8 harmonic audition artifacts
- `1e7354a5` — v6.8.1 inference controls
- `76fdaad2` — expanded v6.8 audition song set; current `origin/main` HEAD at research time

Key artifacts:
- `outputs/auditions_v68_compare/AUDITION_COMPARISON.md`
- `outputs/auditions_v68_fixed_harmonic/FIX_VARIANT_COMPARISON.md`
- `outputs/auditions_v681_selected/V681_GRID_COMPARISON.md`
- `outputs/auditions_v68_expanded_songs_selected/V68_EXPANDED_SONG_COMPARISON.md`
- `outputs/v68_expanded_songs_audition.zip`

Initial v6.8 comparison:
- More vocabulary/color direction than v6.6, but raw harmonic safety was weaker.
- Aggregate v6.8 on 3 probes:
  - chord-tone around 0.612
  - chord+approach around 0.881

v6.8.1 inference controls:
- density calibration threshold
- register resample cap
- best-attempt fallback
- repeat guard
- ii-V-I bias/grid controls

Expanded 8-song audition:
- all_the_things_you_are
- autumn_leaves
- blues_F
- giant_steps_cycle
- ii_V_I_C
- minor_blues_C
- modal_so_what_Dm
- rhythm_changes_Bb

Selected aggregate:
- 8 tunes
- 1283 notes
- chord-tone 0.754
- chord+approach 0.935
- density 3.80
- avg interval 3.17
- big leap 0.007
- repeat 0.019

Current interpretation:
v6.8/v6.8.1 is the current most promising line, but still needs listening as the final gate. It should not be represented as “solved” purely by metrics.

## Durable subjective judgments

These are website-worthy phrases or section titles:

- “The model was playing, but not listening.”
- “Correct but stiff / exercise-like.”
- “More like an etude over chord changes than a real improvised line.”
- “The model learned chord correctness more than jazz language.”
- “Objective best and musical best diverged.”
- “Correctness is not the same thing as jazz.”
- “Exp 10 stopped optimizing correctness and started adding actual vocabulary.”
- “Inside is not always better.”
- “v6.6 was less wrong, not more genuinely jazzy.”

## Best assets for a future website

### Hero docs
- `experiments/2026-05-18-conditioning-ablation.md`
- `experiments/2026-05-20-v6.1.0-harmonic-loss-finetune.md`
- `docs/plans/2026-05-24-v64-phrase-aware-generation.md`
- `plans/scientific-musicality-iteration-v6.5.md`
- `outputs/_LISTENING_v6.6_final_review/REVIEW.md`
- `outputs/auditions_v68_expanded_songs_selected/V68_EXPANDED_SONG_COMPARISON.md`

### Audio / listening assets
- `outputs/mp3_v65_exp10_bebop_approach/`
- `outputs/_LISTENING_v6.6_final_review/`
- `outputs/auditions_v68_expanded_songs_selected/audio_wav/`

### Metric files
- `outputs/musicality_metrics_v65_exp10_bebop_approach.json`
- `outputs/musicality_metrics_v66_exp17_final_compare.json`
- `outputs/jazz_vocabulary_audit_v1.json`
- `outputs/auditions_v681_selected/V681_GRID_COMPARISON.md`
- `outputs/auditions_v68_expanded_songs_selected/V68_EXPANDED_SONG_COMPARISON.md`

## Recommended website structure

1. Hero: “Teaching an AI to improvise jazz — and learning that correctness is not jazz.”
2. Architecture: PhrasePlanner + NoteExecutor.
3. The diagnosis: conditioning collapse.
4. The repair: harmonic-loss restoration.
5. Phrase-awareness: planning whole solos instead of isolated sections.
6. Scientific iteration: v6.5 experiment ladder.
7. The first musical breakthrough: Exp 10 bebop approach notes.
8. The safety-vs-color trap: v6.6.
9. The current candidate: v6.8/v6.8.1 expanded auditions.
10. Interactive listening lab: compare baselines and candidates.
11. Metrics dashboard: cadence, contour, chord-tone, color, density, repetition, leaps.
12. What’s next: broader listening, curated training, possible v7/event-token model.

## Open gaps to resolve before building the website

1. v6.6/v6.7/v6.8 checkpoint provenance
   - Git history has audition artifacts, but not a perfectly clean “this training command produced this checkpoint” chain for every checkpoint.
   - Need a checkpoint lineage table: checkpoint path, training command, source checkpoint, dataset/cache, commit, result.

2. Ear-based decisions for v6.8/v6.8.1
   - Metrics look promising, but the final website should not claim v6.8.1 is best until listening confirms it.

3. Earlier v1–v5 narrative depth
   - We have enough for technical history, but the public story can probably compress this era unless the user wants a full archaeology section.

4. Current local/remote repo divergence
   - Local repo is behind `origin/main` and has dirty v6.6-era modifications.
   - Before building in-repo website files, reconcile or isolate work carefully.

## Current repo state at research time

Local path:
- `/Users/openclaw/Documents/JazzMLProject/jazz_solo_generator`

Important caveat:
- Local branch was behind `origin/main` by 4 commits when checked.
- Local tree had modified v6.6-era files and untracked expanded-cache script/test files.
- Some latest v6.8 artifacts were read from `origin/main` via git, not necessarily checked into the local working tree.
