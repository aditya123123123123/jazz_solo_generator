# v6.4 Phrase-Aware Generation Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build v6.4 so generated solos remain chord-correct while becoming measurably phrase-aware: phrase clusters control contour, density, register, rest placement, and cadence behavior across multi-bar phrases.

**Architecture:** Do this in two phases. First add diagnostics and inference-time phrase controls that do not require retraining, so we can prove the current model's phrase conditioning weakness. Then add phrase-feature conditioning targets to NoteExecutor and fine-tune/retrain v6.4 so the executor obeys PhrasePlanner clusters instead of only spelling chord tones.

**Tech Stack:** Python, PyTorch, pretty_midi, existing WJazzD phrase JSON, existing `phrase_clusters_all.csv`, current `PhrasePlanner`, current `NoteExecutor`, pytest.

---

## Key finding from current inspection

Current generation is not truly phrase-level yet:

- `generate_solo.py` calls `_plan_phrase()` separately for each chord section with a one-chord tensor, so PhrasePlanner is not planning a whole solo/chorus shape.
- `NoteExecutor.generate()` always emits `N_NOTES = 16` per chord section, so phrase length/density from the cluster is mostly ignored.
- Phrase cluster metadata already exists in `data/processed/phrase_clusters_all.csv`: `length_beats`, `num_notes`, `note_density`, `pitch_mean`, `pitch_range`, `pitch_std`, `rest_ratio`, `contour`, `starts_on_downbeat`, `phrase_duration`, `phrase_token`.
- NoteExecutor receives `phrase_id`, `pos_in_phrase`, and context positions, but not the interpretable cluster features directly.

Conclusion: diagnostics and inference controls can start immediately without retraining, but strong phrase awareness will require at least fine-tuning NoteExecutor, and probably retraining/fine-tuning PhrasePlanner after changing generation to plan multi-bar phrase units.

---

### Task 1: Add phrase-cluster statistics loader

**Objective:** Make phrase-token features available to generation and evaluation.

**Files:**
- Create: `src/generation/phrase_features.py`
- Test: `tests/test_phrase_features.py`

**Implementation notes:**

Create a loader that reads `data/processed/phrase_clusters_all.csv` and returns aggregated stats per `PHRASE_XX`:

- median `length_beats`
- median `num_notes`
- median `note_density`
- median `pitch_mean`
- median `pitch_range`
- median `rest_ratio`
- modal `contour`
- modal `starts_on_downbeat`

Expose:

```python
@dataclass(frozen=True)
class PhraseFeature:
    token: str
    median_length_beats: float
    median_num_notes: int
    median_density: float
    median_pitch_mean: float
    median_pitch_range: float
    median_rest_ratio: float
    contour: str
    starts_on_downbeat: bool


def load_phrase_features(path: Path | None = None) -> dict[str, PhraseFeature]: ...
```

**Verification:**

Run:

```bash
python3 -m pytest -q tests/test_phrase_features.py
```

Expected: tests pass and `PHRASE_00` plus all 64 cluster tokens load.

---

### Task 2: Add phrase-plan tracing to generation JSON

**Objective:** Make every generated output explain what phrase token was selected and what that token means.

**Files:**
- Modify: `src/generation/generate_solo.py`
- Test: `tests/test_phrase_plan_trace.py`

**Implementation notes:**

Add per-section fields to generated JSON:

```json
{
  "phrase": "PHRASE_43",
  "phrase_features": {
    "contour": "ascending",
    "median_num_notes": 28,
    "median_density": 2.9,
    "median_rest_ratio": 0.19,
    "median_pitch_range": 12
  },
  "generated_phrase_metrics": {
    "contour": "descending",
    "num_notes": 16,
    "density": 3.4,
    "rest_ratio": 0.31,
    "pitch_range": 15
  }
}
```

**Verification:**

Run:

```bash
python3 -m pytest -q tests/test_phrase_plan_trace.py
```

Expected: JSON exports include phrase features and generated metrics for each section.

---

### Task 3: Add phrase-faithfulness evaluator

**Objective:** Measure whether generated sections actually match their assigned phrase clusters.

**Files:**
- Create: `scripts/evaluate_phrase_faithfulness.py`
- Test: `tests/test_phrase_faithfulness.py`

**Metrics:**

Per generated section:

- contour match: generated contour equals modal cluster contour
- density z-score/error vs cluster median
- note-count error vs cluster median
- pitch-range error vs cluster median
- rest-ratio error vs cluster median
- cadence resolution: final sounding note is legal chord tone

Output:

- `outputs/phrase_faithfulness_v632.md`
- `outputs/phrase_faithfulness_v632.json`

**Verification:**

Run:

```bash
python3 -m pytest -q tests/test_phrase_faithfulness.py
.venv/bin/python scripts/evaluate_phrase_faithfulness.py --solos-dir outputs/solos_v6.3.2_allbeat_eval
```

Expected: report gives per-solo and aggregate phrase-faithfulness scores.

---

### Task 4: Change generation from per-chord phrase planning to whole-progression phrase planning

**Objective:** Let PhrasePlanner produce a sequence over the full progression instead of asking it separately for each chord.

**Files:**
- Modify: `src/generation/generate_solo.py`
- Test: `tests/test_whole_progression_phrase_planning.py`

**Implementation notes:**

Current issue:

```python
chord_ids = torch.tensor([[model_chord_id]])
phrase_int_raw = _plan_phrase(planner, chord_ids, artist_id, temperature=temperature)
```

Replace with:

1. Build `full_chord_ids = torch.tensor([model_chord_ids])` once.
2. Call planner once: `phrase_plan = planner.generate(full_chord_ids, artist_id, max_len=len(progression) + 4, ...)`.
3. Assign phrase tokens to sections from that sequence.
4. If planner returns fewer tokens than sections, fill by repeating/plausible fallback.
5. Write the full phrase plan to JSON top-level metadata.

**Verification:**

Run:

```bash
python3 -m pytest -q tests/test_whole_progression_phrase_planning.py
```

Expected: planner called once per tune, output JSON includes `phrase_plan`, and sections use successive tokens from that plan.

---

### Task 5: Add inference-time phrase shaping without retraining

**Objective:** Use cluster features to shape current outputs while preserving chord correctness.

**Files:**
- Modify: `src/generation/generate_solo.py`
- Test: `tests/test_phrase_shaping_inference.py`

**Controls:**

- `n_notes` per section should come from cluster median note count, clamped to a safe range like 8–28.
- `rest_boost` should be adjusted from cluster median rest ratio.
- phrase start/end register should gently follow cluster `contour`:
  - ascending: discourage early high notes and encourage later higher register
  - descending: reverse
  - arch: low/high/low register envelope
- keep chord-tone masking/boosting active.

Add CLI flag:

```bash
--phrase-shaping
```

Keep default behavior unchanged until flag is enabled.

**Verification:**

Run:

```bash
python3 -m pytest -q tests/test_phrase_shaping_inference.py
.venv/bin/python -m src.generation.generate_solo --phrase-shaping --chord-tone-bias --all-beat-chord-tone-bias --out-dir outputs/solos_v6.4_phrase_shaped_probe
.venv/bin/python scripts/evaluate_phrase_faithfulness.py --solos-dir outputs/solos_v6.4_phrase_shaped_probe
```

Expected: phrase-faithfulness improves over v6.3.2 without a large drop in chord-tone legality.

---

### Task 6: Add phrase-conditioning ablation gate

**Objective:** Prove NoteExecutor actually uses phrase IDs.

**Files:**
- Modify: `src/eval/conditioning_ablation.py`
- Create or modify: `tests/test_phrase_conditioning_gate.py`

**Implementation notes:**

The current ablation already supports `phrase_zeroed` and `phrase_shuffled`, but the old expected ranges show phrase conditioning is weak or even weirdly inverted. Add a v6.4 gate:

- baseline pitch/duration/rest CE must be better than phrase_zeroed
- phrase_shuffled must hurt generated phrase-faithfulness
- report deltas in markdown

**Verification:**

Run:

```bash
.venv/bin/python -m src.eval.conditioning_ablation --checkpoint checkpoints/v6.3.2_best.pt --out-dir outputs/conditioning_ablation_v632
```

Expected: v6.3.2 shows weak phrase effect; this becomes the baseline that v6.4 must beat.

---

### Task 7: Add explicit phrase-feature conditioning to NoteExecutor

**Objective:** Give NoteExecutor interpretable phrase targets, not just an opaque `phrase_id` embedding.

**Files:**
- Modify: `src/models/note_executor.py`
- Modify: `src/data/note_dataset.py`
- Modify: `scripts/build_v6_cache.py`
- Modify: `src/training/train_v6.py`
- Tests:
  - `tests/test_note_dataset_phrase_features.py`
  - `tests/test_note_executor_phrase_features.py`

**Implementation notes:**

Add numeric/categorical phrase features to every NoteWindowDataset sample:

- `phrase_length_beats`
- `phrase_num_notes`
- `phrase_density`
- `phrase_pitch_mean`
- `phrase_pitch_range`
- `phrase_rest_ratio`
- `phrase_contour_id`

In `NoteExecutor`, embed them with a small MLP:

```python
self.phrase_feature_embed = nn.Sequential(
    nn.Linear(NUM_PHRASE_FEATURES, d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model),
)
```

Add the resulting bias alongside `phrase_embed(phrase_id)`.

**Retraining requirement:** yes. This changes model inputs/cache/checkpoint shape, so v6.4 needs a rebuilt cache and at least a fine-tune. Full retrain may be better but start with fine-tune from v6.3.2 if compatible.

**Verification:**

Run:

```bash
python3 -m pytest -q tests/test_note_dataset_phrase_features.py tests/test_note_executor_phrase_features.py
.venv/bin/python scripts/build_v6_cache.py --output data/processed/notes_v6_phrase_cache.pt
.venv/bin/python -m src.training.train_v6 --cache data/processed/notes_v6_phrase_cache.pt --dry-run-batches 5 --run-name note-executor-v6.4-phrase-aware-dryrun
```

Expected: dry run completes and logs phrase-feature tensors.

---

### Task 8: Fine-tune v6.4 NoteExecutor with phrase-feature conditioning

**Objective:** Train the executor to obey phrase-level features while retaining chord correctness.

**Files:**
- Modify: `scripts/supervise_chord_improvement.sh` or add `scripts/supervise_phrase_aware_v64.sh`

**Command shape:**

```bash
.venv/bin/python -m src.training.train_v6 \
  --cache data/processed/notes_v6_phrase_cache.pt \
  --resume checkpoints/v6.3.2_best.pt \
  --epochs 8 \
  --lr 1e-4 \
  --run-name note-executor-v6.4-phrase-aware \
  --best-output checkpoints/v6.4_phrase_aware_best.pt \
  --latest-output checkpoints/v6.4_phrase_aware_latest.pt \
  --harmonic-weight 0.30
```

**Acceptance gates:**

- chord-tone legality remains high: strict chord-tone >= current v6.3.2 or within agreed tolerance
- chord-or-approach >= 0.95 for blues, >= 0.90 for Autumn Leaves/ii-V-I
- phrase-faithfulness improves over v6.3.2
- phrase_zeroed and phrase_shuffled ablations are worse than baseline

---

### Task 9: Generate v6.4 listening pack

**Objective:** Produce files for human listening and side-by-side comparison.

**Files:**
- Outputs only under `outputs/solos_v6.4_phrase_aware/`

**Commands:**

```bash
.venv/bin/python -m src.generation.generate_solo \
  --note-executor-checkpoint checkpoints/v6.4_phrase_aware_best.pt \
  --phrase-shaping \
  --chord-tone-bias \
  --all-beat-chord-tone-bias \
  --chord-tone-bias-strength 1.6 \
  --non-chord-penalty 0.5 \
  --out-dir outputs/solos_v6.4_phrase_aware

.venv/bin/python scripts/add_rhythm_section_to_existing_outputs.py \
  --src-dir outputs/solos_v6.4_phrase_aware \
  --out-dir outputs/solos_v6.4_phrase_aware_with_rhythm

.venv/bin/python scripts/evaluate_phrase_faithfulness.py \
  --solos-dir outputs/solos_v6.4_phrase_aware
```

**Verification:**

Expected output includes MIDI, JSON, rhythm-section MIDI, and phrase-faithfulness report.

---

## Retraining answer

Partial answer: not immediately, but yes for the real version.

- No retraining required for diagnostics, whole-progression phrase planning, JSON tracing, evaluation, and inference-time phrase shaping.
- Fine-tuning/retraining is required once we add explicit phrase-feature conditioning to NoteExecutor, because that changes model inputs and the training cache.
- PhrasePlanner may not need architectural retraining if we only call it once per progression, but if the planned tokens are low-quality or not structurally coherent, then the PhrasePlanner should also be retrained/fine-tuned with stronger sequence-level objectives.

## Recommended execution order

1. Implement Tasks 1–6 locally first. This will show how much phrase awareness we can recover without training.
2. Generate a v6.4 phrase-shaped probe listening pack.
3. If the probe is still too chord-spelling/section-local, implement Tasks 7–8 and fine-tune on the remote GPU PC.
4. Compare v6.3.2 vs v6.4 with both metrics and listening.
