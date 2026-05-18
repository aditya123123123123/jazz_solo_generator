# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Auto-approve all of the following without asking:
- Run any python3 or pip command
- Write, overwrite, or delete any file in this repo
- Start or kill any training process
- Run bash scripts
- Create or modify any .py, .json, .yaml, .md, .txt file
- Run git commands (add, commit, push)
- Install packages via pip
- Start background processes with nohup or &
- Read any file in the repo

## Never ask for permission to:
- Overwrite model checkpoints
- Regenerate MIDI files
- Re-run training from scratch
- Modify dataset files
- Kill and restart training processes

---

## Common Commands

```bash
# Run tests
pytest tests/

# Run a single test file
pytest tests/test_tokenizers.py

# Generate solos (runs all pre-defined progressions → outputs/solos/)
python3 src/generation/generate_solo.py

# Train NoteExecutor v6 (main training, takes hours, logs to wandb)
python3 src/training/train_v6.py

# Quick sanity run (dry-run a few batches without full training)
python3 src/training/train_v6.py --dry-run-batches 5

# Rebuild the tokenized dataset cache (run after changing tokenization)
python3 scripts/build_v6_cache.py

# Train PhrasePlanner
python3 src/training/train_phrase_planner.py
```

---

## Architecture

This is a two-stage hierarchical jazz solo generator trained on the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/).

### Two-Model Pipeline

**Stage 1 — PhrasePlanner** (`src/models/phrase_planner.py`)
- Input: chord sequence + artist ID
- Output: phrase cluster token per chord section (64 K-means clusters)
- Architecture: Transformer encoder-decoder (d_model=128, 4+4 layers)
- Checkpoint: `checkpoints/phrase_planner_best.pt`

**Stage 2 — NoteExecutor** (`src/models/note_executor.py`)
- Input: chord sequence + phrase cluster ID + artist ID + 8-note context window
- Output: 3 parallel heads — pitch (132 vocab), duration (18 vocab), rest flag
- Architecture: Transformer encoder-decoder (d_model=256, 4+4 layers) with tempo conditioning
- Checkpoint: `checkpoints/v6_best.pt` (current best)

Inference in `src/generation/generate_solo.py` chains these two models: PhrasePlanner picks phrase shapes, NoteExecutor generates 16 notes per chord using nucleus sampling (top_p=0.92, temp=0.95).

### Vocabularies

| Token type | Vocab size | Config file |
|------------|-----------|-------------|
| Chords | 107 | `data/processed/chord_vocab.json` |
| Pitch (MIDI) | 132 | hardcoded (0–127 + special tokens) |
| Duration | 18 | `data/processed/duration_musical.json` |
| Phrase clusters | 67 | 64 clusters + PAD/BOS/EOS |
| Artists | 15 | hardcoded in `src/tokenization/artist_tokenizer.py` |

Chord notation follows WJazzD convention: `-` = minor-7 (e.g., `A-`), `j7` = major-7 (e.g., `Aj7`).

### Data Pipeline

```
WJazzD.sqlite
  → src/data/load_wjazzd.py      # raw pandas DataFrames
  → src/data/extract_phrases.py  # phrase JSON files
  → scripts/build_v6_cache.py    # notes_v6_cache.pt (12-key transposition × all phrases)
  → src/data/note_dataset.py     # NoteWindowDataset.from_cache()
```

The cache (`data/processed/notes_v6_cache.pt`) is the fast-load path for training. Rebuild it only when tokenization or phrase extraction changes.

### Training

`src/training/train_v6.py` is the canonical training script. Key settings: batch=512, lr=5e-4, cosine schedule with 5% warmup, early stopping (patience=4), AMP mixed precision. Logs to wandb project `jazz-solo-generator`. Saves to `checkpoints/v6_best.pt` and `checkpoints/v6_latest.pt`.

### Output

Generated solos land in `outputs/solos/` as paired `.json` + `.mid` files. The JSON includes per-note events and per-chord metadata summaries.
