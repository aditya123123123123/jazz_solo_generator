# Jazz Solo Generator — Project Context

## What This Is
A phrasing-aware hierarchical jazz solo generator trained on the Weimar Jazz Database.
Target artists: Charlie Parker and Miles Davis.

## Architecture
- Level 1 (Phrase Planner): Transformer that takes chord changes + artist token → sequence of phrase-type tokens
- Level 2 (Note Executor): Seq2Seq Transformer that takes chord tokens + phrase token + artist token → note sequence
- Training data: WJazzD SQLite database in data/raw/wjazzd/

## Repo Structure
- src/data/ — data loading and phrase extraction
- src/features/ — phrase feature computation and clustering
- src/tokenization/ — chord, note, phrase, artist tokenizers
- src/models/ — Phrase Planner and Note Executor
- src/training/ — training loops for both models
- src/inference/ — end-to-end generation pipeline
- src/evaluation/ — metrics and evaluation tools
- data/raw/wjazzd/ — WJazt in git)
- data/processed/ — generated datasets and tokenizer files
- checkpoints/ — saved model weights

## Key Rules
- Always use Plan Mode before touching more than one file
- Never modify existing working files without showing a plan first
- Keep each module independently testable
- Add shape printouts after every major tensor operation in PyTorch
- Commit after every milestone

## Tech Stack
Python, PyTorch, pandas, music21, pretty_midi, scikit-learn, numpy
