"""Build the pretokenized v6 cache for NoteWindowDataset.

Drives the slow JSON→tensor build path once, then converts the resulting
list of per-sample dicts into a single torch.save dict so that
NoteWindowDataset.from_cache() can skip retokenization on every epoch.

Run once after Step 3 (note_tokenizer rewrite) + Step 2 (tempo backfill):

    python3 scripts/build_v6_cache.py
"""
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

# Allow running as `python3 scripts/build_v6_cache.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.note_dataset import NoteWindowDataset, TRANSPOSE_SEMITONES, WINDOW
from src.tokenization import (
    ArtistTokenizer, ChordTokenizer, NoteTokenizer, PhraseTokenizer,
)
from src.tokenization.note_tokenizer import DUR_OFFSET, MUSICAL_DURATIONS_BEATS

_REPO       = Path(__file__).resolve().parents[1]
CACHE_PATH  = _REPO / "data" / "processed" / "notes_v6_cache.pt"
PHRASES_REL = "phrases_all_with_tempo.json"
CLUSTERS_REL = "phrase_clusters_all.csv"


def main():
    chord_tok  = ChordTokenizer.from_json()
    note_tok   = NoteTokenizer.from_json()
    artist_tok = ArtistTokenizer()
    phrase_tok = PhraseTokenizer()

    print("Building NoteWindowDataset (slow JSON path) …")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = NoteWindowDataset(chord_tok, note_tok, artist_tok, phrase_tok)
    print(f"  built {len(ds):,} samples in {time.time() - t0:.1f}s")

    samples = ds._samples
    N = len(samples)

    print("Stacking fixed-shape tensors …")
    t1 = time.time()
    cache = {
        "ctx_pitch":     torch.stack([s["ctx_pitch"]     for s in samples]),
        "ctx_dur":       torch.stack([s["ctx_dur"]       for s in samples]),
        "ctx_rest":      torch.stack([s["ctx_rest"]      for s in samples]),
        "target_pitch":  torch.stack([s["target_pitch"]  for s in samples]),
        "target_dur":    torch.stack([s["target_dur"]    for s in samples]),
        "target_rest":   torch.stack([s["target_rest"]   for s in samples]),
        "phrase_id":     torch.stack([s["phrase_id"]     for s in samples]),
        "artist_id":     torch.stack([s["artist_id"]     for s in samples]),
        "pos_in_phrase": torch.stack([s["pos_in_phrase"] for s in samples]),
        "tempo_bpm":     torch.stack([s["tempo_bpm"]     for s in samples]),
    }
    print(f"  stacked in {time.time() - t1:.1f}s")

    print("Building CSR chord_ids …")
    t2 = time.time()
    chord_lens = torch.tensor([s["chord_ids"].numel() for s in samples], dtype=torch.long)
    chord_offsets = torch.zeros(N + 1, dtype=torch.long)
    chord_offsets[1:] = torch.cumsum(chord_lens, dim=0)
    chord_ids_flat = torch.cat([s["chord_ids"] for s in samples])
    cache["chord_ids_flat"] = chord_ids_flat
    cache["chord_offsets"]  = chord_offsets
    cache["chord_len"]      = chord_lens
    print(f"  CSR ready in {time.time() - t2:.1f}s "
          f"(chord_ids_flat: {chord_ids_flat.numel():,} entries)")

    cache["meta"] = {
        "vocab_version":       "v6",
        "n_samples":           N,
        "window_size":         WINDOW,
        "transpose_semitones": list(TRANSPOSE_SEMITONES),
        "phrases_path":        PHRASES_REL,
        "clusters_path":       CLUSTERS_REL,
    }

    print(f"Saving to {CACHE_PATH} …")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE_PATH)

    size_mb = CACHE_PATH.stat().st_size / 1e6
    print(f"  wrote {size_mb:.1f} MB")

    # ---- duration histogram on target_dur tokens ----
    print("\nDuration token histogram (target_dur):")
    hist = Counter(cache["target_dur"].tolist())
    print(f"  {'tok':>3}  {'beats':>7}  {'count':>10}  {'pct':>6}  bar")
    for tok in sorted(hist.keys()):
        idx   = tok - DUR_OFFSET
        beats = MUSICAL_DURATIONS_BEATS[idx] if 0 <= idx < 14 else float("nan")
        cnt   = hist[tok]
        pct   = 100 * cnt / N
        bar   = "#" * int(pct / 2)
        print(f"  {tok:>3}  {beats:>7.3f}  {cnt:>10,}  {pct:5.2f}%  {bar}")

    print(f"\nDone. {N:,} samples → {size_mb:.1f} MB → {CACHE_PATH}")


if __name__ == "__main__":
    main()
