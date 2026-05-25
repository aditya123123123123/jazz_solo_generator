"""
NoteWindowDataset with 12-key transposition augmentation.

For every solo, 12 transposed copies are generated (semitones -5 … +6).
Pitch tokens and chord symbols are transposed together; duration, is_rest,
phrase_token, and artist_token are unchanged across transpositions.
Cross-phrase context is maintained within each (solo × transposition) pair.

Two construction paths:
  1. NoteWindowDataset(chord_tok, note_tok, ...) — slow build from JSON; emits
     a deprecation warning. Use this only for debugging.
  2. NoteWindowDataset.from_cache(path)        — fast load from a pretokenized
     cache produced by scripts/build_v6_cache.py. This is the path training
     scripts should use.
"""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.data.jazz_vocab_labels import compute_jazz_vocab_labels
import torch
from torch.utils.data import Dataset

_REPO = Path(__file__).parents[2]
_PHRASES_PATH  = _REPO / "data" / "processed" / "phrases_all_with_tempo.json"
_CLUSTERS_PATH = _REPO / "data" / "processed" / "phrase_clusters_all.csv"
_CACHE_PATH    = _REPO / "data" / "processed" / "notes_v6_cache.pt"

WINDOW             = 8
REST_GAP_THRESHOLD = 0.1
MAX_POS_IN_PHRASE  = 15


def _derive_phrase_position(pos_in_phrase: torch.Tensor, window: int) -> torch.Tensor:
    """Best-effort context positions for legacy caches without this tensor."""
    end = int(pos_in_phrase.item())
    return torch.arange(end - window, end, dtype=torch.long).clamp(0, MAX_POS_IN_PHRASE)

# ---------------------------------------------------------------------------
# 12-key transposition
# ---------------------------------------------------------------------------

TRANSPOSE_SEMITONES = list(range(-5, 7))   # all 12 keys

CHORD_TRANSPOSE_MAP = {
    "C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
}

_ROOTS_SORTED = sorted(CHORD_TRANSPOSE_MAP.keys(), key=len, reverse=True)

# Pitch-class → note name (flat spellings, Gb for pc=6)
_PC_TO_NAME: dict = {v: k for k, v in CHORD_TRANSPOSE_MAP.items()
                     if k not in ("F#", "Gb")}
_PC_TO_NAME[6] = "Gb"


def transpose_chord_symbol(symbol: str, semitones: int) -> str:
    """Transpose a WJazzD chord symbol by `semitones`, returning new symbol."""
    if not symbol or symbol in ("NC", "", "PAD", "UNK", "BOS", "EOS"):
        return symbol
    for r in _ROOTS_SORTED:
        if symbol.startswith(r):
            new_pc = (CHORD_TRANSPOSE_MAP[r] + semitones) % 12
            return _PC_TO_NAME[new_pc] + symbol[len(r):]
    return symbol


def _transposed_chord_changes(notes, chord_tok, semitones: int):
    """Return list of chord token IDs for chord-change events, transposed."""
    ids, prev = [], None
    for note in notes:
        trans = transpose_chord_symbol(note["chord"], semitones)
        if trans != prev:
            ids.append(chord_tok.encode(trans))
            prev = trans
    return ids or [chord_tok.UNK]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NoteWindowDataset(Dataset):
    """
    One sample per (note × transposition).
    Context window spans phrase boundaries within the same (solo × transposition).
    """

    def __init__(
        self,
        chord_tok,
        note_tok,
        artist_tok,
        phrase_tok,
        phrases_path=_PHRASES_PATH,
        clusters_path=_CLUSTERS_PATH,
        window=WINDOW,
        semitones_list=TRANSPOSE_SEMITONES,
    ):
        warnings.warn(
            "Building NoteWindowDataset from JSON; prefer .from_cache() for "
            "training to skip per-epoch tokenization.",
            stacklevel=2,
        )
        self._cache = None  # marks the in-memory _samples path

        with open(phrases_path) as f:
            phrases = json.load(f)

        df = pd.read_csv(clusters_path)
        token_map = {
            (int(row.solo_id), int(row.phrase_number)): row.phrase_token
            for row in df.itertuples()
        }

        PAD_P = note_tok.PITCH_PAD
        PAD_D = note_tok.DUR_PAD

        solos: dict = defaultdict(list)
        for p in phrases:
            solos[p["solo_id"]].append(p)

        self._samples = []

        for solo_id, solo_phrases in solos.items():
            solo_phrases.sort(key=lambda p: p["phrase_number"])
            performer    = solo_phrases[0]["performer"]
            artist_id_v  = artist_tok.encode(performer)

            # Pre-encode note durations and is_rest flags once (shared across transpositions)
            # Structure: list of per-phrase tuples (phrase_int, notes, note_enc, tempo_bpm)
            phrase_data = []
            for ph in solo_phrases:
                notes = sorted(ph["notes"], key=lambda x: x["onset"])
                if not notes:
                    continue
                tempo = ph["tempo_bpm"]  # raises KeyError if backfill missing
                phrase_int = phrase_tok.encode(
                    token_map.get((solo_id, ph["phrase_number"]), "PHRASE_00")
                )
                labels = compute_jazz_vocab_labels(notes)
                note_enc = []
                for i, note in enumerate(notes):
                    dur_tok = note_tok.encode_duration(note["duration"], tempo)
                    if i == 0:
                        is_rest = 0
                    else:
                        gap = (note["onset"]
                               - notes[i - 1]["onset"]
                               - notes[i - 1]["duration"])
                        is_rest = 1 if gap > REST_GAP_THRESHOLD else 0
                    note_enc.append((
                        note["pitch"],
                        dur_tok,
                        is_rest,
                        min(i, MAX_POS_IN_PHRASE),
                        labels[i],
                    ))
                phrase_data.append((phrase_int, notes, note_enc, tempo))

            # Build one flat sequence + samples per transposition
            for semitones in semitones_list:
                flat = []
                for phrase_int, notes, note_enc, tempo in phrase_data:
                    # Transposed chord changes for this phrase
                    trans_ids = _transposed_chord_changes(notes, chord_tok, semitones)
                    chord_ids_t = torch.tensor(trans_ids, dtype=torch.long)
                    chord_len   = torch.tensor(len(trans_ids), dtype=torch.long)
                    p_id_t = torch.tensor(phrase_int,   dtype=torch.long)
                    a_id_t = torch.tensor(artist_id_v,  dtype=torch.long)
                    tempo_t = torch.tensor(tempo, dtype=torch.float)

                    for note, (midi_pitch, dur_tok, is_rest, pos_in_ph, jazz_labels) in zip(notes, note_enc):
                        trans_midi = max(48, min(84, midi_pitch + semitones))
                        pitch_tok  = note_tok.encode_pitch(trans_midi)
                        target_chord_id = torch.tensor(
                            chord_tok.encode(transpose_chord_symbol(note["chord"], semitones)),
                            dtype=torch.long,
                        )
                        flat.append((pitch_tok, dur_tok, is_rest,
                                     p_id_t, a_id_t, chord_ids_t, chord_len,
                                     pos_in_ph, tempo_t, target_chord_id,
                                     jazz_labels))

                # Sliding window samples
                for abs_idx, target in enumerate(flat):
                    t_pitch, t_dur, t_rest, p_id, a_id, c_ids, c_len, pos, tempo_t, target_chord_id, jazz_labels = target
                    ctx_start   = max(0, abs_idx - window)
                    ctx_entries = flat[ctx_start:abs_idx]
                    pad_len     = window - len(ctx_entries)

                    ctx_pitch = [PAD_P] * pad_len + [e[0] for e in ctx_entries]
                    ctx_dur   = [PAD_D] * pad_len + [e[1] for e in ctx_entries]
                    ctx_rest  = [0]     * pad_len + [e[2] for e in ctx_entries]
                    phrase_position = [0] * pad_len + [min(e[7], MAX_POS_IN_PHRASE) for e in ctx_entries]

                    self._samples.append({
                        "chord_ids":     c_ids,
                        "chord_len":     c_len,
                        "phrase_id":     p_id,
                        "artist_id":     a_id,
                        "ctx_pitch":     torch.tensor(ctx_pitch, dtype=torch.long),
                        "ctx_dur":       torch.tensor(ctx_dur,   dtype=torch.long),
                        "ctx_rest":      torch.tensor(ctx_rest,  dtype=torch.long),
                        "phrase_position": torch.tensor(phrase_position, dtype=torch.long),
                        "target_pitch":  torch.tensor(t_pitch,   dtype=torch.long),
                        "target_dur":    torch.tensor(t_dur,     dtype=torch.long),
                        "target_rest":   torch.tensor(t_rest,    dtype=torch.long),
                        "target_chord_id": target_chord_id,
                        "jazz_vocab_label": torch.tensor(jazz_labels["jazz_vocab_label"], dtype=torch.long),
                        "chromatic_approach_label": torch.tensor(jazz_labels["is_chromatic_approach_target"], dtype=torch.long),
                        "enclosure_label": torch.tensor(jazz_labels["is_enclosure_target"], dtype=torch.long),
                        "guide_tone_label": torch.tensor(jazz_labels["is_guide_tone"], dtype=torch.long),
                        "blues_color_label": torch.tensor(jazz_labels["is_dominant_blues_color"], dtype=torch.long),
                        "pos_in_phrase": torch.tensor(pos,        dtype=torch.long),
                        "tempo_bpm":     tempo_t,
                    })

    @classmethod
    def from_cache(cls, cache_path=_CACHE_PATH):
        """Skip 12-key augmentation + tokenization; load preprocessed tensors.

        Returns a NoteWindowDataset that backs __getitem__ with the cache
        produced by scripts/build_v6_cache.py instead of an in-memory list
        of dicts. Validates `meta.vocab_version == "v6"`.
        """
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        meta  = cache.get("meta", {})
        if meta.get("vocab_version") != "v6":
            raise ValueError(
                f"Cache vocab_version mismatch: got {meta.get('vocab_version')!r}, expected 'v6'"
            )
        obj = cls.__new__(cls)
        obj._cache   = cache
        obj._samples = None
        obj._n       = int(meta["n_samples"])
        return obj

    def __len__(self):
        if self._cache is not None:
            return self._n
        return len(self._samples)

    def __getitem__(self, idx):
        if self._cache is None:
            return self._samples[idx]
        c = self._cache
        s = int(c["chord_offsets"][idx].item())
        e = int(c["chord_offsets"][idx + 1].item())
        return {
            "chord_ids":     c["chord_ids_flat"][s:e],
            "chord_len":     c["chord_len"][idx],
            "phrase_id":     c["phrase_id"][idx],
            "artist_id":     c["artist_id"][idx],
            "ctx_pitch":     c["ctx_pitch"][idx],
            "ctx_dur":       c["ctx_dur"][idx],
            "ctx_rest":      c["ctx_rest"][idx],
            "phrase_position": c["phrase_position"][idx] if "phrase_position" in c else _derive_phrase_position(c["pos_in_phrase"][idx], c["ctx_pitch"].size(1)),
            "target_pitch":  c["target_pitch"][idx],
            "target_dur":    c["target_dur"][idx],
            "target_rest":   c["target_rest"][idx],
            "target_chord_id": c["target_chord_id"][idx],
            "jazz_vocab_label": c["jazz_vocab_label"][idx] if "jazz_vocab_label" in c else torch.tensor(0, dtype=torch.long),
            "chromatic_approach_label": c["chromatic_approach_label"][idx] if "chromatic_approach_label" in c else torch.tensor(0, dtype=torch.long),
            "enclosure_label": c["enclosure_label"][idx] if "enclosure_label" in c else torch.tensor(0, dtype=torch.long),
            "guide_tone_label": c["guide_tone_label"][idx] if "guide_tone_label" in c else torch.tensor(0, dtype=torch.long),
            "blues_color_label": c["blues_color_label"][idx] if "blues_color_label" in c else torch.tensor(0, dtype=torch.long),
            "pos_in_phrase": c["pos_in_phrase"][idx],
            "tempo_bpm":     c["tempo_bpm"][idx],
        }


def collate_note_window(batch):
    chord_lens = [b["chord_len"].item() for b in batch]
    max_chord  = max(chord_lens)
    B          = len(batch)

    chord_ids  = torch.zeros(B, max_chord, dtype=torch.long)
    chord_mask = torch.zeros(B, max_chord, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = chord_lens[i]
        chord_ids[i, :L] = b["chord_ids"]
        chord_mask[i, :L] = True

    return {
        "chord_ids":     chord_ids,
        "chord_mask":    chord_mask,
        "phrase_id":     torch.stack([b["phrase_id"]     for b in batch]),
        "artist_id":     torch.stack([b["artist_id"]     for b in batch]),
        "ctx_pitch":     torch.stack([b["ctx_pitch"]     for b in batch]),
        "ctx_dur":       torch.stack([b["ctx_dur"]       for b in batch]),
        "ctx_rest":      torch.stack([b["ctx_rest"]      for b in batch]),
        "phrase_position": torch.stack([b["phrase_position"] for b in batch]),
        "target_pitch":  torch.stack([b["target_pitch"]  for b in batch]),
        "target_dur":    torch.stack([b["target_dur"]    for b in batch]),
        "target_rest":   torch.stack([b["target_rest"]   for b in batch]),
        "target_chord_id": torch.stack([b["target_chord_id"] for b in batch]),
        "jazz_vocab_label": torch.stack([b.get("jazz_vocab_label", torch.tensor(0, dtype=torch.long)) for b in batch]),
        "chromatic_approach_label": torch.stack([b.get("chromatic_approach_label", torch.tensor(0, dtype=torch.long)) for b in batch]),
        "enclosure_label": torch.stack([b.get("enclosure_label", torch.tensor(0, dtype=torch.long)) for b in batch]),
        "guide_tone_label": torch.stack([b.get("guide_tone_label", torch.tensor(0, dtype=torch.long)) for b in batch]),
        "blues_color_label": torch.stack([b.get("blues_color_label", torch.tensor(0, dtype=torch.long)) for b in batch]),
        "pos_in_phrase": torch.stack([b["pos_in_phrase"] for b in batch]),
    }
