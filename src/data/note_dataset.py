"""
NoteWindowDataset with cross-phrase context and position-in-phrase tracking.

Samples are grouped by solo so the context window can span across phrase
boundaries within the same solo — giving the model continuity across chords.
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

_REPO = Path(__file__).parents[2]
_PHRASES_PATH = _REPO / "data" / "processed" / "phrases_expanded.json"
_CLUSTERS_PATH = _REPO / "data" / "processed" / "phrase_clusters_expanded.csv"

WINDOW = 8
REST_GAP_THRESHOLD = 0.1   # seconds
MAX_POS_IN_PHRASE  = 15    # position embedding vocabulary size - 1


def _chord_changes(notes, chord_tok):
    ids, prev = [], None
    for note in notes:
        c = note["chord"]
        if c != prev:
            ids.append(chord_tok.encode(c))
            prev = c
    return ids or [chord_tok.UNK]


class NoteWindowDataset(Dataset):
    """
    One sample per note.
    Context window spans phrase boundaries within the same solo.
    Each sample includes pos_in_phrase (0-15) for the TARGET note.
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
    ):
        with open(phrases_path) as f:
            phrases = json.load(f)

        df = pd.read_csv(clusters_path)
        token_map = {
            (int(row.solo_id), int(row.phrase_number)): row.phrase_token
            for row in df.itertuples()
        }

        PAD_P = note_tok.PITCH_PAD   # 0
        PAD_D = note_tok.DUR_PAD     # 0

        # Group phrases by solo, sort temporally
        solos: dict = defaultdict(list)
        for p in phrases:
            solos[p["solo_id"]].append(p)

        self._samples = []

        for solo_id, solo_phrases in solos.items():
            solo_phrases.sort(key=lambda p: p["phrase_number"])
            performer = solo_phrases[0]["performer"]
            artist_id_val = artist_tok.encode(performer)

            # Build flat note sequence for this solo (across all phrases)
            flat: list = []  # each entry is a dict of note data

            for ph in solo_phrases:
                notes = sorted(ph["notes"], key=lambda x: x["onset"])
                if not notes:
                    continue

                phrase_int = phrase_tok.encode(
                    token_map.get((solo_id, ph["phrase_number"]), "PHRASE_00")
                )
                chord_ids_t = torch.tensor(_chord_changes(notes, chord_tok), dtype=torch.long)
                chord_len   = torch.tensor(len(chord_ids_t), dtype=torch.long)

                for pos_in_phrase, note in enumerate(notes):
                    pitch = note_tok.encode_pitch(note["pitch"])
                    dur   = note_tok.encode_duration(note["duration"])
                    if pos_in_phrase == 0:
                        is_rest = 0
                    else:
                        gap = (note["onset"]
                               - notes[pos_in_phrase - 1]["onset"]
                               - notes[pos_in_phrase - 1]["duration"])
                        is_rest = 1 if gap > REST_GAP_THRESHOLD else 0

                    flat.append({
                        "pitch":         pitch,
                        "dur":           dur,
                        "rest":          is_rest,
                        "phrase_id":     torch.tensor(phrase_int,    dtype=torch.long),
                        "artist_id":     torch.tensor(artist_id_val, dtype=torch.long),
                        "chord_ids":     chord_ids_t,
                        "chord_len":     chord_len,
                        "pos_in_phrase": min(pos_in_phrase, MAX_POS_IN_PHRASE),
                    })

            # Create one dataset sample per note in this solo
            for abs_idx, target in enumerate(flat):
                # Context: up to `window` notes immediately before the target
                ctx_start   = max(0, abs_idx - window)
                ctx_entries = flat[ctx_start:abs_idx]
                pad_len     = window - len(ctx_entries)

                ctx_pitch = [PAD_P] * pad_len + [e["pitch"] for e in ctx_entries]
                ctx_dur   = [PAD_D] * pad_len + [e["dur"]   for e in ctx_entries]
                ctx_rest  = [0]     * pad_len + [e["rest"]  for e in ctx_entries]

                self._samples.append({
                    "chord_ids":     target["chord_ids"],
                    "chord_len":     target["chord_len"],
                    "phrase_id":     target["phrase_id"],
                    "artist_id":     target["artist_id"],
                    "ctx_pitch":     torch.tensor(ctx_pitch, dtype=torch.long),
                    "ctx_dur":       torch.tensor(ctx_dur,   dtype=torch.long),
                    "ctx_rest":      torch.tensor(ctx_rest,  dtype=torch.long),
                    "target_pitch":  torch.tensor(target["pitch"], dtype=torch.long),
                    "target_dur":    torch.tensor(target["dur"],   dtype=torch.long),
                    "target_rest":   torch.tensor(target["rest"],  dtype=torch.long),
                    "pos_in_phrase": torch.tensor(target["pos_in_phrase"], dtype=torch.long),
                })

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def collate_note_window(batch):
    chord_lens = [b["chord_len"].item() for b in batch]
    max_chord  = max(chord_lens)
    B          = len(batch)

    chord_ids = torch.zeros(B, max_chord, dtype=torch.long)
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
        "target_pitch":  torch.stack([b["target_pitch"]  for b in batch]),
        "target_dur":    torch.stack([b["target_dur"]    for b in batch]),
        "target_rest":   torch.stack([b["target_rest"]   for b in batch]),
        "pos_in_phrase": torch.stack([b["pos_in_phrase"] for b in batch]),
    }
