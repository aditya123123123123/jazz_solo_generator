import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

_REPO = Path(__file__).parents[2]
_PHRASES_PATH = _REPO / "data" / "processed" / "phrases_expanded.json"
_CLUSTERS_PATH = _REPO / "data" / "processed" / "phrase_clusters_expanded.csv"

WINDOW = 8
REST_GAP_THRESHOLD = 0.1  # seconds; gap > this between consecutive notes -> is_rest=1


def _chord_changes(notes, chord_tok):
    ids, prev = [], None
    for note in notes:
        c = note["chord"]
        if c != prev:
            ids.append(chord_tok.encode(c))
            prev = c
    return ids or [chord_tok.UNK]


class NoteWindowDataset(Dataset):
    """One sample per note: (chord_context, phrase_id, artist_id, 8-note window) → next note."""

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

        PAD_P = note_tok.PITCH_PAD
        PAD_D = note_tok.DUR_PAD
        BOS_P = note_tok.PITCH_BOS
        BOS_D = note_tok.DUR_BOS

        self._samples = []
        for p in phrases:
            notes = sorted(p["notes"], key=lambda x: x["onset"])
            if not notes:
                continue

            solo_id, phrase_number, performer = p["solo_id"], p["phrase_number"], p["performer"]

            # Phrase-level chord context (shared across all notes in this phrase)
            chord_ids = torch.tensor(_chord_changes(notes, chord_tok), dtype=torch.long)

            pt = token_map.get((solo_id, phrase_number), "PHRASE_00")
            phrase_id = torch.tensor(phrase_tok.encode(pt), dtype=torch.long)
            artist_id = torch.tensor(artist_tok.encode(performer), dtype=torch.long)

            # Encode every note
            encoded = []  # list of (pitch_tok, dur_tok, is_rest)
            for i, note in enumerate(notes):
                pitch = note_tok.encode_pitch(note["pitch"])
                dur = note_tok.encode_duration(note["duration"])
                if i == 0:
                    is_rest = 0
                else:
                    gap = notes[i]["onset"] - (notes[i - 1]["onset"] + notes[i - 1]["duration"])
                    is_rest = 1 if gap > REST_GAP_THRESHOLD else 0
                encoded.append((pitch, dur, is_rest))

            chord_len = torch.tensor(len(chord_ids), dtype=torch.long)

            for target_idx in range(len(encoded)):
                # Build fixed-size context window: BOS + up to (window-1) previous notes
                prev_start = max(0, target_idx - (window - 1))
                ctx = [(BOS_P, BOS_D, 0)] + encoded[prev_start:target_idx]

                # Left-pad to exactly `window` positions
                pad = window - len(ctx)
                ctx = [(PAD_P, PAD_D, 0)] * pad + ctx

                ctx_pitch = torch.tensor([t[0] for t in ctx], dtype=torch.long)
                ctx_dur   = torch.tensor([t[1] for t in ctx], dtype=torch.long)
                ctx_rest  = torch.tensor([t[2] for t in ctx], dtype=torch.long)

                t_pitch, t_dur, t_rest = encoded[target_idx]
                self._samples.append({
                    "chord_ids":    chord_ids,
                    "phrase_id":    phrase_id,
                    "artist_id":    artist_id,
                    "ctx_pitch":    ctx_pitch,
                    "ctx_dur":      ctx_dur,
                    "ctx_rest":     ctx_rest,
                    "target_pitch": torch.tensor(t_pitch, dtype=torch.long),
                    "target_dur":   torch.tensor(t_dur,   dtype=torch.long),
                    "target_rest":  torch.tensor(t_rest,  dtype=torch.long),
                    "chord_len":    chord_len,
                })

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def collate_note_window(batch):
    chord_lens = [b["chord_len"].item() for b in batch]
    max_chord = max(chord_lens)
    B = len(batch)

    chord_ids = torch.zeros(B, max_chord, dtype=torch.long)
    chord_mask = torch.zeros(B, max_chord, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = chord_lens[i]
        chord_ids[i, :L] = b["chord_ids"]
        chord_mask[i, :L] = True

    return {
        "chord_ids":    chord_ids,
        "chord_mask":   chord_mask,
        "phrase_id":    torch.stack([b["phrase_id"]    for b in batch]),
        "artist_id":    torch.stack([b["artist_id"]    for b in batch]),
        "ctx_pitch":    torch.stack([b["ctx_pitch"]    for b in batch]),
        "ctx_dur":      torch.stack([b["ctx_dur"]      for b in batch]),
        "ctx_rest":     torch.stack([b["ctx_rest"]     for b in batch]),
        "target_pitch": torch.stack([b["target_pitch"] for b in batch]),
        "target_dur":   torch.stack([b["target_dur"]   for b in batch]),
        "target_rest":  torch.stack([b["target_rest"]  for b in batch]),
    }
