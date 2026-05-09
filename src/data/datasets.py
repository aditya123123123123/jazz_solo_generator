import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

_REPO = Path(__file__).parents[2]
_PHRASES_PATH = _REPO / "data" / "processed" / "phrases_expanded.json"
_CLUSTERS_PATH = _REPO / "data" / "processed" / "phrase_clusters_expanded.csv"


def _chord_changes(notes, chord_tok):
    ids = []
    prev = None
    for note in notes:
        c = note["chord"]
        if c != prev:
            ids.append(chord_tok.encode(c))
            prev = c
    return ids or [chord_tok.UNK]


class PhrasePlannerDataset(Dataset):
    def __init__(self, chord_tok, note_tok, artist_tok, phrase_tok,
                 phrases_path=_PHRASES_PATH, clusters_path=_CLUSTERS_PATH):
        with open(phrases_path) as f:
            phrases = json.load(f)

        df = pd.read_csv(clusters_path)
        token_map = {
            (int(row.solo_id), int(row.phrase_number)): row.phrase_token
            for row in df.itertuples()
        }

        solos = defaultdict(list)
        for p in phrases:
            solos[p["solo_id"]].append(p)

        self._samples = []
        for solo_id, solo_phrases in sorted(solos.items()):
            solo_phrases.sort(key=lambda p: p["phrase_number"])
            performer = solo_phrases[0]["performer"]

            all_notes = [n for p in solo_phrases for n in p["notes"]]
            chord_ids = _chord_changes(all_notes, chord_tok)

            phrase_ids = [phrase_tok.BOS]
            for p in solo_phrases:
                pt = token_map.get((solo_id, p["phrase_number"]), "PHRASE_00")
                phrase_ids.append(phrase_tok.encode(pt))
            phrase_ids.append(phrase_tok.EOS)

            self._samples.append({
                "chord_ids": torch.tensor(chord_ids, dtype=torch.long),
                "artist_id": torch.tensor(artist_tok.encode(performer), dtype=torch.long),
                "phrase_ids": torch.tensor(phrase_ids, dtype=torch.long),
                "chord_len": torch.tensor(len(chord_ids), dtype=torch.long),
                "phrase_len": torch.tensor(len(phrase_ids), dtype=torch.long),
            })

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


class NoteExecutorDataset(Dataset):
    def __init__(self, chord_tok, note_tok, artist_tok, phrase_tok,
                 phrases_path=_PHRASES_PATH, clusters_path=_CLUSTERS_PATH):
        with open(phrases_path) as f:
            phrases = json.load(f)

        df = pd.read_csv(clusters_path)
        token_map = {
            (int(row.solo_id), int(row.phrase_number)): row.phrase_token
            for row in df.itertuples()
        }

        self._samples = []
        for p in phrases:
            solo_id = p["solo_id"]
            phrase_number = p["phrase_number"]
            notes = p["notes"]

            pt = token_map.get((solo_id, phrase_number), "PHRASE_00")
            chord_ids = _chord_changes(notes, chord_tok)

            pitch_ids = [note_tok.PITCH_BOS] + [note_tok.encode_pitch(n["pitch"]) for n in notes] + [note_tok.PITCH_EOS]
            dur_ids = [note_tok.DUR_BOS] + [note_tok.encode_duration(n["duration"]) for n in notes] + [note_tok.DUR_EOS]

            self._samples.append({
                "phrase_id": torch.tensor(phrase_tok.encode(pt), dtype=torch.long),
                "artist_id": torch.tensor(artist_tok.encode(p["performer"]), dtype=torch.long),
                "chord_ids": torch.tensor(chord_ids, dtype=torch.long),
                "pitch_ids": torch.tensor(pitch_ids, dtype=torch.long),
                "dur_ids": torch.tensor(dur_ids, dtype=torch.long),
                "chord_len": torch.tensor(len(chord_ids), dtype=torch.long),
                "note_len": torch.tensor(len(pitch_ids), dtype=torch.long),
            })

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]
