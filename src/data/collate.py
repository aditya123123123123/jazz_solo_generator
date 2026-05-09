import torch
from torch.nn.utils.rnn import pad_sequence


def _pad(seqs, pad_val=0):
    max_len = max(s.size(0) for s in seqs)
    B = len(seqs)
    out = torch.full((B, max_len), pad_val, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = s.size(0)
        out[i, :L] = s
        mask[i, :L] = True
    return out, mask


def collate_phrase_planner(batch):
    chord_ids, chord_mask = _pad([b["chord_ids"] for b in batch])
    phrase_ids, phrase_mask = _pad([b["phrase_ids"] for b in batch])
    return {
        "chord_ids": chord_ids,
        "phrase_ids": phrase_ids,
        "artist_id": torch.stack([b["artist_id"] for b in batch]),
        "chord_mask": chord_mask,
        "phrase_mask": phrase_mask,
        "chord_len": torch.stack([b["chord_len"] for b in batch]),
        "phrase_len": torch.stack([b["phrase_len"] for b in batch]),
    }


def collate_note_executor(batch):
    chord_ids, chord_mask = _pad([b["chord_ids"] for b in batch])
    pitch_ids, pitch_mask = _pad([b["pitch_ids"] for b in batch])
    dur_ids, _ = _pad([b["dur_ids"] for b in batch])
    return {
        "phrase_id": torch.stack([b["phrase_id"] for b in batch]),
        "artist_id": torch.stack([b["artist_id"] for b in batch]),
        "chord_ids": chord_ids,
        "pitch_ids": pitch_ids,
        "dur_ids": dur_ids,
        "chord_mask": chord_mask,
        "pitch_mask": pitch_mask,
        "chord_len": torch.stack([b["chord_len"] for b in batch]),
        "note_len": torch.stack([b["note_len"] for b in batch]),
    }
