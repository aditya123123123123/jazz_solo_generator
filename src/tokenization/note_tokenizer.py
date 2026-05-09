import json
import math
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parents[2]
PHRASES_PATH = _REPO / "data" / "processed" / "phrases.json"
BINS_PATH = _REPO / "data" / "processed" / "duration_bins.json"

N_DUR_BINS = 16


class NoteTokenizer:
    # Pitch special tokens
    PITCH_PAD = 0
    PITCH_BOS = 1
    PITCH_EOS = 2
    PITCH_REST = 3
    PITCH_OFFSET = 4  # MIDI pitch p -> token p + 4

    # Duration special tokens
    DUR_PAD = 0
    DUR_BOS = 1
    DUR_EOS = 2
    DUR_OFFSET = 3  # bin index b -> token b + 3

    def __init__(self, bin_edges: list):
        self._bin_edges = list(bin_edges)

    @classmethod
    def from_phrases(cls, phrases_path=PHRASES_PATH):
        with open(phrases_path) as f:
            phrases = json.load(f)
        durations = [n["duration"] for p in phrases for n in p["notes"]]
        min_dur = min(durations)
        max_dur = max(durations)
        edges = np.geomspace(min_dur, max_dur, N_DUR_BINS + 1).tolist()
        return cls(edges)

    @classmethod
    def from_json(cls, bins_path=BINS_PATH):
        with open(bins_path) as f:
            data = json.load(f)
        return cls(data["bin_edges"])

    def save_bins(self, path=BINS_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"bin_edges": self._bin_edges}, f, indent=2)

    # --- pitch ---

    def encode_pitch(self, midi_pitch: int) -> int:
        return int(midi_pitch) + self.PITCH_OFFSET

    def decode_pitch(self, token: int) -> int:
        return token - self.PITCH_OFFSET

    @property
    def pitch_vocab_size(self) -> int:
        return 128 + self.PITCH_OFFSET  # 132

    # --- duration ---

    def encode_duration(self, seconds: float) -> int:
        edges = self._bin_edges
        # internal edges (between first and last) define 16 bins
        bin_idx = int(np.digitize(seconds, edges[1:-1]))  # 0..15
        bin_idx = max(0, min(bin_idx, N_DUR_BINS - 1))
        return bin_idx + self.DUR_OFFSET

    def decode_duration(self, token: int) -> float:
        b = token - self.DUR_OFFSET  # 0..15
        b = max(0, min(b, N_DUR_BINS - 1))
        lo, hi = self._bin_edges[b], self._bin_edges[b + 1]
        return math.sqrt(lo * hi)  # geometric midpoint

    @property
    def dur_vocab_size(self) -> int:
        return N_DUR_BINS + self.DUR_OFFSET  # 19


if __name__ == "__main__":
    tok = NoteTokenizer.from_phrases()
    tok.save_bins()
    print(f"Duration bin edges ({N_DUR_BINS} bins, {N_DUR_BINS + 1} edges):")
    for i, e in enumerate(tok._bin_edges):
        print(f"  edge[{i:2d}] = {e:.6f}s")
    print(f"Saved to {BINS_PATH}")
