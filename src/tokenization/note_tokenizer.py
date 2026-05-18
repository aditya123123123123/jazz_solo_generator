"""Beat-relative musical-value duration tokenizer (v6).

Replaces the v5 geometric-binning tokenizer. Durations are encoded relative
to the local tempo (beats) and quantized to the nearest of 14 fixed musical
values. Pitch encoding is unchanged from v5.
"""
import json
from pathlib import Path

_REPO       = Path(__file__).parents[2]
CONFIG_PATH = _REPO / "data" / "processed" / "duration_musical.json"

VERSION    = "v6"
PAD        = 0
BOS        = 1
EOS        = 2
REST       = 3
DUR_OFFSET = 4

MUSICAL_DURATIONS_BEATS = [
    1/8, 1/6, 1/4, 1/3, 3/8, 1/2, 2/3, 3/4,
    1.0, 4/3, 3/2, 2.0, 3.0, 4.0,
]
VOCAB_SIZE = DUR_OFFSET + len(MUSICAL_DURATIONS_BEATS)  # 18


class NoteTokenizer:
    # Pitch special tokens (unchanged from v5)
    PITCH_PAD    = 0
    PITCH_BOS    = 1
    PITCH_EOS    = 2
    PITCH_REST   = 3
    PITCH_OFFSET = 4

    # Duration special tokens
    DUR_PAD    = PAD
    DUR_BOS    = BOS
    DUR_EOS    = EOS
    DUR_REST   = REST
    DUR_OFFSET = DUR_OFFSET

    def __init__(self, durations_beats=None):
        self._durations_beats = list(durations_beats or MUSICAL_DURATIONS_BEATS)
        if len(self._durations_beats) not in (14, 15):
            raise ValueError(
                f"Expected 14 musical durations, got {len(self._durations_beats)}"
            )

    @classmethod
    def from_json(cls, path=CONFIG_PATH):
        with open(path) as f:
            cfg = json.load(f)
        if cfg.get("version") != VERSION:
            raise ValueError(
                f"Expected version={VERSION!r}, got {cfg.get('version')!r}"
            )
        if cfg.get("vocab_size") != VOCAB_SIZE:
            raise ValueError(
                f"vocab_size mismatch: expected {VOCAB_SIZE}, got {cfg.get('vocab_size')}"
            )
        for name, expected in (("PAD", PAD), ("BOS", BOS), ("EOS", EOS),
                               ("REST", REST), ("DUR_OFFSET", DUR_OFFSET)):
            if cfg.get("offsets", {}).get(name) != expected:
                raise ValueError(f"offsets.{name} mismatch in {path}")
        return cls(cfg["musical_durations_beats"])

    def save(self, path=CONFIG_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "version":                 VERSION,
            "vocab_size":              VOCAB_SIZE,
            "musical_durations_beats": self._durations_beats,
            "offsets": {"PAD": PAD, "BOS": BOS, "EOS": EOS,
                        "REST": REST, "DUR_OFFSET": DUR_OFFSET},
        }
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

    # --- pitch (unchanged from v5) ---

    def encode_pitch(self, midi_pitch: int) -> int:
        return int(midi_pitch) + self.PITCH_OFFSET

    def decode_pitch(self, token: int) -> int:
        return token - self.PITCH_OFFSET

    @property
    def pitch_vocab_size(self) -> int:
        return 128 + self.PITCH_OFFSET  # 132

    # --- duration (beat-relative musical values) ---

    def encode_duration(self, seconds: float, tempo_bpm: float) -> int:
        beats = seconds * tempo_bpm / 60.0
        d = self._durations_beats
        idx = min(range(len(d)), key=lambda i: abs(beats - d[i]))
        return idx + DUR_OFFSET

    def decode_duration(self, token: int, tempo_bpm: float) -> float:
        idx = token - DUR_OFFSET
        if not 0 <= idx < len(self._durations_beats):
            raise ValueError(f"Invalid duration token: {token}")
        return self._durations_beats[idx] * 60.0 / tempo_bpm

    @property
    def dur_vocab_size(self) -> int:
        return VOCAB_SIZE


# --- pytest-style unit tests --------------------------------------------------

def test_tempo_invariance():
    """0.25s @ 120 BPM and 0.125s @ 240 BPM are both 0.5 beats → same token."""
    tok = NoteTokenizer()
    assert tok.encode_duration(0.25, 120.0) == tok.encode_duration(0.125, 240.0)


def test_swing_preservation():
    """A 1/3-beat note must map to the 1/3 token, not 1/2 or 3/8."""
    tok = NoteTokenizer()
    seconds = (1/3) * 60.0 / 120.0
    expected_idx = MUSICAL_DURATIONS_BEATS.index(1/3)
    assert tok.encode_duration(seconds, 120.0) == DUR_OFFSET + expected_idx


def _round_trip_max_err(sample):
    tok = NoteTokenizer()
    max_err = 0.0
    for dur_sec, tempo in sample:
        token     = tok.encode_duration(dur_sec, tempo)
        decoded   = tok.decode_duration(token, tempo)
        beats_err = abs((dur_sec - decoded) * tempo / 60.0)
        max_err   = max(max_err, beats_err)
    return max_err


def _load_phrase_notes():
    path = _REPO / "data" / "processed" / "phrases_all_with_tempo.json"
    with open(path) as f:
        phrases = json.load(f)
    return [(n["duration"], p["tempo_bpm"]) for p in phrases for n in p["notes"]]


def test_round_trip_common_notes():
    """1000 random notes with beats ≤ 1.0 (95.4% of corpus, densest vocab region).
    Theoretical max error in this band is 0.125 (gap between 3/4 and 1.0)."""
    import random
    pool = [(d, t) for d, t in _load_phrase_notes() if d * t / 60.0 <= 1.0]
    random.seed(42)
    sample = random.sample(pool, 1000)
    max_err = _round_trip_max_err(sample)
    assert max_err <= 0.13, f"max round-trip error (≤1.0 beats) = {max_err:.4f} beats"


def test_round_trip_full_corpus():
    """1000 random notes with beats ≤ 4.0 (99.7% of corpus, excludes held-note tail).
    Theoretical max error in this band is 0.5 (gap between 2.0 and 3.0, or 3.0 and 4.0)."""
    import random
    pool = [(d, t) for d, t in _load_phrase_notes() if d * t / 60.0 <= 4.0]
    random.seed(42)
    sample = random.sample(pool, 1000)
    max_err = _round_trip_max_err(sample)
    assert max_err <= 0.5, f"max round-trip error (≤4.0 beats) = {max_err:.4f} beats"
