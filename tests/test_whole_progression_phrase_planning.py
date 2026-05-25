import torch

from src.generation import generate_solo as gen


class DummyChordTok:
    UNK = 0
    def __init__(self):
        self.map = {"Dm7": 4, "G7": 5, "Cj7": 6, "D-7": 4}
    def encode(self, chord):
        return self.map.get(chord, self.UNK)
    def decode(self, idx):
        rev = {v: k for k, v in self.map.items()}
        return rev.get(int(idx), "UNK")


class DummyPhraseTok:
    def encode(self, label):
        return int(label.split("_")[1]) + 3
    def decode(self, token):
        return f"PHRASE_{int(token) - 3:02d}"


class DummyArtistTok:
    def encode(self, _name):
        return 1


class DummyNoteTok:
    def decode_pitch(self, token):
        return int(token)
    def decode_duration(self, token, tempo_bpm=120):
        return 0.25


class RecordingPlanner:
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []
    def generate(self, chord_ids, artist_id, max_len=32, temperature=1.0):
        self.calls.append((chord_ids.clone(), artist_id.clone(), max_len, temperature))
        return list(self.tokens)


class RecordingExecutor:
    def __init__(self):
        self.calls = []
    def generate(self, chord_ids, phrase_id, artist_id, n_notes=16, **kwargs):
        self.calls.append({"chord_ids": chord_ids.clone(), "phrase_id": phrase_id.clone(), "n_notes": n_notes, **kwargs})
        return [(60 + i, 4, 0) for i in range(n_notes)]


def decode_dur(_token, _tempo):
    return 0.25


def make_common(planner_tokens=(13, 14, 15)):
    return {
        "chord_tok": DummyChordTok(),
        "note_tok": DummyNoteTok(),
        "phrase_tok": DummyPhraseTok(),
        "artist_tok": DummyArtistTok(),
        "planner": RecordingPlanner(planner_tokens),
        "executor": RecordingExecutor(),
        "decode_dur": decode_dur,
    }


def test_generate_solo_calls_phrase_planner_once_for_full_progression():
    common = make_common((13, 14, 15))
    progression = [("Dm7", 4), ("G7", 4), ("Cj7", 4)]

    _notes, summaries, _unknowns = gen.generate_solo(progression, **common)

    planner = common["planner"]
    assert len(planner.calls) == 1
    chord_ids, _artist_id, max_len, _temperature = planner.calls[0]
    assert chord_ids.tolist() == [[4, 5, 6]]
    assert max_len == len(progression) + 4
    assert [s["phrase_token"] for s in summaries] == ["PHRASE_10", "PHRASE_11", "PHRASE_12"]
    assert summaries[0]["phrase_plan"] == ["PHRASE_10", "PHRASE_11", "PHRASE_12"]


def test_generate_solo_repeats_short_phrase_plan_to_cover_sections():
    common = make_common((13,))
    progression = [("Dm7", 4), ("G7", 4), ("Cj7", 4)]

    _notes, summaries, _unknowns = gen.generate_solo(progression, **common)

    assert [s["phrase_token"] for s in summaries] == ["PHRASE_10", "PHRASE_10", "PHRASE_10"]
