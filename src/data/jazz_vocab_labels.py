"""Per-note jazz vocabulary labels for NoteWindowDataset training targets."""
from __future__ import annotations

from src.generation.chord_utils import chord_tones, parse_chord

GUIDE_TONE_INTERVALS = {
    "maj7": (4, 11),
    "min7": (3, 10),
    "dom7": (4, 10),
    "halfdim": (3, 10),
    "dim7": (3, 9),
    "minmaj7": (3, 11),
}


def _pc(pitch: int) -> int:
    return int(pitch) % 12


def _pitch_distance_mod12(a: int, b: int) -> int:
    d = abs(_pc(a) - _pc(b)) % 12
    return min(d, 12 - d)


def _is_integer_beat(value: float, tolerance: float = 1e-6) -> bool:
    return abs(float(value) - round(float(value))) <= tolerance


def _is_weak_beat(note: dict) -> bool:
    beat = float(note.get("beat", 0.0) or 0.0)
    return not _is_integer_beat(beat)


def _is_strong_beat(note: dict) -> bool:
    beat = float(note.get("beat", 0.0) or 0.0)
    return _is_integer_beat(beat)


def _is_beat_boundary_pair(prev: dict, current: dict) -> bool:
    prev_beat = float(prev.get("beat", 0.0) or 0.0)
    cur_beat = float(current.get("beat", 0.0) or 0.0)
    if _is_weak_beat(prev) and _is_strong_beat(current):
        return True
    return prev_beat != cur_beat and _is_strong_beat(current)


def forward_fill_chords(notes: list[dict]) -> list[str]:
    filled: list[str] = []
    current = ""
    for note in notes:
        chord = str(note.get("chord", "") or "").strip()
        if chord:
            current = chord
        filled.append(current)
    return filled


def chord_info(chord: str):
    parsed = parse_chord(chord)
    if parsed is None:
        return None
    root_pc, quality = parsed
    tones = set(chord_tones(root_pc, quality))
    guide = {(root_pc + interval) % 12 for interval in GUIDE_TONE_INTERVALS.get(quality, ())}
    return root_pc, quality, tones, guide


def is_chromatic_approach(prev_pitch: int, target_pitch: int, target_tones: set[int]) -> bool:
    return _pc(target_pitch) in target_tones and _pitch_distance_mod12(prev_pitch, target_pitch) == 1


def is_enclosure(p1: int, p2: int, target: int, target_tones: set[int]) -> bool:
    if _pc(target) not in target_tones:
        return False
    return (p1 - target) * (p2 - target) < 0 and max(abs(p1 - target), abs(p2 - target)) <= 3


def compute_jazz_vocab_labels(notes: list[dict]) -> list[dict[str, int]]:
    """Return target-aligned scalar labels for each note in a phrase.

    Labels describe the note being predicted, not a phrase-level list. This is
    deliberate: NoteWindowDataset trains one target note per sample.
    """
    notes = sorted(notes, key=lambda n: (float(n.get("onset", 0.0)), int(n.get("eventid", 0))))
    filled_chords = forward_fill_chords(notes)
    labels = [
        {
            "is_chromatic_approach_target": 0,
            "is_enclosure_target": 0,
            "is_guide_tone": 0,
            "is_dominant_blues_color": 0,
            "jazz_vocab_label": 0,
        }
        for _ in notes
    ]

    for i, note in enumerate(notes):
        info = chord_info(filled_chords[i])
        if info is None:
            continue
        root_pc, quality, tones, guide = info
        pitch = int(note["pitch"])

        if _pc(pitch) in guide:
            labels[i]["is_guide_tone"] = 1
        if quality == "dom7":
            blues_pcs = {(root_pc + 3) % 12, (root_pc + 6) % 12, (root_pc + 10) % 12}
            if _pc(pitch) in blues_pcs:
                labels[i]["is_dominant_blues_color"] = 1
        if i >= 1 and _is_beat_boundary_pair(notes[i - 1], note):
            if is_chromatic_approach(int(notes[i - 1]["pitch"]), pitch, tones):
                labels[i]["is_chromatic_approach_target"] = 1
        if i >= 2 and _pc(pitch) in guide:
            if is_enclosure(int(notes[i - 2]["pitch"]), int(notes[i - 1]["pitch"]), pitch, guide):
                labels[i]["is_enclosure_target"] = 1

        labels[i]["jazz_vocab_label"] = int(
            labels[i]["is_chromatic_approach_target"]
            or labels[i]["is_enclosure_target"]
            or labels[i]["is_guide_tone"]
            or labels[i]["is_dominant_blues_color"]
        )

    return labels
