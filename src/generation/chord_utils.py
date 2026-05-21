"""Shared chord parsing helpers for generation-time harmony logic."""
from __future__ import annotations

import re

_ROOT_PC = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

# [root, 3rd, 5th, 7th] semitone offsets from root.
_QUALITY_INTERVALS = {
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
    "halfdim": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "minmaj7": (0, 3, 7, 11),
}

_ROOT_RE = re.compile(r"^([A-G][#b]?)(.*)$")


def parse_chord(symbol: str) -> tuple[int, str] | None:
    """Return (root_pc, quality_key) or None if the symbol is unparseable."""
    m = _ROOT_RE.match(symbol)
    if not m:
        return None
    root_name, qual = m.group(1), m.group(2)
    if root_name not in _ROOT_PC:
        return None
    root_pc = _ROOT_PC[root_name]

    if "m7b5" in qual or "ø" in qual or "-7b5" in qual:
        quality = "halfdim"
    elif "dim7" in qual or "o7" in qual:
        quality = "dim7"
    elif qual.startswith("m") or qual.startswith("-"):
        if "maj7" in qual or "M7" in qual or "j7" in qual:
            quality = "minmaj7"
        else:
            quality = "min7"
    elif "j7" in qual or "maj7" in qual or "M7" in qual or "Δ" in qual:
        quality = "maj7"
    elif "7" in qual:
        quality = "dom7"
    else:
        quality = "maj7"
    return root_pc, quality


def chord_tones(root_pc: int, quality: str) -> list[int]:
    """Return [root, 3rd, 5th, 7th] absolute pitch classes (0-11)."""
    return [(root_pc + interval) % 12 for interval in _QUALITY_INTERVALS[quality]]
