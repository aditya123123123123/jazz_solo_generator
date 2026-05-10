"""Rule-based walking bass line generator.

Generates a quarter-note walking line over a chord progression using:
  Beat 1: chord root
  Beat 2: 5th or 3rd (alternates per bar)
  Beat 3: 3rd or 7th (alternates per bar)
  Beat 4: chromatic approach to next chord's root (or 7th/5th if no change)

Returns sequential (pitch_midi, duration_sec, is_rest) tuples — the bass
plays one note per beat with no rests.
"""
import re

# Bass-register root pitches (MIDI 36-57, octave 2-3).
# Roots listed in the spec are used verbatim; remaining accidentals are
# placed at their natural location within the same range.
ROOT_PITCH = {
    "C": 48,
    "C#": 49, "Db": 49,
    "D": 50,
    "D#": 51, "Eb": 51,
    "E": 52,
    "F": 53,
    "F#": 54, "Gb": 54,
    "G": 55,
    "G#": 44, "Ab": 44,
    "A": 57,
    "A#": 46, "Bb": 46,
    "B": 47,
}

# Chord quality → semitone intervals from root: [root, 3rd, 5th, 7th]
QUALITIES = {
    "maj7":    [0, 4, 7, 11],
    "min7":    [0, 3, 7, 10],
    "dom7":    [0, 4, 7, 10],
    "halfdim": [0, 3, 6, 10],
    "dim7":    [0, 3, 6, 9],
    "minmaj7": [0, 3, 7, 11],
}

BASS_LO, BASS_HI = 36, 57

_ROOT_RE = re.compile(r"^([A-G][#b]?)(.*)$")


def _clamp_bass(pitch):
    while pitch > BASS_HI:
        pitch -= 12
    while pitch < BASS_LO:
        pitch += 12
    return pitch


def parse_chord(symbol):
    """Return (root_name, quality_key) for a chord symbol."""
    m = _ROOT_RE.match(symbol)
    if not m:
        raise ValueError(f"Cannot parse chord: {symbol}")
    root, qual = m.group(1), m.group(2)

    if "m7b5" in qual or "ø" in qual or "-7b5" in qual:
        q = "halfdim"
    elif "dim7" in qual or "o7" in qual:
        q = "dim7"
    elif qual.startswith("m") or qual.startswith("-"):
        if "maj7" in qual or "M7" in qual or "j7" in qual:
            q = "minmaj7"
        else:
            q = "min7"
    elif "j7" in qual or "maj7" in qual or "M7" in qual or "Δ" in qual:
        q = "maj7"
    elif "7" in qual:
        q = "dom7"
    else:
        q = "maj7"
    return root, q


def chord_tones_in_bass(symbol):
    """Return [root, 3rd, 5th, 7th] all clamped into MIDI 36-57."""
    root, q = parse_chord(symbol)
    root_pitch = ROOT_PITCH[root]
    return [_clamp_bass(root_pitch + iv) for iv in QUALITIES[q]]


def _approach_note(current_root, next_root):
    """One semitone away from next_root, direction inferred from motion."""
    if next_root > current_root:
        approach = next_root - 1
    elif next_root < current_root:
        approach = next_root + 1
    else:
        approach = next_root - 1
    return _clamp_bass(approach)


def generate_walking_bass(progression, tempo_bpm=120):
    """
    progression: list of (chord_str, beats) tuples
        e.g. [("Dm7", 4), ("G7", 4), ("Cj7", 8)]
    Returns: list of (pitch_midi, duration_sec, is_rest) tuples
    """
    beat_dur = 60.0 / tempo_bpm

    # Pre-compute root pitches for approach-note lookup.
    roots = [ROOT_PITCH[parse_chord(c)[0]] for c, _ in progression]

    events = []
    abs_beat = 0
    for ci, (chord, n_beats) in enumerate(progression):
        tones = chord_tones_in_bass(chord)  # [root, 3rd, 5th, 7th]
        for b in range(n_beats):
            beat_in_bar = abs_beat % 4
            bar_idx = abs_beat // 4
            is_last = (b == n_beats - 1)
            has_next = ci < len(progression) - 1

            if is_last and has_next:
                pitch = _approach_note(tones[0], roots[ci + 1])
            elif beat_in_bar == 0:
                pitch = tones[0]                                       # root
            elif beat_in_bar == 1:
                pitch = tones[2] if bar_idx % 2 == 0 else tones[1]      # 5th / 3rd
            elif beat_in_bar == 2:
                pitch = tones[1] if bar_idx % 2 == 0 else tones[3]      # 3rd / 7th
            else:  # beat_in_bar == 3, no chord change ahead
                pitch = tones[3] if bar_idx % 2 == 0 else tones[2]      # 7th / 5th

            events.append((pitch, beat_dur, False))
            abs_beat += 1

    return events
