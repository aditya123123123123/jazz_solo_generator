"""Rule-based rhythm section generator (piano comping + walking bass + drums).

Returns a list of pretty_midi.Instrument objects synchronized to a chord
progression. Deterministic by seed — same chords + tempo + style + seed
produce identical output. No global RNG state is touched.

Public API:
    generate_rhythm_section(chords, tempo_bpm,
                            time_signature=(4, 4),
                            style="swing", seed=42)
        -> list[pretty_midi.Instrument]
"""
from __future__ import annotations

import random
import re
from typing import List, Tuple

import pretty_midi

# ---------------------------------------------------------------------------
# Chord parsing
# ---------------------------------------------------------------------------

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
    "maj7":    (0, 4, 7, 11),
    "min7":    (0, 3, 7, 10),
    "dom7":    (0, 4, 7, 10),
    "halfdim": (0, 3, 6, 10),
    "dim7":    (0, 3, 6, 9),
    "minmaj7": (0, 3, 7, 11),
}

_ROOT_RE = re.compile(r"^([A-G][#b]?)(.*)$")


def _parse_chord(symbol: str) -> Tuple[int, str] | None:
    """Return (root_pc, quality_key) or None if the symbol is unparseable."""
    m = _ROOT_RE.match(symbol)
    if not m:
        return None
    root_name, qual = m.group(1), m.group(2)
    if root_name not in _ROOT_PC:
        return None
    root_pc = _ROOT_PC[root_name]

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
    return root_pc, q


def _chord_tones(root_pc: int, quality: str) -> List[int]:
    """Return [root, 3rd, 5th, 7th] absolute pitch classes (0-11)."""
    return [(root_pc + iv) % 12 for iv in _QUALITY_INTERVALS[quality]]


def _clamp(pitch: int, lo: int, hi: int) -> int:
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


# ---------------------------------------------------------------------------
# Drum-kit pitches (General MIDI)
# ---------------------------------------------------------------------------
KICK, SNARE, SIDESTICK = 36, 38, 37
CLOSED_HH, PEDAL_HH, OPEN_HH = 42, 44, 46
LO_TOM, MID_TOM, HI_TOM = 45, 47, 50
CRASH, RIDE, RIDE_BELL = 49, 51, 53
COWBELL = 56


# ---------------------------------------------------------------------------
# Per-chord segment info
# ---------------------------------------------------------------------------

def _segments(chords, beat_dur):
    """Yield (chord_idx, chord_str, start_sec, n_beats, next_root_pc_or_None)."""
    abs_beat = 0
    parsed = []
    for sym, beats in chords:
        parsed.append(_parse_chord(sym))
    for i, ((sym, beats), p) in enumerate(zip(chords, parsed)):
        next_root_pc = None
        for j in range(i + 1, len(chords)):
            if parsed[j] is not None:
                next_root_pc = parsed[j][0]
                break
        yield i, sym, p, abs_beat * beat_dur, beats, next_root_pc, abs_beat
        abs_beat += beats


# ---------------------------------------------------------------------------
# Style: swing
# ---------------------------------------------------------------------------

def _swing(chords, beat_dur, beats_per_bar, piano_inst, bass_inst, drum_inst, rng):
    BASS_LO, BASS_HI = 36, 57
    PIANO_LO, PIANO_HI = 55, 76    # G3 .. E5 — comping range
    total_beats = sum(b for _, b in chords)

    # ---- Bass: 1 note per beat ----
    parsed = [_parse_chord(s) for s, _ in chords]
    chord_starts = []
    abs_b = 0
    for _, b in chords:
        chord_starts.append(abs_b)
        abs_b += b

    abs_beat = 0
    for ci, (sym, n_beats) in enumerate(chords):
        p = parsed[ci]
        if p is None:
            # Unknown chord: alternate root–fifth–root–fifth (using best-effort root=0)
            root_pc = 0
            tones_pc = [0, 7, 0, 7]
        else:
            root_pc, quality = p
            tones_pc = _chord_tones(root_pc, quality)  # [root, 3, 5, 7]

        # Next chord's root for approach note
        next_root_pc = None
        for j in range(ci + 1, len(chords)):
            if parsed[j] is not None:
                next_root_pc = parsed[j][0]
                break

        for b in range(n_beats):
            beat_in_bar = abs_beat % 4
            bar_idx = abs_beat // 4
            is_last = (b == n_beats - 1) and (ci < len(chords) - 1) and (next_root_pc is not None)

            if is_last:
                # Chromatic approach: ±1 semitone from next_root in same octave
                base = _clamp(48 + next_root_pc, BASS_LO, BASS_HI)
                pitch = _clamp(base - 1, BASS_LO, BASS_HI)
            elif beat_in_bar == 0:
                pitch = _clamp(48 + tones_pc[0], BASS_LO, BASS_HI)
            elif beat_in_bar == 1:
                pc = tones_pc[2] if bar_idx % 2 == 0 else tones_pc[1]
                pitch = _clamp(48 + pc, BASS_LO, BASS_HI)
            elif beat_in_bar == 2:
                pc = tones_pc[1] if bar_idx % 2 == 0 else tones_pc[3]
                pitch = _clamp(48 + pc, BASS_LO, BASS_HI)
            else:
                pc = tones_pc[3] if bar_idx % 2 == 0 else tones_pc[2]
                pitch = _clamp(48 + pc, BASS_LO, BASS_HI)

            t0 = abs_beat * beat_dur
            t1 = t0 + beat_dur * 0.95
            vel = 80 + rng.randint(-5, 5)
            bass_inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=t0, end=t1))
            abs_beat += 1

    # ---- Piano: syncopated shell voicings ----
    # Hits at beats 1.0 and 2.5 of each bar within each chord segment.
    abs_beat = 0
    for ci, (sym, n_beats) in enumerate(chords):
        p = parsed[ci]
        if p is None:
            abs_beat += n_beats
            continue
        root_pc, quality = p
        tones_pc = _chord_tones(root_pc, quality)
        # Shell voicing: 3rd + 7th + 9th(approx) → use root, 3rd, 7th
        voicing_pcs = [tones_pc[1], tones_pc[3], (root_pc + 14) % 12]
        voicing = sorted(_clamp(60 + pc, PIANO_LO, PIANO_HI) for pc in voicing_pcs)

        for bar_offset in range(0, n_beats, 4):
            bar_beats = min(4, n_beats - bar_offset)
            # hit 1: beat 0
            if bar_beats >= 1:
                t = (abs_beat + bar_offset + 0.0) * beat_dur
                _piano_hit(piano_inst, voicing, t, beat_dur * 1.2, rng)
            # hit 2: "& of 2" (beat 2.5) with swing offset
            if bar_beats >= 3:
                t = (abs_beat + bar_offset + 2.0 + 0.67) * beat_dur
                _piano_hit(piano_inst, voicing, t, beat_dur * 0.6, rng)
        abs_beat += n_beats

    # ---- Drums: ride + hi-hat 2&4 + light kick on 1 ----
    long_dur, short_dur = 0.67 * beat_dur, 0.33 * beat_dur
    for beat in range(total_beats):
        t = beat * beat_dur
        beat_in_bar = beat % 4
        drum_inst.notes.append(pretty_midi.Note(
            velocity=80 + rng.randint(-5, 5), pitch=RIDE,
            start=t, end=t + long_dur))
        drum_inst.notes.append(pretty_midi.Note(
            velocity=65 + rng.randint(-5, 5), pitch=RIDE,
            start=t + 0.67 * beat_dur, end=t + 0.67 * beat_dur + short_dur))
        if beat_in_bar == 0:
            drum_inst.notes.append(pretty_midi.Note(
                velocity=70, pitch=KICK,
                start=t, end=t + beat_dur * 0.5))
        if beat_in_bar in (1, 3):
            drum_inst.notes.append(pretty_midi.Note(
                velocity=60, pitch=PEDAL_HH,
                start=t, end=t + beat_dur * 0.5))


def _piano_hit(inst, voicing, start, dur, rng):
    """Append a chord (multiple Notes at same start) with velocity jitter."""
    for pitch in voicing:
        vel = 70 + rng.randint(-5, 5)
        inst.notes.append(pretty_midi.Note(
            velocity=vel, pitch=pitch, start=start, end=start + dur))


# ---------------------------------------------------------------------------
# Style: bossa
# ---------------------------------------------------------------------------

def _bossa(chords, beat_dur, beats_per_bar, piano_inst, bass_inst, drum_inst, rng):
    BASS_LO, BASS_HI = 36, 57
    PIANO_LO, PIANO_HI = 55, 76
    total_beats = sum(b for _, b in chords)
    parsed = [_parse_chord(s) for s, _ in chords]

    # Bass: root on 1, fifth on 3 (half-note feel, dotted-quarter pickup variant)
    abs_beat = 0
    for ci, (sym, n_beats) in enumerate(chords):
        p = parsed[ci]
        if p is None:
            root_pc, fifth_pc = 0, 7
        else:
            root_pc, q = p
            tones = _chord_tones(root_pc, q)
            fifth_pc = tones[2]
        for bar_offset in range(0, n_beats, 4):
            t_root = (abs_beat + bar_offset) * beat_dur
            t_fifth = (abs_beat + bar_offset + 2) * beat_dur
            bass_inst.notes.append(pretty_midi.Note(
                velocity=80, pitch=_clamp(36 + root_pc, BASS_LO, BASS_HI),
                start=t_root, end=t_root + beat_dur * 1.9))
            if bar_offset + 2 < n_beats:
                bass_inst.notes.append(pretty_midi.Note(
                    velocity=75, pitch=_clamp(36 + fifth_pc, BASS_LO, BASS_HI),
                    start=t_fifth, end=t_fifth + beat_dur * 1.9))
        abs_beat += n_beats

    # Piano: bossa rhythm — hits on beats 1, 1.5, 2.5, 4 of each bar
    abs_beat = 0
    bossa_offsets = (0.0, 1.5, 2.5, 4.0)
    for ci, (sym, n_beats) in enumerate(chords):
        p = parsed[ci]
        if p is None:
            abs_beat += n_beats
            continue
        root_pc, q = p
        tones = _chord_tones(root_pc, q)
        voicing = sorted(_clamp(60 + pc, PIANO_LO, PIANO_HI) for pc in tones)
        for bar_offset in range(0, n_beats, 4):
            bar_beats = min(4, n_beats - bar_offset)
            for off in bossa_offsets:
                if off < bar_beats:
                    t = (abs_beat + bar_offset + off) * beat_dur
                    _piano_hit(piano_inst, voicing, t, beat_dur * 0.6, rng)
        abs_beat += n_beats

    # Drums: clave (3-2 son) on side-stick + closed hi-hat 8ths
    # 3-side: beats 1, 1.5, 2.5;  2-side: beats 3.5, 4.5  (per 2-bar)
    clave_offsets_bar0 = (0.0, 1.5, 2.5)
    clave_offsets_bar1 = (1.5, 2.5)  # within bar 1 of the 2-bar pattern
    abs_beat = 0
    total_bars = total_beats // 4 + (1 if total_beats % 4 else 0)
    for bar in range(total_bars):
        bar_start = bar * 4 * beat_dur
        offsets = clave_offsets_bar0 if bar % 2 == 0 else clave_offsets_bar1
        for off in offsets:
            t = bar_start + off * beat_dur
            drum_inst.notes.append(pretty_midi.Note(
                velocity=75, pitch=SIDESTICK, start=t, end=t + beat_dur * 0.25))
    # Continuous 8ths on closed hi-hat
    for beat in range(total_beats):
        for sub in (0.0, 0.5):
            t = (beat + sub) * beat_dur
            drum_inst.notes.append(pretty_midi.Note(
                velocity=55, pitch=CLOSED_HH, start=t, end=t + beat_dur * 0.3))
        if beat % 4 == 0:
            drum_inst.notes.append(pretty_midi.Note(
                velocity=70, pitch=KICK, start=beat * beat_dur,
                end=beat * beat_dur + beat_dur * 0.4))


# ---------------------------------------------------------------------------
# Style: ballad
# ---------------------------------------------------------------------------

def _ballad(chords, beat_dur, beats_per_bar, piano_inst, bass_inst, drum_inst, rng):
    BASS_LO, BASS_HI = 36, 57
    PIANO_LO, PIANO_HI = 55, 76
    total_beats = sum(b for _, b in chords)
    parsed = [_parse_chord(s) for s, _ in chords]

    # Bass: root + fifth half notes
    abs_beat = 0
    for ci, (sym, n_beats) in enumerate(chords):
        p = parsed[ci]
        if p is None:
            root_pc, fifth_pc = 0, 7
        else:
            root_pc, q = p
            tones = _chord_tones(root_pc, q)
            fifth_pc = tones[2]
        for bar_offset in range(0, n_beats, 4):
            t_root = (abs_beat + bar_offset) * beat_dur
            t_fifth = (abs_beat + bar_offset + 2) * beat_dur
            bass_inst.notes.append(pretty_midi.Note(
                velocity=70, pitch=_clamp(36 + root_pc, BASS_LO, BASS_HI),
                start=t_root, end=t_root + beat_dur * 1.9))
            if bar_offset + 2 < n_beats:
                bass_inst.notes.append(pretty_midi.Note(
                    velocity=65, pitch=_clamp(36 + fifth_pc, BASS_LO, BASS_HI),
                    start=t_fifth, end=t_fifth + beat_dur * 1.9))
        abs_beat += n_beats

    # Piano: sustained whole-bar voicing per chord segment
    abs_beat = 0
    for ci, (sym, n_beats) in enumerate(chords):
        p = parsed[ci]
        if p is None:
            abs_beat += n_beats
            continue
        root_pc, q = p
        tones = _chord_tones(root_pc, q)
        voicing = sorted(_clamp(60 + pc, PIANO_LO, PIANO_HI) for pc in tones)
        for bar_offset in range(0, n_beats, 4):
            bar_beats = min(4, n_beats - bar_offset)
            t = (abs_beat + bar_offset) * beat_dur
            dur = bar_beats * beat_dur * 0.95
            for pitch in voicing:
                piano_inst.notes.append(pretty_midi.Note(
                    velocity=55 + rng.randint(-3, 3),
                    pitch=pitch, start=t, end=t + dur))
        abs_beat += n_beats

    # Drums: brush quarter-notes on ride + light snare on 4
    for beat in range(total_beats):
        t = beat * beat_dur
        drum_inst.notes.append(pretty_midi.Note(
            velocity=45, pitch=RIDE,
            start=t, end=t + beat_dur * 0.9))
        if beat % 4 == 3:
            drum_inst.notes.append(pretty_midi.Note(
                velocity=40, pitch=SNARE,
                start=t, end=t + beat_dur * 0.5))


# ---------------------------------------------------------------------------
# Style: latin
# ---------------------------------------------------------------------------

def _latin(chords, beat_dur, beats_per_bar, piano_inst, bass_inst, drum_inst, rng):
    BASS_LO, BASS_HI = 36, 57
    PIANO_LO, PIANO_HI = 55, 76
    total_beats = sum(b for _, b in chords)
    parsed = [_parse_chord(s) for s, _ in chords]

    # Bass: tumbao — root on 1, fifth on 2.5, root on 4 (with octave variation)
    abs_beat = 0
    for ci, (sym, n_beats) in enumerate(chords):
        p = parsed[ci]
        if p is None:
            root_pc, fifth_pc = 0, 7
        else:
            root_pc, q = p
            tones = _chord_tones(root_pc, q)
            fifth_pc = tones[2]
        root_p = _clamp(36 + root_pc, BASS_LO, BASS_HI)
        fifth_p = _clamp(36 + fifth_pc, BASS_LO, BASS_HI)
        for bar_offset in range(0, n_beats, 4):
            bar_beats = min(4, n_beats - bar_offset)
            tb = (abs_beat + bar_offset) * beat_dur
            bass_inst.notes.append(pretty_midi.Note(
                velocity=80, pitch=root_p, start=tb, end=tb + beat_dur * 0.9))
            if bar_beats >= 3:
                t2 = tb + 2.5 * beat_dur
                bass_inst.notes.append(pretty_midi.Note(
                    velocity=75, pitch=fifth_p, start=t2, end=t2 + beat_dur * 0.9))
            if bar_beats >= 4:
                t3 = tb + 3.5 * beat_dur
                bass_inst.notes.append(pretty_midi.Note(
                    velocity=75, pitch=root_p, start=t3, end=t3 + beat_dur * 0.9))
        abs_beat += n_beats

    # Piano montuno: offbeat 8ths emphasizing chord tones
    abs_beat = 0
    montuno_offsets = (0.5, 1.0, 2.5, 3.0)
    for ci, (sym, n_beats) in enumerate(chords):
        p = parsed[ci]
        if p is None:
            abs_beat += n_beats
            continue
        root_pc, q = p
        tones = _chord_tones(root_pc, q)
        voicing = sorted(_clamp(60 + pc, PIANO_LO, PIANO_HI) for pc in tones)
        for bar_offset in range(0, n_beats, 4):
            bar_beats = min(4, n_beats - bar_offset)
            for off in montuno_offsets:
                if off < bar_beats:
                    t = (abs_beat + bar_offset + off) * beat_dur
                    _piano_hit(piano_inst, voicing, t, beat_dur * 0.45, rng)
        abs_beat += n_beats

    # Drums: cascara on ride + clave (2-3 son) + cowbell on 1
    cascara_offsets = (0.0, 0.75, 1.5, 2.0, 2.75, 3.5)  # cascara feel
    for beat0 in range(0, total_beats, 4):
        bar_start = beat0 * beat_dur
        for off in cascara_offsets:
            if off < min(4, total_beats - beat0):
                t = bar_start + off * beat_dur
                drum_inst.notes.append(pretty_midi.Note(
                    velocity=70, pitch=RIDE, start=t, end=t + beat_dur * 0.3))
    # 2-3 son clave (2 bars): bar0 = beats 2.5, 4 ; bar1 = beats 1, 2.5, 4
    abs_beat = 0
    total_bars = (total_beats + 3) // 4
    for bar in range(total_bars):
        bar_start = bar * 4 * beat_dur
        if bar % 2 == 0:
            offsets = (1.5, 3.0)
        else:
            offsets = (0.0, 1.5, 2.5)
        for off in offsets:
            t = bar_start + off * beat_dur
            drum_inst.notes.append(pretty_midi.Note(
                velocity=80, pitch=SIDESTICK, start=t, end=t + beat_dur * 0.2))
        drum_inst.notes.append(pretty_midi.Note(
            velocity=70, pitch=COWBELL, start=bar_start, end=bar_start + beat_dur * 0.3))


# ---------------------------------------------------------------------------
# Style: funk
# ---------------------------------------------------------------------------

def _funk(chords, beat_dur, beats_per_bar, piano_inst, bass_inst, drum_inst, rng):
    BASS_LO, BASS_HI = 36, 57
    PIANO_LO, PIANO_HI = 55, 76
    total_beats = sum(b for _, b in chords)
    parsed = [_parse_chord(s) for s, _ in chords]

    # Bass: root on 1, octave on 2.5, fifth on 4 + ghost notes
    abs_beat = 0
    for ci, (sym, n_beats) in enumerate(chords):
        p = parsed[ci]
        if p is None:
            root_pc, fifth_pc = 0, 7
        else:
            root_pc, q = p
            tones = _chord_tones(root_pc, q)
            fifth_pc = tones[2]
        root_p = _clamp(36 + root_pc, BASS_LO, BASS_HI)
        oct_up = _clamp(root_p + 12, BASS_LO, BASS_HI)
        fifth_p = _clamp(36 + fifth_pc, BASS_LO, BASS_HI)
        for bar_offset in range(0, n_beats, 4):
            bar_beats = min(4, n_beats - bar_offset)
            tb = (abs_beat + bar_offset) * beat_dur
            bass_inst.notes.append(pretty_midi.Note(
                velocity=95, pitch=root_p, start=tb, end=tb + beat_dur * 0.4))
            # ghost on & of 1
            if rng.random() < 0.5:
                bass_inst.notes.append(pretty_midi.Note(
                    velocity=35, pitch=root_p,
                    start=tb + beat_dur * 0.5, end=tb + beat_dur * 0.7))
            if bar_beats >= 3:
                t = tb + 2.5 * beat_dur
                bass_inst.notes.append(pretty_midi.Note(
                    velocity=85, pitch=oct_up, start=t, end=t + beat_dur * 0.4))
            if bar_beats >= 4:
                t = tb + 3.0 * beat_dur
                bass_inst.notes.append(pretty_midi.Note(
                    velocity=80, pitch=fifth_p, start=t, end=t + beat_dur * 0.6))
        abs_beat += n_beats

    # Piano: 16th-note offbeat comping (hits on the 'e' and 'a' of select beats)
    abs_beat = 0
    funk_offsets = (0.75, 1.25, 2.75, 3.25)
    for ci, (sym, n_beats) in enumerate(chords):
        p = parsed[ci]
        if p is None:
            abs_beat += n_beats
            continue
        root_pc, q = p
        tones = _chord_tones(root_pc, q)
        voicing = sorted(_clamp(60 + pc, PIANO_LO, PIANO_HI) for pc in tones[1:])
        for bar_offset in range(0, n_beats, 4):
            bar_beats = min(4, n_beats - bar_offset)
            for off in funk_offsets:
                if off < bar_beats:
                    t = (abs_beat + bar_offset + off) * beat_dur
                    _piano_hit(piano_inst, voicing, t, beat_dur * 0.2, rng)
        abs_beat += n_beats

    # Drums: backbeat — kick 1 + &-of-2, snare 2 & 4, closed hat 16ths
    for beat in range(total_beats):
        t = beat * beat_dur
        beat_in_bar = beat % 4
        # 16th hi-hat
        for sub in (0.0, 0.25, 0.5, 0.75):
            ts = t + sub * beat_dur
            vel = 60 if sub in (0.0, 0.5) else 45
            drum_inst.notes.append(pretty_midi.Note(
                velocity=vel + rng.randint(-3, 3), pitch=CLOSED_HH,
                start=ts, end=ts + beat_dur * 0.2))
        if beat_in_bar == 0:
            drum_inst.notes.append(pretty_midi.Note(
                velocity=95, pitch=KICK, start=t, end=t + beat_dur * 0.4))
        if beat_in_bar == 2:
            drum_inst.notes.append(pretty_midi.Note(
                velocity=85, pitch=KICK,
                start=t + 0.5 * beat_dur, end=t + 0.9 * beat_dur))
        if beat_in_bar in (1, 3):
            drum_inst.notes.append(pretty_midi.Note(
                velocity=90, pitch=SNARE, start=t, end=t + beat_dur * 0.4))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_STYLES = {
    "swing":  _swing,
    "bossa":  _bossa,
    "ballad": _ballad,
    "latin":  _latin,
    "funk":   _funk,
}


def generate_rhythm_section(
    chords: List[Tuple[str, int]],
    tempo_bpm: float,
    time_signature: Tuple[int, int] = (4, 4),
    style: str = "swing",
    seed: int = 42,
) -> List[pretty_midi.Instrument]:
    """Generate piano, bass, and drum tracks for a chord progression.

    Args:
        chords: list of (chord_symbol, n_beats) pairs.
        tempo_bpm: tempo in beats per minute (drives the time grid).
        time_signature: (numerator, denominator); only numerator (beats_per_bar) is used.
        style: one of 'swing', 'bossa', 'ballad', 'latin', 'funk'.
        seed: integer seed for the local RNG (humanization only — pitches and
              timing are deterministic without it).

    Returns:
        [piano, bass, drums] pretty_midi.Instrument objects.

    Determinism: same chords + tempo_bpm + style + seed produces byte-identical
    instrument output. No global random state is touched.
    """
    if style not in _STYLES:
        raise ValueError(f"Unknown style {style!r}; choices: {sorted(_STYLES)}")
    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be positive")
    beats_per_bar = time_signature[0]
    beat_dur = 60.0 / float(tempo_bpm)

    piano = pretty_midi.Instrument(program=0,  name="Piano")
    bass  = pretty_midi.Instrument(program=32, name="Bass")
    drums = pretty_midi.Instrument(program=0,  is_drum=True, name="Drums")

    rng = random.Random(seed)
    _STYLES[style](list(chords), beat_dur, beats_per_bar, piano, bass, drums, rng)

    # Stable ordering for reproducibility across Python versions.
    for inst in (piano, bass, drums):
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
    return [piano, bass, drums]
