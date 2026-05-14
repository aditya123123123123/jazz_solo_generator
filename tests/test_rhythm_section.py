"""Tests for the rhythm-section generator and JSON-output invariance."""
import json
from pathlib import Path

import pretty_midi
import pytest

from src.generation.rhythm_section import generate_rhythm_section


REPO = Path(__file__).parents[1]
FIXTURES = REPO / "tests" / "fixtures"

PROG_ii_V_I = [("Dm7", 4), ("G7", 4), ("Cj7", 4)]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def _instr_signature(inst):
    return [
        (n.pitch, round(n.start, 9), round(n.end, 9), n.velocity)
        for n in inst.notes
    ]


@pytest.mark.parametrize("style", ["swing", "bossa", "ballad", "latin", "funk"])
def test_determinism(style):
    a = generate_rhythm_section(PROG_ii_V_I, tempo_bpm=120, style=style, seed=42)
    b = generate_rhythm_section(PROG_ii_V_I, tempo_bpm=120, style=style, seed=42)
    assert len(a) == len(b) == 3
    for ia, ib in zip(a, b):
        assert _instr_signature(ia) == _instr_signature(ib)


def test_different_seeds_differ_via_humanization():
    """Two distinct seeds should differ at least in velocity jitter (swing)."""
    a = generate_rhythm_section(PROG_ii_V_I, tempo_bpm=120, style="swing", seed=1)
    b = generate_rhythm_section(PROG_ii_V_I, tempo_bpm=120, style="swing", seed=999)
    assert _instr_signature(a[1]) != _instr_signature(b[1])


# ---------------------------------------------------------------------------
# Musical correctness: swing bass root on each chord start
# ---------------------------------------------------------------------------

def test_swing_bass_root_on_chord_start():
    tempo = 120
    beat_dur = 60.0 / tempo
    insts = generate_rhythm_section(PROG_ii_V_I, tempo_bpm=tempo, style="swing", seed=42)
    _, bass, _ = insts

    chord_starts = [0.0, 4 * beat_dur, 8 * beat_dur]
    expected_root_pc = {0: 2, 1: 7, 2: 0}   # D, G, C

    for ci, t_start in enumerate(chord_starts):
        notes_here = [n for n in bass.notes if abs(n.start - t_start) < 1e-6]
        assert notes_here, f"no bass note at chord {ci} start"
        assert notes_here[0].pitch % 12 == expected_root_pc[ci], (
            f"chord {ci}: expected root pc {expected_root_pc[ci]}, "
            f"got pitch {notes_here[0].pitch}"
        )


# ---------------------------------------------------------------------------
# Multitrack export (Solo + Piano + Bass + Drums)
# ---------------------------------------------------------------------------

def test_multitrack_export(tmp_path):
    insts = generate_rhythm_section(PROG_ii_V_I, tempo_bpm=120, style="swing", seed=42)

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    solo = pretty_midi.Instrument(program=0, name="Solo")
    solo.notes.append(pretty_midi.Note(velocity=64, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(solo)
    pm.instruments.extend(insts)

    out = tmp_path / "test_multitrack.mid"
    pm.write(str(out))

    loaded = pretty_midi.PrettyMIDI(str(out))
    assert len(loaded.instruments) >= 4
    assert any(i.is_drum for i in loaded.instruments)
    names = {i.name for i in loaded.instruments}
    assert {"Solo", "Piano", "Bass", "Drums"}.issubset(names)


# ---------------------------------------------------------------------------
# Unknown-chord fallback
# ---------------------------------------------------------------------------

def test_unknown_chord_fallback():
    bad = [("Xyz123", 4), ("###", 4)]
    insts = generate_rhythm_section(bad, tempo_bpm=120, style="swing", seed=42)
    piano, bass, drums = insts
    assert len(bass.notes) >= 1
    assert len(drums.notes) >= 1
    # Piano is silent for unparseable chords (no shell voicing emitted).
    assert len(piano.notes) == 0


# ---------------------------------------------------------------------------
# JSON-output byte identity (regression guard against RNG leaks)
# ---------------------------------------------------------------------------

def test_json_export_byte_identical_to_baseline(tmp_path):
    """Re-running export_json with the synthetic baseline input must produce
    bytes identical to the snapshot captured before the rhythm-section work.

    If this fails, something in the JSON path has shifted (likely a global
    RNG leak from the new code, or unintended changes to export_json/helpers).
    """
    from src.generation.generate_solo import export_json

    # Touch the rhythm-section module before exporting JSON to surface any
    # import-time global-state mutation.
    _ = generate_rhythm_section(PROG_ii_V_I, tempo_bpm=120, style="swing", seed=42)

    note_events = [
        (60, 0.5, False),
        (62, 0.25, False),
        (0,  0.25, True),
        (64, 1.0,  False),
        (-1, 0.5,  False),
    ]
    summaries = [
        {
            "chord": "Dm7", "wjazz": "D-7", "beats": 4,
            "in_vocab": True, "phrase_token": "PHRASE_18",
            "phrase_remapped": True, "phrase_original": "PHRASE_00",
            "n_notes": 16, "n_clamped": 2, "pitches": [60, 62, 64],
        },
        {
            "chord": "G7", "wjazz": "G7", "beats": 4,
            "in_vocab": True, "phrase_token": "PHRASE_57",
            "phrase_remapped": False, "phrase_original": None,
            "n_notes": 16, "n_clamped": 0, "pitches": [55, 59, 62, 65],
        },
    ]
    out = tmp_path / "regen.json"
    export_json(note_events, summaries, out)

    baseline_bytes = (FIXTURES / "json_baseline.json").read_bytes()
    new_bytes = out.read_bytes()
    assert new_bytes == baseline_bytes, (
        "JSON export bytes differ from baseline — JSON schema/behavior changed."
    )
