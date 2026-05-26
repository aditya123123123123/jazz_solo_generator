import pytest

from src.generation.generate_solo import _apply_timing_aware_harmony
from src.generation.chord_utils import chord_tones, parse_chord


def _pcs(events):
    return [int(p) % 12 for p, _d, r in events if not r]


def test_timing_aware_harmony_retunes_notes_by_actual_playback_chord():
    # The second note starts after the Dm7 section has ended. A C natural is not
    # in G7, so section-local scoring would miss the playback-time clash.
    note_events = [
        (62, 2.0, False),  # D over Dm7, 4 beats at 120 BPM
        (60, 0.5, False),  # C starts over G7: must be retuned to G7 tone
    ]
    progression = [("Dm7", 4), ("G7", 4)]

    adjusted, stats = _apply_timing_aware_harmony(
        note_events,
        progression,
        tempo_bpm=120,
        mode="strict_chord_tone",
    )

    assert stats["retuned"] == 1
    assert _pcs(adjusted)[0] in chord_tones(*parse_chord("Dm7"))
    assert _pcs(adjusted)[1] in chord_tones(*parse_chord("G7"))


def test_timing_aware_harmony_splits_sustained_notes_at_chord_boundaries():
    # One long E starts over Cmaj7 but sustains into F7, where E natural is not
    # a strict F7 chord tone. The postprocessor should split at the boundary and
    # retune only the later segment.
    note_events = [(64, 1.0, False)]  # 2 beats at 120 BPM
    progression = [("Cj7", 1), ("F7", 1)]

    adjusted, stats = _apply_timing_aware_harmony(
        note_events,
        progression,
        tempo_bpm=120,
        mode="strict_chord_tone",
    )

    assert stats["split_notes"] == 1
    assert len(adjusted) == 2
    assert adjusted[0][0] == 64
    assert adjusted[0][1] == pytest.approx(0.5)
    assert adjusted[1][1] == pytest.approx(0.5)
    assert int(adjusted[1][0]) % 12 in chord_tones(*parse_chord("F7"))


def test_timing_aware_harmony_keeps_rests_and_total_duration():
    note_events = [(60, 0.25, False), (0, 0.5, True), (61, 0.25, False)]
    adjusted, stats = _apply_timing_aware_harmony(
        note_events,
        [("Cj7", 4)],
        tempo_bpm=120,
        mode="strict_chord_tone",
    )

    assert sum(d for _p, d, _r in adjusted) == pytest.approx(1.0)
    assert [r for _p, _d, r in adjusted] == [False, True, False]
    assert stats["retuned"] == 1


def test_timing_aware_harmony_color_mode_keeps_safe_extensions_but_retunes_avoid_notes():
    note_events = [
        (62, 0.5, False),  # D = 9th over Cmaj7, safe color
        (65, 0.5, False),  # F = natural 11 over Cmaj7, avoid; should retune
    ]

    adjusted, stats = _apply_timing_aware_harmony(
        note_events,
        [("Cj7", 4)],
        tempo_bpm=120,
        mode="chord_color",
    )

    assert adjusted[0][0] == 62
    assert int(adjusted[1][0]) % 12 in {0, 2, 4, 7, 9, 11}
    assert int(adjusted[1][0]) % 12 != 5
    assert stats["retuned"] == 1
