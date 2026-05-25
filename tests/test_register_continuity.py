from src.generation import generate_solo as gen
from tests.test_whole_progression_phrase_planning import make_common


def test_nearest_octave_pitch_preserves_pitch_class_and_minimizes_leap():
    assert gen._nearest_octave_pitch(84, 60) == 60
    assert gen._nearest_octave_pitch(48, 83) == 84


def test_register_continuity_reduces_octave_teleports_without_changing_pitch_classes():
    events = [
        (60, 0.25, False),
        (84, 0.25, False),
        (50, 0.25, False),
        (0, 0.25, True),
        (77, 0.25, False),
    ]

    smoothed, final_pitch, adjusted = gen._apply_register_continuity(events)
    smoothed_pitches = [p for p, _d, r in smoothed if not r]

    assert [p % 12 for p in smoothed_pitches] == [0, 0, 2, 5]
    assert smoothed_pitches == [60, 60, 62, 65]
    assert final_pitch == 65
    assert adjusted == 3
    assert max(abs(b - a) for a, b in zip(smoothed_pitches, smoothed_pitches[1:])) <= 5


def test_generate_solo_records_register_continuity_adjustments():
    common = make_common((13,))

    class JumpingExecutor:
        def generate(self, chord_ids, phrase_id, artist_id, n_notes=16, **kwargs):
            return [(60, 4, 0), (84, 4, 0), (50, 4, 0), (77, 4, 0)]

    common["executor"] = JumpingExecutor()
    notes, summaries, _unknowns = gen.generate_solo(
        [("Dm7", 4)],
        register_continuity=True,
        **common,
    )

    sounding = [p for p, _d, r in notes if not r]
    assert summaries[0]["register_continuity"] is True
    assert summaries[0]["register_continuity_adjusted"] > 0
    assert summaries[0]["pitches"] == sounding
    assert max(abs(b - a) for a, b in zip(sounding, sounding[1:])) <= 5
