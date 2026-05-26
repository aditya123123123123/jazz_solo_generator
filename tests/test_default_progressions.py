from src.generation.generate_solo import PROGRESSIONS


def test_default_generation_includes_eight_audition_progressions():
    assert len(PROGRESSIONS) >= 8


def test_default_generation_includes_expanded_song_probes():
    expected = {
        "rhythm_changes_Bb",
        "all_the_things_you_are",
        "giant_steps_cycle",
        "minor_blues_C",
        "modal_so_what_Dm",
    }

    assert expected.issubset(PROGRESSIONS)


def test_added_progressions_have_valid_section_shapes():
    for name in [
        "rhythm_changes_Bb",
        "all_the_things_you_are",
        "giant_steps_cycle",
        "minor_blues_C",
        "modal_so_what_Dm",
    ]:
        progression = PROGRESSIONS[name]
        assert len(progression) >= 4
        assert sum(beats for _chord, beats in progression) >= 16
        assert all(isinstance(chord, str) and chord for chord, _beats in progression)
        assert all(beats > 0 for _chord, beats in progression)
