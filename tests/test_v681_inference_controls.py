from src.generation.phrase_features import PhraseFeature
from tests.test_whole_progression_phrase_planning import make_common
from src.generation import generate_solo as gen


def test_density_calibration_only_runs_below_min_ratio():
    common = make_common((13, 14))
    features = {
        "PHRASE_10": PhraseFeature("PHRASE_10", 4, 8, 2.0, 60, 8, 0.05, "flat", True),
        "PHRASE_11": PhraseFeature("PHRASE_11", 4, 8, 2.0, 60, 8, 0.05, "flat", True),
    }
    calls = {"n": 0}

    def generate(_chord_ids, _phrase_id, _artist_id, n_notes=16, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # 5/8 = 62.5% of target; should not calibrate when threshold is 60%.
            return [(60 + i, 4, 0 if i < 5 else 1) for i in range(8)]
        # 4/8 = 50% of target; should calibrate up to target.
        return [(60 + i, 4, 0 if i < 4 else 1) for i in range(8)]

    common["executor"].generate = generate
    notes, summaries, _unknowns = gen.generate_solo(
        [("Dm7", 4), ("G7", 4)],
        phrase_features=features,
        phrase_shaping=True,
        rhythm_density_calibration=True,
        rhythm_density_calibration_min_ratio=0.6,
        **common,
    )

    first_sounding = sum(1 for _p, _d, r in notes[:8] if not r)
    second_sounding = sum(1 for _p, _d, r in notes[8:] if not r)
    assert first_sounding == 5
    assert second_sounding == 8
    assert summaries[0]["rhythm_density_calibration_adjusted"] == 0
    assert summaries[1]["rhythm_density_calibration_adjusted"] == 4


def test_register_continuity_overcorrection_resamples_once():
    common = make_common((13,))
    features = {"PHRASE_10": PhraseFeature("PHRASE_10", 4, 6, 1.5, 60, 8, 0.05, "flat", True)}
    calls = {"n": 0}

    def generate(_chord_ids, _phrase_id, _artist_id, n_notes=16, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [(96, 4, 0), (48, 4, 0), (96, 4, 0), (48, 4, 0), (96, 4, 0), (48, 4, 0)]
        return [(60, 4, 0), (62, 4, 0), (64, 4, 0), (65, 4, 0), (67, 4, 0), (69, 4, 0)]

    common["executor"].generate = generate
    notes, summaries, _unknowns = gen.generate_solo(
        [("Dm7", 4)],
        phrase_features=features,
        phrase_shaping=True,
        register_continuity=True,
        register_continuity_max_adjustments=2,
        register_continuity_resample_attempts=1,
        **common,
    )

    assert calls["n"] == 2
    assert [p for p, _d, r in notes if not r] == [60, 62, 64, 65, 67, 69]
    assert summaries[0]["register_continuity_resampled"] is True
    assert summaries[0]["register_continuity_attempts"] == 2


def test_repeat_guard_changes_excess_consecutive_pitch_repeats_to_chord_tones():
    events = [(60, 0.25, False), (60, 0.25, False), (60, 0.25, False), (60, 0.25, False)]

    guarded, changed = gen._apply_repeat_guard(events, "Cj7", max_repeats=2)

    assert changed == 1
    assert [p for p, _d, r in guarded] == [60, 60, 59, 60]
    assert all(not r for _p, _d, r in guarded)


def test_ii_v_i_bias_override_applies_only_to_ii_and_v_sections():
    common = make_common((13, 14, 15, 16))

    gen.generate_solo(
        [("Dm7", 4), ("G7", 4), ("Cj7", 4), ("F7", 4)],
        chord_tone_bias=True,
        chord_tone_bias_strength=1.5,
        ii_v_i_chord_tone_bias_strength=1.0,
        **common,
    )

    strengths = [call["chord_tone_bias_strength"] for call in common["executor"].calls]
    assert strengths == [1.0, 1.0, 1.5, 1.5]



def test_register_resample_keeps_lowest_edit_attempt_when_all_attempts_exceed_cap():
    common = make_common((13,))
    features = {"PHRASE_10": PhraseFeature("PHRASE_10", 4, 6, 1.5, 60, 8, 0.05, "flat", True)}
    calls = {"n": 0}

    def generate(_chord_ids, _phrase_id, _artist_id, n_notes=16, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Three octave folds after continuity.
            return [(96, 4, 0), (48, 4, 0), (96, 4, 0), (48, 4, 0), (60, 4, 0), (62, 4, 0)]
        # More folds than the first attempt; should not replace the better first attempt.
        return [(96, 4, 0), (48, 4, 0), (96, 4, 0), (48, 4, 0), (96, 4, 0), (48, 4, 0)]

    common["executor"].generate = generate
    notes, summaries, _unknowns = gen.generate_solo(
        [("Dm7", 4)],
        phrase_features=features,
        phrase_shaping=True,
        register_continuity=True,
        register_continuity_max_adjustments=2,
        register_continuity_resample_attempts=1,
        **common,
    )

    assert calls["n"] == 2
    assert summaries[0]["register_continuity_adjusted"] == 3
    assert [p for p, _d, r in notes if not r][-2:] == [84, 84]
