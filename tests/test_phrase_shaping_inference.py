from src.generation.phrase_features import PhraseFeature
from tests.test_whole_progression_phrase_planning import make_common
from src.generation import generate_solo as gen


def test_phrase_shaping_uses_cluster_note_count_and_rest_ratio():
    common = make_common((13, 14))  # PHRASE_10, PHRASE_11
    features = {
        "PHRASE_10": PhraseFeature("PHRASE_10", 4, 9, 2.25, 60, 8, 0.05, "flat", True),
        "PHRASE_11": PhraseFeature("PHRASE_11", 4, 40, 10.0, 64, 16, 0.55, "ascending", False),
    }

    _notes, summaries, _unknowns = gen.generate_solo(
        [("Dm7", 4), ("G7", 4)],
        phrase_features=features,
        phrase_shaping=True,
        rest_boost=1.8,
        **common,
    )

    calls = common["executor"].calls
    assert calls[0]["n_notes"] == 9
    assert calls[1]["n_notes"] == 28  # safely clamped from 40
    assert calls[0]["rest_boost"] < 1.8
    assert calls[1]["rest_boost"] > 1.8
    assert summaries[0]["phrase_shaping"] is True
    assert summaries[1]["n_notes"] == 28


def test_default_generation_keeps_legacy_note_count_without_phrase_shaping():
    common = make_common((13,))
    features = {"PHRASE_10": PhraseFeature("PHRASE_10", 4, 9, 2.25, 60, 8, 0.05, "flat", True)}

    _notes, summaries, _unknowns = gen.generate_solo(
        [("Dm7", 4)],
        phrase_features=features,
        phrase_shaping=False,
        **common,
    )

    assert common["executor"].calls[0]["n_notes"] == 16
    assert summaries[0]["n_notes"] == 16


def test_rhythm_density_calibration_reactivates_sampled_rests_to_phrase_target():
    common = make_common((13,))  # PHRASE_10
    features = {"PHRASE_10": PhraseFeature("PHRASE_10", 4, 8, 2.0, 60, 8, 0.05, "flat", True)}

    def sparse_generate(_chord_ids, _phrase_id, _artist_id, n_notes=16, **_kwargs):
        assert n_notes == 8
        return [
            (60, 4, 0),
            (62, 4, 1),
            (64, 4, 1),
            (65, 4, 0),
            (67, 4, 1),
            (69, 4, 1),
            (71, 4, 1),
            (72, 4, 1),
        ]

    common["executor"].generate = sparse_generate
    notes, summaries, _unknowns = gen.generate_solo(
        [("Dm7", 4)],
        phrase_features=features,
        phrase_shaping=True,
        rhythm_density_calibration=True,
        **common,
    )

    sounding = [p for p, _d, r in notes if not r]
    assert len(sounding) == 8
    assert summaries[0]["rhythm_density_calibration"] is True
    assert summaries[0]["rhythm_density_calibration_adjusted"] == 6


def test_rhythm_density_calibration_is_opt_in():
    common = make_common((13,))
    features = {"PHRASE_10": PhraseFeature("PHRASE_10", 4, 6, 1.5, 60, 8, 0.05, "flat", True)}

    common["executor"].generate = lambda *_args, **_kwargs: [
        (60, 4, 0), (62, 4, 1), (64, 4, 1), (65, 4, 0), (67, 4, 1), (69, 4, 1)
    ]
    notes, summaries, _unknowns = gen.generate_solo(
        [("Dm7", 4)],
        phrase_features=features,
        phrase_shaping=True,
        rhythm_density_calibration=False,
        **common,
    )

    assert sum(1 for _p, _d, r in notes if not r) == 2
    assert summaries[0]["rhythm_density_calibration"] is False
    assert summaries[0]["rhythm_density_calibration_adjusted"] == 0


def test_section_cadence_enforcement_moves_only_final_sounding_pitch_to_chord_tone():
    common = make_common((13,))

    common["executor"].generate = lambda *_args, **_kwargs: [
        (61, 4, 0),  # Db, deliberately outside Dm7 but not final
        (65, 4, 1),  # rest with a valid decoded pitch; must remain a rest
        (64, 4, 0),  # E, outside Dm7 final; nearest shell tone is F
    ]
    notes, summaries, _unknowns = gen.generate_solo(
        [("Dm7", 4)],
        section_cadence_enforcement=True,
        **common,
    )

    assert notes == [(61, 0.25, False), (65, 0.25, True), (65, 0.25, False)]
    assert summaries[0]["section_cadence_enforcement"] is True
    assert summaries[0]["section_cadence_adjusted"] is True
    assert summaries[0]["generated_phrase_metrics"]["final_pitch"] == 65


def test_section_cadence_enforcement_can_preserve_generated_contour():
    common = make_common((13,))

    common["executor"].generate = lambda *_args, **_kwargs: [
        (48, 4, 0),
        (53, 4, 0),
        (53, 4, 0),  # F, outside Cj7; nearest E would flip contour to arch
    ]
    notes, summaries, _unknowns = gen.generate_solo(
        [("Cj7", 4)],
        section_cadence_enforcement=True,
        section_cadence_preserve_contour=True,
        **common,
    )

    assert notes == [(48, 0.25, False), (53, 0.25, False), (55, 0.25, False)]
    assert summaries[0]["section_cadence_preserve_contour"] is True
    assert summaries[0]["generated_phrase_metrics"]["contour"] == "ascending"


def test_section_cadence_enforcement_can_prefer_target_phrase_contour():
    common = make_common((13,))
    features = {
        "PHRASE_10": PhraseFeature("PHRASE_10", 4, 8, 2.0, 60, 8, 0.05, "descending", True)
    }

    common["executor"].generate = lambda *_args, **_kwargs: [
        (72, 4, 0),
        (70, 4, 0),
        (73, 4, 0),  # C#, outside Cj7; nearest C keeps flat generated contour
    ]
    notes, summaries, _unknowns = gen.generate_solo(
        [("Cj7", 4)],
        phrase_features=features,
        phrase_shaping=True,
        section_cadence_enforcement=True,
        section_cadence_target_contour=True,
        **common,
    )

    assert notes == [(84, 0.25, False), (70, 0.25, False), (60, 0.25, False)]
    assert summaries[0]["section_cadence_target_contour"] is True
    assert summaries[0]["generated_phrase_metrics"]["contour"] == "descending"


def test_bebop_approach_notes_put_weak_beat_chromatic_approach_before_chord_tone():
    common = make_common((13,))

    common["executor"].generate = lambda *_args, **_kwargs: [
        (60, 4, 0),  # C on beat 0.0, already a Cj7 chord tone
        (62, 4, 0),  # D on beat 0.5, weak beat; should become D# into E
        (64, 4, 0),  # E on beat 1.0, strong beat chord tone target
        (67, 4, 0),  # G on beat 1.5, chord tone but not a target position
    ]
    notes, summaries, _unknowns = gen.generate_solo(
        [("Cj7", 4)],
        bebop_approach_notes=True,
        **common,
    )

    assert notes == [
        (60, 0.25, False),
        (63, 0.25, False),
        (64, 0.25, False),
        (67, 0.25, False),
    ]
    assert summaries[0]["bebop_approach_notes"] is True
    assert summaries[0]["bebop_approach_adjusted"] == 1


def test_bebop_approach_notes_keep_rests_and_cadence_final_pitch_safe():
    common = make_common((13,))

    common["executor"].generate = lambda *_args, **_kwargs: [
        (60, 4, 0),
        (62, 4, 1),  # rest should not become a chromatic sounding note
        (64, 4, 0),
        (66, 4, 0),  # final outside pitch should still be cadence-enforced
    ]
    notes, summaries, _unknowns = gen.generate_solo(
        [("Cj7", 4)],
        bebop_approach_notes=True,
        section_cadence_enforcement=True,
        **common,
    )

    assert notes == [
        (60, 0.25, False),
        (62, 0.25, True),
        (64, 0.25, False),
        (67, 0.25, False),
    ]
    assert summaries[0]["bebop_approach_adjusted"] == 0
    assert summaries[0]["section_cadence_adjusted"] is True


def test_section_cadence_enforcement_is_opt_in():
    common = make_common((13,))
    common["executor"].generate = lambda *_args, **_kwargs: [(64, 4, 0)]

    notes, summaries, _unknowns = gen.generate_solo(
        [("Dm7", 4)],
        section_cadence_enforcement=False,
        **common,
    )

    assert notes == [(64, 0.25, False)]
    assert summaries[0]["section_cadence_enforcement"] is False
    assert summaries[0]["section_cadence_adjusted"] is False
