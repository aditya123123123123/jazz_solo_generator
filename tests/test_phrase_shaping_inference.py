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
