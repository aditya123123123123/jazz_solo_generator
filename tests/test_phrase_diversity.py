from src.generation import generate_solo as gen
from src.generation.phrase_features import PhraseFeature
from tests.test_whole_progression_phrase_planning import make_common


def feature(token, notes=8, density=2.0, rest=0.1, contour="descending", final_chord_tone_rate=0.0):
    return PhraseFeature(token, 4.0, notes, density, 60.0, 8.0, rest, contour, True, final_chord_tone_rate)


def test_diversify_phrase_plan_limits_overused_phrase_with_similar_contour_alternatives():
    common = make_common((56, 56, 56, 56, 56))  # PHRASE_53 repeated
    phrase_tok = common["phrase_tok"]
    features = {
        "PHRASE_53": feature("PHRASE_53", notes=8, density=2.0, contour="descending"),
        "PHRASE_42": feature("PHRASE_42", notes=9, density=2.1, contour="descending"),
        "PHRASE_26": feature("PHRASE_26", notes=18, density=4.0, contour="descending"),
        "PHRASE_24": feature("PHRASE_24", notes=11, density=2.8, contour="ascending"),
    }

    diversified, changed = gen._diversify_phrase_plan(
        [phrase_tok.encode("PHRASE_53")] * 5,
        phrase_tok,
        features,
        max_uses=2,
    )
    labels = [phrase_tok.decode(t) for t in diversified]

    assert changed == 3
    assert labels.count("PHRASE_53") == 2
    assert labels[:2] == ["PHRASE_53", "PHRASE_53"]
    assert set(labels[2:]).issubset({"PHRASE_42", "PHRASE_26"})
    assert "PHRASE_24" not in labels


def test_diversify_phrase_plan_can_preserve_cadence_profile_before_feature_distance():
    common = make_common((56, 56, 56))  # PHRASE_53 repeated
    phrase_tok = common["phrase_tok"]
    features = {
        "PHRASE_53": feature("PHRASE_53", notes=8, density=2.0, contour="descending", final_chord_tone_rate=0.85),
        # Feature-nearest candidate, but cadence profile is much worse.
        "PHRASE_42": feature("PHRASE_42", notes=8, density=2.0, contour="descending", final_chord_tone_rate=0.10),
        # Slightly farther in feature space, but similar final chord-tone behavior.
        "PHRASE_26": feature("PHRASE_26", notes=9, density=2.1, contour="descending", final_chord_tone_rate=0.82),
    }

    diversified, changed = gen._diversify_phrase_plan(
        [phrase_tok.encode("PHRASE_53")] * 3,
        phrase_tok,
        features,
        max_uses=2,
        preserve_cadence=True,
    )
    labels = [phrase_tok.decode(t) for t in diversified]

    assert changed == 1
    assert labels == ["PHRASE_53", "PHRASE_53", "PHRASE_26"]


def test_generate_solo_applies_phrase_diversity_to_plan_and_executor_calls():
    common = make_common((56, 56, 56, 56))  # PHRASE_53 repeated
    features = {
        "PHRASE_53": feature("PHRASE_53", notes=8, density=2.0, contour="descending"),
        "PHRASE_42": feature("PHRASE_42", notes=9, density=2.1, contour="descending"),
        "PHRASE_26": feature("PHRASE_26", notes=18, density=4.0, contour="descending"),
    }

    _notes, summaries, _unknowns = gen.generate_solo(
        [("F7", 4), ("Bb7", 4), ("F7", 4), ("C7", 4)],
        phrase_features=features,
        phrase_shaping=True,
        phrase_diversity=True,
        phrase_max_uses=2,
        **common,
    )

    labels = [s["phrase_token"] for s in summaries]
    assert labels.count("PHRASE_53") == 2
    assert max(labels.count(label) for label in set(labels)) <= 2
    assert summaries[0]["phrase_plan"] == labels
    assert sum(s["phrase_diversity_remapped"] for s in summaries) == 2
    executor_phrase_labels = [common["phrase_tok"].decode(call["phrase_id"].item()) for call in common["executor"].calls]
    assert executor_phrase_labels == labels
