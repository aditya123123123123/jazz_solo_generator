import json

from src.generation.generate_solo import export_json


def test_export_json_includes_top_level_plan_features_and_generated_metrics(tmp_path):
    summaries = [
        {
            "chord": "Cj7",
            "model_chord": "Cj7",
            "chord_condition": "normal",
            "beats": 4,
            "phrase_token": "PHRASE_43",
            "phrase_plan": ["PHRASE_43"],
            "phrase_features": {
                "contour": "ascending",
                "median_num_notes": 28,
                "median_density": 2.9,
                "median_rest_ratio": 0.19,
                "median_pitch_range": 12,
            },
            "generated_phrase_metrics": {
                "contour": "ascending",
                "num_notes": 16,
                "density": 4.0,
                "rest_ratio": 0.0,
                "pitch_range": 15,
            },
            "n_notes": 16,
            "n_clamped": 0,
            "dur_fallback": 0,
            "pitches": [60, 64, 67, 75],
            "phrase_remapped": False,
            "phrase_original": None,
        }
    ]
    out = tmp_path / "solo.json"

    export_json([(60, 0.5, False)], summaries, out, name="probe", tempo_bpm=120)
    data = json.loads(out.read_text())

    assert data["phrase_plan"] == ["PHRASE_43"]
    section = data["sections"][0]
    assert section["phrase_features"]["contour"] == "ascending"
    assert section["phrase_features"]["median_num_notes"] == 28
    assert section["generated_phrase_metrics"]["num_notes"] == 16
    assert section["generated_phrase_metrics"]["pitch_range"] == 15
