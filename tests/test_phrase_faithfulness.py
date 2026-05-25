import json

from pathlib import Path

from scripts.evaluate_phrase_faithfulness import evaluate_solos_dir, parse_args, write_reports


def test_phrase_faithfulness_evaluator_computes_aggregate_and_reports(tmp_path):
    solos_dir = tmp_path / "solos"
    solos_dir.mkdir()
    (solos_dir / "probe.json").write_text(json.dumps({
        "name": "probe",
        "sections": [
            {
                "chord": "Cj7",
                "phrase": "PHRASE_01",
                "phrase_features": {
                    "contour": "ascending",
                    "median_num_notes": 10,
                    "median_density": 2.5,
                    "median_rest_ratio": 0.2,
                    "median_pitch_range": 12,
                },
                "generated_phrase_metrics": {
                    "contour": "ascending",
                    "num_notes": 12,
                    "density": 3.0,
                    "rest_ratio": 0.25,
                    "pitch_range": 15,
                    "final_pitch": 64,
                    "final_note_chord_tone": True,
                },
            },
            {
                "chord": "G7",
                "phrase": "PHRASE_02",
                "phrase_features": {
                    "contour": "descending",
                    "median_num_notes": 8,
                    "median_density": 2.0,
                    "median_rest_ratio": 0.1,
                    "median_pitch_range": 8,
                },
                "generated_phrase_metrics": {
                    "contour": "ascending",
                    "num_notes": 6,
                    "density": 1.5,
                    "rest_ratio": 0.0,
                    "pitch_range": 6,
                    "final_pitch": 61,
                    "final_note_chord_tone": False,
                },
            },
        ],
    }))

    result = evaluate_solos_dir(solos_dir)

    assert result["aggregate"]["sections"] == 2
    assert result["aggregate"]["contour_match_rate"] == 0.5
    assert result["aggregate"]["cadence_resolution_rate"] == 0.5
    assert result["solos"][0]["sections"][0]["note_count_error"] == 2

    md_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    write_reports(result, md_path, json_path)
    assert "Phrase Faithfulness" in md_path.read_text()
    assert json.loads(json_path.read_text())["aggregate"]["sections"] == 2


def test_phrase_faithfulness_default_outputs_are_v64_named():
    args = parse_args(["--solos-dir", "outputs/example"])

    assert args.md_out == Path("outputs/phrase_faithfulness_v64.md")
    assert args.json_out == Path("outputs/phrase_faithfulness_v64.json")
