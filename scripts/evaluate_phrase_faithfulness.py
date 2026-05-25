#!/usr/bin/env python3
"""Evaluate phrase-faithfulness of generated solo JSON traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.generation.chord_utils import chord_tones, parse_chord


def _abs_error(generated: Any, target: Any) -> float | None:
    if generated is None or target is None:
        return None
    return abs(float(generated) - float(target))


def _cadence_is_chord_tone(section: dict[str, Any], metrics: dict[str, Any]) -> bool | None:
    if "final_note_chord_tone" in metrics:
        return bool(metrics["final_note_chord_tone"])
    final_pitch = metrics.get("final_pitch")
    if final_pitch is None:
        return None
    parsed = parse_chord(section.get("chord", ""))
    if parsed is None:
        return None
    return int(final_pitch) % 12 in set(chord_tones(*parsed))


def evaluate_section(section: dict[str, Any]) -> dict[str, Any]:
    features = section.get("phrase_features") or {}
    metrics = section.get("generated_phrase_metrics") or {}
    contour_match = None
    if features.get("contour") is not None and metrics.get("contour") is not None:
        contour_match = metrics.get("contour") == features.get("contour")
    cadence = _cadence_is_chord_tone(section, metrics)
    return {
        "chord": section.get("chord"),
        "phrase": section.get("phrase") or section.get("phrase_token"),
        "target_contour": features.get("contour"),
        "generated_contour": metrics.get("contour"),
        "contour_match": contour_match,
        "density_error": _abs_error(metrics.get("density"), features.get("median_density")),
        "note_count_error": _abs_error(metrics.get("num_notes"), features.get("median_num_notes")),
        "pitch_range_error": _abs_error(metrics.get("pitch_range"), features.get("median_pitch_range")),
        "rest_ratio_error": _abs_error(metrics.get("rest_ratio"), features.get("median_rest_ratio")),
        "cadence_resolution": cadence,
    }


def _rate(values: list[bool | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(1 for v in clean if v) / len(clean)


def _avg(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return mean(clean) if clean else None


def evaluate_solo_json(path: Path) -> dict[str, Any] | None:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "sections" not in data:
        return None
    sections = [evaluate_section(s) for s in data.get("sections", [])]
    return {
        "file": str(path),
        "name": data.get("name", path.stem),
        "sections": sections,
        "summary": summarize_sections(sections),
    }


def summarize_sections(sections: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sections": len(sections),
        "contour_match_rate": _rate([s.get("contour_match") for s in sections]),
        "cadence_resolution_rate": _rate([s.get("cadence_resolution") for s in sections]),
        "mean_density_error": _avg([s.get("density_error") for s in sections]),
        "mean_note_count_error": _avg([s.get("note_count_error") for s in sections]),
        "mean_pitch_range_error": _avg([s.get("pitch_range_error") for s in sections]),
        "mean_rest_ratio_error": _avg([s.get("rest_ratio_error") for s in sections]),
    }


def evaluate_solos_dir(solos_dir: Path | str) -> dict[str, Any]:
    solos_path = Path(solos_dir)
    solos = [s for s in (evaluate_solo_json(p) for p in sorted(solos_path.glob("*.json"))) if s is not None]
    all_sections = [section for solo in solos for section in solo["sections"]]
    return {
        "solos_dir": str(solos_path),
        "solos": solos,
        "aggregate": summarize_sections(all_sections),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_report(result: dict[str, Any]) -> str:
    agg = result["aggregate"]
    lines = [
        "# Phrase Faithfulness Report",
        "",
        f"Solos dir: `{result['solos_dir']}`",
        "",
        "## Aggregate",
        "",
        f"- Sections: {_fmt(agg['sections'])}",
        f"- Contour match rate: {_fmt(agg['contour_match_rate'])}",
        f"- Cadence resolution rate: {_fmt(agg['cadence_resolution_rate'])}",
        f"- Mean density error: {_fmt(agg['mean_density_error'])}",
        f"- Mean note-count error: {_fmt(agg['mean_note_count_error'])}",
        f"- Mean pitch-range error: {_fmt(agg['mean_pitch_range_error'])}",
        f"- Mean rest-ratio error: {_fmt(agg['mean_rest_ratio_error'])}",
        "",
        "## Per solo",
        "",
        "| Solo | Sections | Contour | Cadence | Notes err | Density err | Range err | Rest err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for solo in result["solos"]:
        s = solo["summary"]
        lines.append(
            f"| {solo['name']} | {_fmt(s['sections'])} | {_fmt(s['contour_match_rate'])} | "
            f"{_fmt(s['cadence_resolution_rate'])} | {_fmt(s['mean_note_count_error'])} | "
            f"{_fmt(s['mean_density_error'])} | {_fmt(s['mean_pitch_range_error'])} | "
            f"{_fmt(s['mean_rest_ratio_error'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(result: dict[str, Any], md_path: Path | str, json_path: Path | str) -> None:
    md = Path(md_path)
    js = Path(json_path)
    md.parent.mkdir(parents=True, exist_ok=True)
    js.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(markdown_report(result))
    js.write_text(json.dumps(result, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate generated phrase faithfulness.")
    parser.add_argument("--solos-dir", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, default=Path("outputs/phrase_faithfulness_v64.md"))
    parser.add_argument("--json-out", type=Path, default=Path("outputs/phrase_faithfulness_v64.json"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = evaluate_solos_dir(args.solos_dir)
    write_reports(result, args.md_out, args.json_out)
    print(f"Wrote {args.md_out} and {args.json_out}")


if __name__ == "__main__":
    main()
