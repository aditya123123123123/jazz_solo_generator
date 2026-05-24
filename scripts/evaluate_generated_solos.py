#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.generation.chord_utils import chord_tones, parse_chord

VERSIONS = {
    "v6.2.0": REPO / "outputs" / "solos_v620_eval",
    "v6.3.0": REPO / "outputs" / "solos_v630_eval",
    "v6.3.1": REPO / "outputs" / "solos_v631_eval",
}


def chord_tone_pitch_classes(symbol: str) -> set[int]:
    parsed = parse_chord(symbol)
    if parsed is None:
        return set()
    return set(chord_tones(*parsed))


def approach_tone_pitch_classes(symbol: str) -> set[int]:
    tones = chord_tone_pitch_classes(symbol)
    return {((pc - 1) % 12) for pc in tones} | {((pc + 1) % 12) for pc in tones}


def score_json(path: Path):
    data = json.loads(path.read_text())
    total = chord_hits = approach_hits = chord_or_approach_hits = 0
    section_rows = []
    for section in data["sections"]:
        chord = section["chord"]
        tones = chord_tone_pitch_classes(chord)
        approaches = approach_tone_pitch_classes(chord)
        pitches = [int(p) for p in section.get("pitches", []) if 0 <= int(p) <= 127]
        sec_total = len(pitches)
        sec_chord = sum((p % 12) in tones for p in pitches)
        sec_approach = sum((p % 12) in approaches for p in pitches)
        sec_either = sum(((p % 12) in tones) or ((p % 12) in approaches) for p in pitches)
        total += sec_total
        chord_hits += sec_chord
        approach_hits += sec_approach
        chord_or_approach_hits += sec_either
        section_rows.append({
            "chord": chord,
            "notes": sec_total,
            "chord_tone_hit_rate": sec_chord / sec_total if sec_total else None,
            "chord_or_approach_hit_rate": sec_either / sec_total if sec_total else None,
        })
    if total == 0:
        raise RuntimeError(f"no section pitches in {path}")
    return {
        "piece": path.stem,
        "notes": total,
        "chord_tone_hit_rate": chord_hits / total,
        "approach_hit_rate": approach_hits / total,
        "chord_or_approach_hit_rate": chord_or_approach_hits / total,
        "sections": section_rows,
    }


def weighted(rows, key) -> float:
    numer = sum(r[key] * r["notes"] for r in rows)
    denom = sum(r["notes"] for r in rows)
    return numer / denom


def main():
    all_results = {}
    for version, out_dir in VERSIONS.items():
        rows = [score_json(path) for path in sorted(out_dir.glob("*.json"))]
        if not rows:
            raise RuntimeError(f"no json outputs in {out_dir}")
        all_results[version] = rows

    report = []
    report.append("# Generated Solo Harmonic Evaluation")
    report.append("")
    report.append("Scoring method: exact section-based scoring from each generated JSON's `sections[].pitches`, so each generated pitch is compared to the chord whose section produced it. This avoids timing drift from generated note durations/rest injection.")
    report.append("")
    report.append("Generated artifacts:")
    for version, out_dir in VERSIONS.items():
        report.append(f"- {version}: `{out_dir.relative_to(REPO)}`")
    report.append("")
    report.append("## Aggregate")
    report.append("")
    report.append("| version | section pitches scored | chord-tone hit | chord+approach hit |")
    report.append("|---|---:|---:|---:|")
    for version, rows in all_results.items():
        notes = sum(r["notes"] for r in rows)
        report.append(
            f"| {version} | {notes} | {weighted(rows, 'chord_tone_hit_rate'):.3f} | "
            f"{weighted(rows, 'chord_or_approach_hit_rate'):.3f} |"
        )
    report.append("")
    report.append("## Per tune")
    report.append("")
    report.append("| version | tune | section pitches scored | chord-tone hit | chord+approach hit |")
    report.append("|---|---|---:|---:|---:|")
    for version, rows in all_results.items():
        for r in rows:
            report.append(
                f"| {version} | {r['piece']} | {r['notes']} | {r['chord_tone_hit_rate']:.3f} | "
                f"{r['chord_or_approach_hit_rate']:.3f} |"
            )
    report.append("")
    report.append("## Delta vs v6.3.0")
    report.append("")
    base = all_results["v6.3.0"]
    for version in ["v6.2.0", "v6.3.1"]:
        rows = all_results[version]
        report.append(
            f"- {version}: chord-tone Δ={weighted(rows, 'chord_tone_hit_rate') - weighted(base, 'chord_tone_hit_rate'):+.3f}; "
            f"chord+approach Δ={weighted(rows, 'chord_or_approach_hit_rate') - weighted(base, 'chord_or_approach_hit_rate'):+.3f}"
        )
    out = REPO / "outputs" / "solo_eval_report.md"
    out.write_text("\n".join(report) + "\n")
    print(out)
    print("\n".join(report))


if __name__ == "__main__":
    main()
