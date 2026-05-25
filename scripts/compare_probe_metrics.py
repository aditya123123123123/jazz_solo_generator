#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def chord_pcs(chord: str) -> tuple[set[int], set[int]]:
    match = re.match(r"([A-G](?:b|#)?)(.*)", chord.replace("j", "maj"))
    if not match:
        return set(), set()
    root = ROOT_PC[match.group(1)]
    quality = match.group(2)
    if "m7b5" in quality:
        intervals = [0, 3, 6, 10]
    elif quality.startswith("m") and "maj" not in quality:
        intervals = [0, 3, 7, 10]
    elif "7" in quality and "maj" not in quality:
        intervals = [0, 4, 7, 10]
    elif "maj" in quality:
        intervals = [0, 4, 7, 11]
    else:
        intervals = [0, 4, 7]
    pure = {(root + interval) % 12 for interval in intervals}
    color = {(root + interval) % 12 for interval in intervals + [2, 5, 9]}
    return pure, color


def analyze(solos_dir: str | Path) -> dict[str, dict]:
    out = {}
    for path in sorted(Path(solos_dir).glob("*.json")):
        data = json.loads(path.read_text())
        chord = color = outside = total = 0
        pitches: list[int] = []
        diversity_remaps = 0
        for section in data.get("sections", []):
            pure, color_set = chord_pcs(section.get("chord", ""))
            diversity_remaps += int(bool(section.get("phrase_diversity_remapped")))
            for pitch in section.get("pitches", []):
                pitch = int(pitch)
                pitches.append(pitch)
                pc = pitch % 12
                total += 1
                if pc in pure:
                    chord += 1
                elif pc in color_set:
                    color += 1
                else:
                    outside += 1
        leaps = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
        plan = data.get("phrase_plan", [])
        out[data.get("name", path.stem)] = {
            "notes": len(pitches),
            "octave_leaps": sum(leap >= 12 for leap in leaps),
            "max_leap": max(leaps) if leaps else 0,
            "mean_leap": round(sum(leaps) / len(leaps), 2) if leaps else 0,
            "outside_pct": round(outside / total * 100, 1) if total else None,
            "pure_pct": round(chord / total * 100, 1) if total else None,
            "color_pct": round(color / total * 100, 1) if total else None,
            "phrase_repeat_max": max([plan.count(p) for p in set(plan)] or [0]),
            "phrase_total": len(plan),
            "phrase_diversity_remaps": diversity_remaps,
            "phrase_plan": plan,
        }
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: compare_probe_metrics.py OUT.json NAME=DIR [NAME=DIR ...]", file=sys.stderr)
        return 2
    out_path = Path(argv[1])
    result = {}
    for item in argv[2:]:
        name, directory = item.split("=", 1)
        result[name] = analyze(directory)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
