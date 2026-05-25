#!/usr/bin/env python3
"""Audit training phrases for bebop/blues jazz-vocabulary devices.

The goal is not to judge generated solos directly; it is to measure whether the
training corpus contains enough idiomatic material for the model to learn jazz
language natively: chromatic approaches, enclosures, guide-tone targets, blues
color, and offbeat/syncopated placement.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from src.generation.chord_utils import chord_tones, parse_chord

REPO = Path(__file__).parents[1]
DEFAULT_PHRASES = REPO / "data" / "processed" / "phrases_all_with_tempo.json"
DEFAULT_JSON = REPO / "outputs" / "jazz_vocabulary_audit_v1.json"
DEFAULT_MD = REPO / "outputs" / "jazz_vocabulary_audit_v1.md"

GUIDE_TONE_INTERVALS = {
    "maj7": (4, 11),
    "min7": (3, 10),
    "dom7": (4, 10),
    "halfdim": (3, 10),
    "dim7": (3, 9),
    "minmaj7": (3, 11),
}


def _pc(pitch: int) -> int:
    return int(pitch) % 12


def _pitch_distance_mod12(a: int, b: int) -> int:
    d = abs(_pc(a) - _pc(b)) % 12
    return min(d, 12 - d)


def _is_integer_beat(value: float, tolerance: float = 1e-6) -> bool:
    return abs(float(value) - round(float(value))) <= tolerance


def _is_weak_beat(note: dict) -> bool:
    beat = float(note.get("beat", 0.0) or 0.0)
    return not _is_integer_beat(beat)


def _is_strong_beat(note: dict) -> bool:
    beat = float(note.get("beat", 0.0) or 0.0)
    return _is_integer_beat(beat)


def _is_beat_boundary_pair(prev: dict, current: dict) -> bool:
    """Return True when prev behaves like a pickup into current's beat.

    WJazzD's `beat` field is often an integer beat label, not a fractional
    onset. A chromatic note right before the next beat therefore appears as a
    note whose beat label differs from the following note's beat label. Keep the
    fractional-beat path for synthetic/generated fixtures.
    """
    prev_beat = float(prev.get("beat", 0.0) or 0.0)
    cur_beat = float(current.get("beat", 0.0) or 0.0)
    if _is_weak_beat(prev) and _is_strong_beat(current):
        return True
    return prev_beat != cur_beat and _is_strong_beat(current)


def _forward_fill_chords(notes: list[dict]) -> list[str]:
    filled: list[str] = []
    current = ""
    for note in notes:
        chord = str(note.get("chord", "") or "").strip()
        if chord:
            current = chord
        filled.append(current)
    return filled


def _chord_info(chord: str):
    parsed = parse_chord(chord)
    if parsed is None:
        return None
    root_pc, quality = parsed
    tones = set(chord_tones(root_pc, quality))
    guide = {(root_pc + interval) % 12 for interval in GUIDE_TONE_INTERVALS.get(quality, ())}
    return root_pc, quality, tones, guide


def _is_chromatic_approach(prev_pitch: int, target_pitch: int, target_tones: set[int]) -> bool:
    return _pc(target_pitch) in target_tones and _pitch_distance_mod12(prev_pitch, target_pitch) == 1


def _is_enclosure(p1: int, p2: int, target: int, target_tones: set[int]) -> bool:
    if _pc(target) not in target_tones:
        return False
    lower = min(_pc(p1), _pc(p2))
    upper = max(_pc(p1), _pc(p2))
    target_pc = _pc(target)
    # Use signed melodic relation, not pitch-class numeric wrap, to avoid false
    # positives around C/B unless the line literally brackets the target pitch.
    return (p1 - target) * (p2 - target) < 0 and max(abs(p1 - target), abs(p2 - target)) <= 3


def audit_phrase(phrase: dict) -> dict:
    notes = sorted(phrase.get("notes", []), key=lambda n: (float(n.get("onset", 0.0)), int(n.get("eventid", 0))))
    filled_chords = _forward_fill_chords(notes)
    analyzable = 0
    chord_tone_landings = 0
    guide_tone_landings = 0
    dominant_notes = 0
    dominant_blues_color_notes = 0
    offbeat_notes = 0
    weak_to_strong_pairs = 0
    chromatic_approaches = 0
    enclosure_targets = 0
    enclosures = 0

    for i, note in enumerate(notes):
        info = _chord_info(filled_chords[i])
        if info is None:
            continue
        root_pc, quality, tones, guide = info
        pitch = int(note["pitch"])
        analyzable += 1
        if _pc(pitch) in tones:
            chord_tone_landings += 1
        if _pc(pitch) in guide:
            guide_tone_landings += 1
        if _is_weak_beat(note):
            offbeat_notes += 1
        if quality == "dom7":
            dominant_notes += 1
            blues_pcs = {
                (root_pc + 3) % 12,   # b3
                (root_pc + 6) % 12,   # b5
                (root_pc + 10) % 12,  # b7
            }
            if _pc(pitch) in blues_pcs:
                dominant_blues_color_notes += 1

        if i >= 1:
            prev = notes[i - 1]
            prev_info = _chord_info(filled_chords[i - 1])
            if prev_info is not None and _is_beat_boundary_pair(prev, note):
                weak_to_strong_pairs += 1
                if _is_chromatic_approach(int(prev["pitch"]), pitch, tones):
                    chromatic_approaches += 1
        if i >= 2 and _pc(pitch) in guide:
            enclosure_targets += 1
            if _is_enclosure(int(notes[i - 2]["pitch"]), int(notes[i - 1]["pitch"]), pitch, guide):
                enclosures += 1

    return {
        "solo_id": phrase.get("solo_id"),
        "performer": phrase.get("performer", ""),
        "title": phrase.get("title", ""),
        "phrase_number": phrase.get("phrase_number"),
        "notes": len(notes),
        "analyzable_notes": analyzable,
        "offbeat_notes": offbeat_notes,
        "offbeat_rate": offbeat_notes / analyzable if analyzable else 0.0,
        "chord_tone_landings": chord_tone_landings,
        "chord_tone_landing_rate": chord_tone_landings / analyzable if analyzable else 0.0,
        "guide_tone_landings": guide_tone_landings,
        "guide_tone_landing_rate": guide_tone_landings / analyzable if analyzable else 0.0,
        "dominant_notes": dominant_notes,
        "dominant_blues_color_notes": dominant_blues_color_notes,
        "dominant_blues_color_rate": dominant_blues_color_notes / dominant_notes if dominant_notes else 0.0,
        "weak_to_strong_pairs": weak_to_strong_pairs,
        "chromatic_approaches": chromatic_approaches,
        "chromatic_approach_rate": chromatic_approaches / weak_to_strong_pairs if weak_to_strong_pairs else 0.0,
        "enclosure_targets": enclosure_targets,
        "enclosures": enclosures,
        "enclosure_rate": enclosures / enclosure_targets if enclosure_targets else 0.0,
    }


def _aggregate(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    totals = defaultdict(float)
    for row in rows:
        for key in [
            "notes", "analyzable_notes", "offbeat_notes", "chord_tone_landings",
            "guide_tone_landings", "dominant_notes", "dominant_blues_color_notes",
            "weak_to_strong_pairs", "chromatic_approaches", "enclosure_targets", "enclosures",
        ]:
            totals[key] += float(row.get(key, 0))
    analyzable = totals["analyzable_notes"]
    weak_pairs = totals["weak_to_strong_pairs"]
    dom = totals["dominant_notes"]
    enclosure_targets = totals["enclosure_targets"]
    return {
        "phrases": len(rows),
        "notes": int(totals["notes"]),
        "analyzable_notes": int(analyzable),
        "offbeat_notes": int(totals["offbeat_notes"]),
        "offbeat_rate": totals["offbeat_notes"] / analyzable if analyzable else 0.0,
        "beat_boundary_candidate_pairs": int(weak_pairs),
        "beat_boundary_candidate_rate": weak_pairs / analyzable if analyzable else 0.0,
        "chord_tone_landings": int(totals["chord_tone_landings"]),
        "chord_tone_landing_rate": totals["chord_tone_landings"] / analyzable if analyzable else 0.0,
        "guide_tone_landings": int(totals["guide_tone_landings"]),
        "guide_tone_landing_rate": totals["guide_tone_landings"] / analyzable if analyzable else 0.0,
        "dominant_notes": int(dom),
        "dominant_blues_color_notes": int(totals["dominant_blues_color_notes"]),
        "dominant_blues_color_rate": totals["dominant_blues_color_notes"] / dom if dom else 0.0,
        "weak_to_strong_pairs": int(weak_pairs),
        "chromatic_approaches": int(totals["chromatic_approaches"]),
        "chromatic_approach_rate": totals["chromatic_approaches"] / weak_pairs if weak_pairs else 0.0,
        "enclosure_targets": int(enclosure_targets),
        "enclosures": int(totals["enclosures"]),
        "enclosure_rate": totals["enclosures"] / enclosure_targets if enclosure_targets else 0.0,
    }


def audit_phrases(phrases: list[dict], top_n: int = 20) -> dict:
    rows = [audit_phrase(p) for p in phrases]
    by_performer_map: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_performer_map[row.get("performer", "")].append(row)
    by_performer = []
    for performer, performer_rows in by_performer_map.items():
        agg = _aggregate(performer_rows)
        agg["performer"] = performer
        by_performer.append(agg)
    by_performer.sort(key=lambda r: (r["analyzable_notes"], r["chromatic_approach_rate"] + r["enclosure_rate"]), reverse=True)
    top_phrases = sorted(
        rows,
        key=lambda r: (r["chromatic_approaches"] + r["enclosures"], r["guide_tone_landings"], r["analyzable_notes"]),
        reverse=True,
    )[:top_n]
    return {
        "overall": _aggregate(rows),
        "by_performer": by_performer,
        "top_phrases": top_phrases,
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_markdown(report: dict, path: Path) -> None:
    overall = report["overall"]
    lines = [
        "# Jazz vocabulary corpus audit",
        "",
        "## Overall",
        "",
        f"- Phrases: {overall['phrases']:,}",
        f"- Analyzable notes: {overall['analyzable_notes']:,}",
        f"- Beat-boundary approach candidates: {overall['beat_boundary_candidate_pairs']:,} ({_pct(overall['beat_boundary_candidate_rate'])} of analyzable notes)",
        f"- Chord-tone landing rate: {_pct(overall['chord_tone_landing_rate'])}",
        f"- Guide-tone landing rate: {_pct(overall['guide_tone_landing_rate'])}",
        f"- Weak→strong chromatic approach rate: {_pct(overall['chromatic_approach_rate'])} ({overall['chromatic_approaches']:,}/{overall['weak_to_strong_pairs']:,})",
        f"- Guide-tone enclosure rate: {_pct(overall['enclosure_rate'])} ({overall['enclosures']:,}/{overall['enclosure_targets']:,})",
        f"- Dominant blues-color note rate: {_pct(overall['dominant_blues_color_rate'])} ({overall['dominant_blues_color_notes']:,}/{overall['dominant_notes']:,})",
        "",
        "## Top performers by analyzable note count",
        "",
        "| Performer | Notes | Approach rate | Enclosure rate | Guide-tone rate | Blues-color rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["by_performer"][:20]:
        lines.append(
            f"| {row['performer']} | {row['analyzable_notes']:,} | {_pct(row['chromatic_approach_rate'])} | "
            f"{_pct(row['enclosure_rate'])} | {_pct(row['guide_tone_landing_rate'])} | {_pct(row['dominant_blues_color_rate'])} |"
        )
    lines += [
        "",
        "## Phrases richest in detected vocabulary devices",
        "",
        "| Performer | Title | Solo | Phrase | Notes | Approaches | Enclosures | Guide tones |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["top_phrases"][:20]:
        lines.append(
            f"| {row['performer']} | {row['title']} | {row['solo_id']} | {row['phrase_number']} | "
            f"{row['analyzable_notes']} | {row['chromatic_approaches']} | {row['enclosures']} | {row['guide_tone_landings']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrases", type=Path, default=DEFAULT_PHRASES)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args(argv)

    phrases = json.loads(args.phrases.read_text())
    report = audit_phrases(phrases, top_n=args.top_n)
    report["source"] = str(args.phrases)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))
    write_markdown(report, args.md_out)
    overall = report["overall"]
    print(f"Wrote {args.json_out} and {args.md_out}")
    print(
        "overall: "
        f"phrases={overall['phrases']} notes={overall['analyzable_notes']} "
        f"approach={_pct(overall['chromatic_approach_rate'])} "
        f"enclosure={_pct(overall['enclosure_rate'])} "
        f"guide={_pct(overall['guide_tone_landing_rate'])} "
        f"blues_color={_pct(overall['dominant_blues_color_rate'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
