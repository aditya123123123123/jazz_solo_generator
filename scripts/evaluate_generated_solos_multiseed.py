#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.generation.chord_utils import chord_tones, parse_chord
from src.generation.generate_solo import (
    BEAT_DURATION,
    PROGRESSIONS,
    ArtistTokenizer,
    ChordTokenizer,
    NoteTokenizer,
    PhraseTokenizer,
    generate_solo,
    load_models,
)

CHECKPOINTS = {
    "v6.2.0": REPO / "checkpoints" / "v6.2.0_best.pt",
    "v6.3.0": REPO / "checkpoints" / "v6.3.0_best.pt",
    "v6.3.1": REPO / "checkpoints" / "v6.3.1_best.pt",
}
SEEDS = [1, 2, 3, 4, 5, 42, 1337]


def chord_tone_pitch_classes(symbol: str) -> set[int]:
    parsed = parse_chord(symbol)
    return set(chord_tones(*parsed)) if parsed else set()


def approach_tone_pitch_classes(symbol: str) -> set[int]:
    tones = chord_tone_pitch_classes(symbol)
    return {((pc - 1) % 12) for pc in tones} | {((pc + 1) % 12) for pc in tones}


def score_summaries(summaries):
    total = chord_hits = either_hits = 0
    for sec in summaries:
        tones = chord_tone_pitch_classes(sec["chord"])
        approaches = approach_tone_pitch_classes(sec["chord"])
        for p in sec.get("pitches", []):
            p = int(p)
            if not (0 <= p <= 127):
                continue
            pc = p % 12
            total += 1
            chord_hits += pc in tones
            either_hits += (pc in tones) or (pc in approaches)
    if total == 0:
        raise RuntimeError("no section pitches scored")
    return total, chord_hits, either_hits


def main():
    chord_tok = ChordTokenizer.from_json()
    note_tok = NoteTokenizer.from_json()
    phrase_tok = PhraseTokenizer()
    artist_tok = ArtistTokenizer()
    tempo_bpm = 60.0 / BEAT_DURATION

    rows = []
    for version, ckpt in CHECKPOINTS.items():
        planner, executor, decode_dur = load_models(chord_tok, artist_tok, note_tok, note_executor_checkpoint=ckpt)
        for seed in SEEDS:
            random.seed(seed)
            torch.manual_seed(seed)
            for tune, progression in PROGRESSIONS.items():
                note_events, summaries, unknowns = generate_solo(
                    progression,
                    chord_tok=chord_tok,
                    note_tok=note_tok,
                    phrase_tok=phrase_tok,
                    artist_tok=artist_tok,
                    planner=planner,
                    executor=executor,
                    window=8,
                    temperature=0.8,
                    decode_dur=decode_dur,
                    tempo_bpm=tempo_bpm,
                    duration_temperature=1.6,
                    rest_boost=1.8,
                    final_cadence_boost=3.0,
                )
                if unknowns:
                    raise RuntimeError(f"unknowns for {tune}: {unknowns}")
                total, chord_hits, either_hits = score_summaries(summaries)
                rows.append({
                    "version": version,
                    "seed": seed,
                    "tune": tune,
                    "notes": total,
                    "chord": chord_hits / total,
                    "either": either_hits / total,
                })

    def agg(version, tune=None):
        subset = [r for r in rows if r["version"] == version and (tune is None or r["tune"] == tune)]
        notes = sum(r["notes"] for r in subset)
        chord = sum(r["chord"] * r["notes"] for r in subset) / notes
        either = sum(r["either"] * r["notes"] for r in subset) / notes
        return notes, chord, either

    lines = []
    lines.append("# Multi-seed Harmonic Evaluation")
    lines.append("")
    lines.append(f"Seeds: `{SEEDS}`")
    lines.append("Scoring: section-based chord tone / chord-or-approach rate from generated section pitches.")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| version | pitches scored | chord-tone hit | chord+approach hit |")
    lines.append("|---|---:|---:|---:|")
    for version in CHECKPOINTS:
        notes, chord, either = agg(version)
        lines.append(f"| {version} | {notes} | {chord:.3f} | {either:.3f} |")
    lines.append("")
    lines.append("## Per tune")
    lines.append("")
    lines.append("| version | tune | pitches scored | chord-tone hit | chord+approach hit |")
    lines.append("|---|---|---:|---:|---:|")
    for version in CHECKPOINTS:
        for tune in PROGRESSIONS:
            notes, chord, either = agg(version, tune)
            lines.append(f"| {version} | {tune} | {notes} | {chord:.3f} | {either:.3f} |")
    lines.append("")
    lines.append("## Delta vs v6.3.0")
    lines.append("")
    base_notes, base_chord, base_either = agg("v6.3.0")
    for version in ["v6.2.0", "v6.3.1"]:
        _, chord, either = agg(version)
        lines.append(f"- {version}: chord-tone Δ={chord - base_chord:+.3f}; chord+approach Δ={either - base_either:+.3f}")
    out = REPO / "outputs" / "solo_eval_multiseed_report.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
