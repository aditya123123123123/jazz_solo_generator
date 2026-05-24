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

CHECKPOINT = REPO / "checkpoints" / "v6.3.1_best.pt"
SEEDS = [1, 2, 3, 4, 5, 42, 1337]
CONFIGS = {
    "none": {},
    "strong_s2": {"chord_tone_bias": True, "chord_tone_bias_strength": 2.0},
    "strong_s3": {"chord_tone_bias": True, "chord_tone_bias_strength": 3.0},
    "strong_s3_p1": {"chord_tone_bias": True, "chord_tone_bias_strength": 3.0, "non_chord_penalty": 1.0},
    "all_s1": {"chord_tone_bias": True, "chord_tone_bias_strength": 1.0, "strong_beat_only": False},
    "all_s2": {"chord_tone_bias": True, "chord_tone_bias_strength": 2.0, "strong_beat_only": False},
}


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
    planner, executor, decode_dur = load_models(chord_tok, artist_tok, note_tok, note_executor_checkpoint=CHECKPOINT)
    tempo_bpm = 60.0 / BEAT_DURATION

    rows = []
    for config_name, config in CONFIGS.items():
        for seed in SEEDS:
            random.seed(seed)
            torch.manual_seed(seed)
            for tune, progression in PROGRESSIONS.items():
                _events, summaries, unknowns = generate_solo(
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
                    **config,
                )
                if unknowns:
                    raise RuntimeError(f"unknowns for {tune}: {unknowns}")
                total, chord_hits, either_hits = score_summaries(summaries)
                rows.append({
                    "config": config_name,
                    "seed": seed,
                    "tune": tune,
                    "notes": total,
                    "chord_hits": chord_hits,
                    "either_hits": either_hits,
                })

    lines = [
        "# v6.3.1 Decoder Bias Grid",
        "",
        f"Checkpoint: `{CHECKPOINT.relative_to(REPO)}`",
        f"Seeds: `{SEEDS}`",
        "Scoring: section-based generated pitch classes against each source chord.",
        "",
        "| config | pitches scored | chord-tone hit | chord+approach hit | chord-tone delta vs none | chord+approach delta vs none |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    def agg(config):
        sub = [r for r in rows if r["config"] == config]
        total = sum(r["notes"] for r in sub)
        chord = sum(r["chord_hits"] for r in sub) / total
        either = sum(r["either_hits"] for r in sub) / total
        return total, chord, either
    _, base_chord, base_either = agg("none")
    for config in CONFIGS:
        total, chord, either = agg(config)
        lines.append(f"| {config} | {total} | {chord:.3f} | {either:.3f} | {chord - base_chord:+.3f} | {either - base_either:+.3f} |")
    out = REPO / "outputs" / "v631_decoder_bias_grid.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
