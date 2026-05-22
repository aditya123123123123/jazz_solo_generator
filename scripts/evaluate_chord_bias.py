#!/usr/bin/env python3
"""Generate and score before/after chord-tone-bias MIDI samples."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pretty_midi
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.generation.chord_utils import approach_tone_pitch_classes, chord_tone_pitch_classes
from src.generation.generate_solo import (
    BEAT_DURATION,
    CHECKPOINTS,
    PROGRESSIONS,
    ChordTokenizer,
    NoteTokenizer,
    PhraseTokenizer,
    ArtistTokenizer,
    export_json,
    export_midi,
    generate_solo,
    load_models,
)

OUT_DIR = REPO / "outputs" / "chord_bias_test"
CHECKPOINT = CHECKPOINTS / "v6.2.0_best.pt"
PROGRESSION_NAME = "autumn_leaves"
CASES = [
    ("no_bias", 0.9, 42, False),
    ("bias", 0.9, 42, True),
    ("no_bias", 1.1, 1337, False),
    ("bias", 1.1, 1337, True),
]


def _segment_bounds(progression):
    bounds = []
    beat_start = 0.0
    time_start = 0.0
    for chord, beats in progression:
        dur = beats * BEAT_DURATION
        bounds.append((chord, beat_start, time_start, time_start + dur))
        beat_start += beats
        time_start += dur
    return bounds


def _chord_at_time(bounds, start):
    for chord, beat_start, t0, t1 in bounds:
        if t0 <= start < t1:
            return chord, beat_start, t0
    return bounds[-1][0], bounds[-1][1], bounds[-1][2]


def score_events(note_events, progression):
    bounds = _segment_bounds(progression)
    sounding = []
    t = 0.0
    chord_hits = 0
    chord_or_approach_hits = 0
    strong_total = 0
    strong_hits = 0

    for pitch, dur, is_rest in note_events:
        if not is_rest and 0 <= pitch <= 127:
            chord, _beat_start, t0 = _chord_at_time(bounds, t)
            tones = chord_tone_pitch_classes(chord)
            approaches = approach_tone_pitch_classes(chord)
            pc = int(pitch) % 12
            is_chord_tone = pc in tones
            chord_hits += int(is_chord_tone)
            chord_or_approach_hits += int(is_chord_tone or pc in approaches)

            local_beat = (t - t0) / BEAT_DURATION
            nearest = round(local_beat)
            if nearest % 4 == 0 and abs(local_beat - nearest) <= 0.25:
                strong_total += 1
                strong_hits += int(is_chord_tone)
            sounding.append((int(pitch), float(t), float(dur)))
        t += float(dur)

    if not sounding:
        raise RuntimeError("generation produced no sounding notes")

    pitches = [p for p, _t, _d in sounding]
    return {
        "note_count": len(sounding),
        "chord_tone_hit_rate": chord_hits / len(sounding),
        "chord_or_approach_hit_rate": chord_or_approach_hits / len(sounding),
        "strong_beat_chord_tone_hit_rate": (strong_hits / strong_total) if strong_total else None,
        "first_24_pitches": pitches[:24],
    }


def generate_case(chord_tok, note_tok, phrase_tok, artist_tok, planner, executor, decode_dur, label, temp, seed, bias):
    random.seed(seed)
    torch.manual_seed(seed)
    progression = PROGRESSIONS[PROGRESSION_NAME]
    note_events, summaries, unknowns = generate_solo(
        progression,
        chord_tok=chord_tok,
        note_tok=note_tok,
        phrase_tok=phrase_tok,
        artist_tok=artist_tok,
        planner=planner,
        executor=executor,
        window=8,
        temperature=temp,
        decode_dur=decode_dur,
        tempo_bpm=60.0 / BEAT_DURATION,
        final_cadence_boost=3.0,
        chord_tone_bias=bias,
        chord_tone_bias_strength=3.0,
        non_chord_penalty=2.0,
        strong_beat_only=True,
    )
    if unknowns:
        raise RuntimeError(f"unknown chords in progression: {unknowns}")

    stem = f"{label}_temp_{temp:.1f}_seed_{seed}"
    midi_path = OUT_DIR / f"{stem}.mid"
    json_path = OUT_DIR / f"{stem}.json"
    export_midi(note_events, midi_path, tempo=60.0 / BEAT_DURATION)
    export_json(note_events, summaries, json_path, name=stem, tempo_bpm=60.0 / BEAT_DURATION)
    return midi_path, json_path, score_events(note_events, progression)


def write_report(results):
    lines = [
        "# Chord-Tone Bias Evaluation",
        "",
        f"Checkpoint: `{CHECKPOINT.relative_to(REPO)}`",
        f"Progression: `{PROGRESSION_NAME}`",
        "Chords: `Cm7 | F7 | Bbj7 | Ebj7 | Am7b5 | D7 | Gm7 | Gm7`",
        "",
        "Pitch token mapping: mapped vocabulary, not direct MIDI. `NoteTokenizer.decode_pitch(token)` maps `token - 4`.",
        "Bias mode: strong-beat-only, using 16-note segment anchors 0, 4, 8, and 12. No fallback was used.",
        "",
        "| case | MIDI | notes | chord-tone hit | chord+approach hit | strong-beat chord-tone hit | first 24 pitches |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        strong = result["metrics"]["strong_beat_chord_tone_hit_rate"]
        strong_text = "n/a" if strong is None else f"{strong:.3f}"
        lines.append(
            f"| {result['case']} | `{result['midi'].relative_to(REPO)}` | "
            f"{result['metrics']['note_count']} | "
            f"{result['metrics']['chord_tone_hit_rate']:.3f} | "
            f"{result['metrics']['chord_or_approach_hit_rate']:.3f} | "
            f"{strong_text} | "
            f"`{result['metrics']['first_24_pitches']}` |"
        )
    lines.extend(["", "## Improvement Check", ""])
    for temp, seed in [(0.9, 42), (1.1, 1337)]:
        no_bias = next(r for r in results if r["label"] == "no_bias" and r["temp"] == temp and r["seed"] == seed)
        bias = next(r for r in results if r["label"] == "bias" and r["temp"] == temp and r["seed"] == seed)
        delta_chord = bias["metrics"]["chord_tone_hit_rate"] - no_bias["metrics"]["chord_tone_hit_rate"]
        delta_approach = bias["metrics"]["chord_or_approach_hit_rate"] - no_bias["metrics"]["chord_or_approach_hit_rate"]
        lines.append(
            f"- temp={temp:.1f}, seed={seed}: chord-tone Δ={delta_chord:+.3f}; "
            f"chord+approach Δ={delta_approach:+.3f}"
        )
    report_path = OUT_DIR / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chord_tok = ChordTokenizer.from_json()
    note_tok = NoteTokenizer.from_json()
    phrase_tok = PhraseTokenizer()
    artist_tok = ArtistTokenizer()
    planner, executor, decode_dur = load_models(
        chord_tok,
        artist_tok,
        note_tok,
        note_executor_checkpoint=CHECKPOINT,
    )

    results = []
    for label, temp, seed, bias in CASES:
        midi, json_path, metrics = generate_case(
            chord_tok, note_tok, phrase_tok, artist_tok, planner, executor, decode_dur,
            label, temp, seed, bias,
        )
        results.append({
            "case": f"{label} temp={temp:.1f} seed={seed}",
            "label": label,
            "temp": temp,
            "seed": seed,
            "midi": midi,
            "json": json_path,
            "metrics": metrics,
        })

    for temp, seed in [(0.9, 42), (1.1, 1337)]:
        no_bias = next(r for r in results if r["label"] == "no_bias" and r["temp"] == temp and r["seed"] == seed)
        bias = next(r for r in results if r["label"] == "bias" and r["temp"] == temp and r["seed"] == seed)
        if (
            bias["metrics"]["chord_tone_hit_rate"] < no_bias["metrics"]["chord_tone_hit_rate"]
            and bias["metrics"]["chord_or_approach_hit_rate"] < no_bias["metrics"]["chord_or_approach_hit_rate"]
        ):
            raise RuntimeError(f"bias did not improve either harmony metric for temp={temp}, seed={seed}")

    report = write_report(results)
    for result in results:
        pm = pretty_midi.PrettyMIDI(str(result["midi"]))
        note_count = sum(len(inst.notes) for inst in pm.instruments)
        if note_count == 0:
            raise RuntimeError(f"empty MIDI written: {result['midi']}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
