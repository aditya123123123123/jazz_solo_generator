#!/usr/bin/env python3
"""Add rule-based rhythm-section tracks to existing generated solo MIDIs.

This does not regenerate model notes. It reads each existing .json/.mid pair,
reconstructs the intended chord progression from JSON sections, and writes a new
multitrack .mid containing:
  - original solo track(s)
  - Piano comping
  - Bass
  - Drums

The rhythm section is generated from the actual chord symbols/beats in the JSON,
so its bass roots and piano voicings follow the intended harmony.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pretty_midi

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.generation.chord_utils import chord_tones, parse_chord
from src.generation.rhythm_section import generate_rhythm_section


def _load_progression(json_path: Path) -> list[tuple[str, int]]:
    data = json.loads(json_path.read_text())
    progression = []
    for section in data["sections"]:
        chord = section["chord"]
        beats = int(section["beats"])
        progression.append((chord, beats))
    return progression


def _clone_solo_midi(midi_path: Path, tempo_bpm: float) -> pretty_midi.PrettyMIDI:
    src = pretty_midi.PrettyMIDI(str(midi_path))
    out = pretty_midi.PrettyMIDI(initial_tempo=float(tempo_bpm))
    for idx, inst in enumerate(src.instruments):
        # Generated melody MIDIs historically used the name "Piano" even though
        # the track is the solo. Rename it so it cannot collide with the new
        # rhythm-section piano track.
        if idx == 0:
            name = "Solo"
        else:
            name = inst.name or f"Solo {idx + 1}"
        cloned = pretty_midi.Instrument(
            program=inst.program,
            is_drum=inst.is_drum,
            name=name,
        )
        cloned.notes = [
            pretty_midi.Note(
                velocity=n.velocity,
                pitch=n.pitch,
                start=n.start,
                end=n.end,
            )
            for n in inst.notes
        ]
        cloned.pitch_bends = list(inst.pitch_bends)
        cloned.control_changes = list(inst.control_changes)
        out.instruments.append(cloned)
    return out


def _active_chord_at_beat(progression: list[tuple[str, int]], beat: float) -> tuple[str, int] | None:
    cursor = 0.0
    for chord, beats in progression:
        if cursor <= beat < cursor + beats:
            parsed = parse_chord(chord)
            return (chord, parsed[0]) if parsed else None
        cursor += beats
    return None


def _progression_duration_sec(progression: list[tuple[str, int]], tempo_bpm: float) -> float:
    return sum(beats for _chord, beats in progression) * 60.0 / float(tempo_bpm)


def _repeat_progression_to_cover(
    progression: list[tuple[str, int]],
    target_end_sec: float,
    tempo_bpm: float,
) -> tuple[list[tuple[str, int]], int]:
    """Loop the intended form until the rhythm section covers the solo.

    The source JSON contains one chorus/form. Some generated melody MIDIs run
    longer than that form because the model durations are free-running. Repeating
    the form is more musically honest than letting the rhythm section stop early:
    the backing harmony remains exactly the intended changes, just over multiple
    choruses.
    """
    if not progression:
        return [], 0
    one_form_sec = _progression_duration_sec(progression, tempo_bpm)
    if one_form_sec <= 0:
        return list(progression), 1
    repeats = max(1, int(-(-target_end_sec // one_form_sec)))  # ceil for floats
    return list(progression) * repeats, repeats


def _verify_rhythm(pm: pretty_midi.PrettyMIDI, progression: list[tuple[str, int]], tempo_bpm: float) -> dict:
    beat_dur = 60.0 / float(tempo_bpm)
    names = {inst.name: inst for inst in pm.instruments}
    bass = names.get("Bass")
    piano = names.get("Piano")
    drums = names.get("Drums")
    if bass is None or piano is None or drums is None:
        raise AssertionError(f"missing rhythm instruments: present={sorted(names)}")

    # Bass must state the active chord root at every chord boundary.
    cursor = 0
    bass_root_checks = 0
    for chord, beats in progression:
        parsed = parse_chord(chord)
        if parsed is not None:
            root_pc, _quality = parsed
            t = cursor * beat_dur
            notes_here = [n for n in bass.notes if abs(n.start - t) < 1e-6]
            if not notes_here:
                raise AssertionError(f"no bass note at {chord} boundary beat {cursor}")
            if notes_here[0].pitch % 12 != root_pc:
                raise AssertionError(
                    f"bass root mismatch at {chord}: got pc {notes_here[0].pitch % 12}, expected {root_pc}"
                )
            bass_root_checks += 1
        cursor += beats

    # Piano comp notes should be chord-tone/shell-tone based for the active chord.
    piano_checks = 0
    for n in piano.notes:
        active = _active_chord_at_beat(progression, n.start / beat_dur)
        if active is None:
            continue
        chord, _root = active
        parsed = parse_chord(chord)
        if parsed is None:
            continue
        root_pc, quality = parsed
        allowed = set(chord_tones(root_pc, quality)) | {(root_pc + 14) % 12}
        if n.pitch % 12 not in allowed:
            raise AssertionError(
                f"piano note {n.pitch} at {n.start:.3f}s is not in allowed tones for {chord}: {sorted(allowed)}"
            )
        piano_checks += 1

    return {
        "tracks": len(pm.instruments),
        "solo_notes": sum(len(i.notes) for i in pm.instruments if i.name not in {"Piano", "Bass", "Drums"}),
        "piano_notes": len(piano.notes),
        "bass_notes": len(bass.notes),
        "drum_notes": len(drums.notes),
        "bass_root_checks": bass_root_checks,
        "piano_tone_checks": piano_checks,
    }


def add_rhythm(input_dir: Path, output_dir: Path, style: str, seed: int, tempo_bpm: float) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for json_path in sorted(input_dir.glob("*.json")):
        midi_path = input_dir / f"{json_path.stem}.mid"
        if not midi_path.exists():
            print(f"SKIP missing MIDI for {json_path.name}", file=sys.stderr)
            continue
        progression = _load_progression(json_path)
        pm = _clone_solo_midi(midi_path, tempo_bpm=tempo_bpm)
        solo_end_sec = pm.get_end_time()
        backing_progression, form_repeats = _repeat_progression_to_cover(
            progression, solo_end_sec, tempo_bpm=tempo_bpm,
        )
        rhythm = generate_rhythm_section(backing_progression, tempo_bpm=tempo_bpm, style=style, seed=seed)
        pm.instruments.extend(rhythm)
        out_path = output_dir / f"{json_path.stem}_with_rhythm_{style}.mid"
        pm.write(str(out_path))
        loaded = pretty_midi.PrettyMIDI(str(out_path))
        stats = _verify_rhythm(loaded, backing_progression, tempo_bpm=tempo_bpm)
        rows.append({
            "name": json_path.stem,
            "source_midi": str(midi_path),
            "source_json": str(json_path),
            "output_midi": str(out_path),
            "chords": len(progression),
            "form_repeats": form_repeats,
            "backing_chords": len(backing_progression),
            "beats": sum(b for _c, b in backing_progression),
            "solo_end_sec": round(solo_end_sec, 6),
            "backing_end_sec": round(_progression_duration_sec(backing_progression, tempo_bpm), 6),
            **stats,
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--style", choices=["swing", "bossa", "ballad", "latin", "funk"], default="swing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tempo-bpm", type=float, default=120.0)
    args = ap.parse_args(argv)

    rows = add_rhythm(args.input_dir, args.output_dir, args.style, args.seed, args.tempo_bpm)
    if not rows:
        raise SystemExit("No .json/.mid pairs found")

    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2))

    report = args.output_dir / "RHYTHM_SECTION_REPORT.md"
    report.write_text(
        "# Rhythm section MIDI export\n\n"
        f"Source: `{args.input_dir}`\n\n"
        f"Style: `{args.style}`  Tempo: `{args.tempo_bpm}` BPM  Seed: `{args.seed}`\n\n"
        "| name | chords | repeats | backing chords | solo end | backing end | solo | piano | bass | drums | bass root checks | piano tone checks | file |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
        + "".join(
            f"| {r['name']} | {r['chords']} | {r['form_repeats']} | {r['backing_chords']} | {r['solo_end_sec']} | {r['backing_end_sec']} | {r['solo_notes']} | {r['piano_notes']} | {r['bass_notes']} | {r['drum_notes']} | {r['bass_root_checks']} | {r['piano_tone_checks']} | `{Path(r['output_midi']).name}` |\n"
            for r in rows
        )
    )

    for r in rows:
        print(
            f"OK {r['name']}: wrote {r['output_midi']} "
            f"(solo={r['solo_notes']}, piano={r['piano_notes']}, bass={r['bass_notes']}, drums={r['drum_notes']}, "
            f"bass_root_checks={r['bass_root_checks']}, piano_tone_checks={r['piano_tone_checks']})"
        )
    print(f"Manifest: {manifest}")
    print(f"Report:   {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
