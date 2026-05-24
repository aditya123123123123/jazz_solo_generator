#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics as stats
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import pretty_midi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.generation.chord_utils import chord_tones, parse_chord

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def pc_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


def note_name(p: int) -> str:
    return pretty_midi.note_number_to_name(int(p))


def tones(symbol: str) -> set[int]:
    parsed = parse_chord(symbol)
    return set(chord_tones(*parsed)) if parsed else set()


def approaches(symbol: str) -> set[int]:
    t = tones(symbol)
    return {(x - 1) % 12 for x in t} | {(x + 1) % 12 for x in t}


def render_sine(pm: pretty_midi.PrettyMIDI, out_wav: Path, sr: int = 22050) -> None:
    # pretty_midi's sine synth is dependency-light; enough for audio feature inspection.
    audio = pm.synthesize(fs=sr)
    if not len(audio):
        audio = np.zeros(sr, dtype=np.float32)
    audio = audio / max(1.0, float(np.max(np.abs(audio)))) * 0.85
    pcm = (audio * 32767).astype(np.int16)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def phrase_density(notes):
    if not notes:
        return 0.0
    span = max(n.end for n in notes) - min(n.start for n in notes)
    return len(notes) / span if span else 0.0


def audit_file(json_path: Path, midi_path: Path, wav_dir: Path) -> dict:
    data = json.loads(json_path.read_text())
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted([n for inst in pm.instruments for n in inst.notes], key=lambda n: (n.start, n.pitch))
    render_sine(pm, wav_dir / (midi_path.stem + '.wav'))

    pitches = [n.pitch for n in notes]
    intervals = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    durations = [n.end - n.start for n in notes]
    pcs = Counter(p % 12 for p in pitches)
    repeats = sum(1 for a, b in zip(pitches, pitches[1:]) if a == b)
    big_leaps = sum(1 for x in intervals if x >= 10)
    stepwise = sum(1 for x in intervals if x <= 2)
    rests = [n for n in data.get('notes', []) if n.get('is_rest')]
    note_events = [n for n in data.get('notes', []) if not n.get('is_rest')]

    section_rows = []
    total = hit = either = 0
    non_tone_examples = []
    for sec in data.get('sections', []):
        chord = sec.get('chord', '')
        ct = tones(chord)
        ap = approaches(chord)
        sec_pitches = [int(p) for p in sec.get('pitches', [])]
        sec_hit = sum((p % 12) in ct for p in sec_pitches)
        sec_either = sum(((p % 12) in ct) or ((p % 12) in ap) for p in sec_pitches)
        total += len(sec_pitches)
        hit += sec_hit
        either += sec_either
        off = [p for p in sec_pitches if (p % 12) not in ct and (p % 12) not in ap]
        if off:
            non_tone_examples.append(f"{chord}: " + ', '.join(note_name(p) for p in off[:5]))
        section_rows.append({
            'chord': chord,
            'tones': ' '.join(pc_name(x) for x in sorted(ct)),
            'notes': len(sec_pitches),
            'chord_tone': sec_hit / len(sec_pitches) if sec_pitches else 0,
            'chord_or_approach': sec_either / len(sec_pitches) if sec_pitches else 0,
            'sample': ' '.join(note_name(p) for p in sec_pitches[:10]),
        })

    pitch_entropy = 0.0
    if pitches:
        counts = Counter(pitches)
        n = len(pitches)
        pitch_entropy = -sum((c/n) * math.log2(c/n) for c in counts.values())

    return {
        'name': json_path.stem,
        'midi': str(midi_path),
        'wav': str(wav_dir / (midi_path.stem + '.wav')),
        'notes': len(notes),
        'duration_sec': max((n.end for n in notes), default=0),
        'range': f"{note_name(min(pitches))}-{note_name(max(pitches))}" if pitches else 'n/a',
        'density_notes_per_sec': phrase_density(notes),
        'median_dur_sec': stats.median(durations) if durations else 0,
        'rest_events': len(rests),
        'rest_ratio_events': len(rests) / max(1, len(rests) + len(note_events)),
        'avg_abs_interval': sum(intervals) / len(intervals) if intervals else 0,
        'stepwise_ratio': stepwise / len(intervals) if intervals else 0,
        'big_leap_ratio': big_leaps / len(intervals) if intervals else 0,
        'repeat_ratio': repeats / len(intervals) if intervals else 0,
        'pitch_entropy': pitch_entropy,
        'top_pitch_classes': ', '.join(f"{pc_name(pc)}:{c}" for pc, c in pcs.most_common(6)),
        'chord_tone': hit / total if total else 0,
        'chord_or_approach': either / total if total else 0,
        'sections': section_rows,
        'non_tone_examples': non_tone_examples[:8],
    }


def main() -> None:
    base = REPO / 'outputs' / 'solos_v6.3.2_allbeat_eval'
    wav_dir = REPO / 'outputs' / 'rendered_audio_v632_allbeat'
    results = []
    for json_path in sorted(base.glob('*.json')):
        midi_path = json_path.with_suffix('.mid')
        if midi_path.exists():
            results.append(audit_file(json_path, midi_path, wav_dir))

    out = REPO / 'outputs' / 'v632_allbeat_musical_audit.md'
    lines = ['# v6.3.2 allbeat generated MIDI musical audit', '']
    lines.append('Rendered WAVs are sine-synth previews for inspection, not performance-quality audio.')
    lines.append('')
    for r in results:
        lines += [f"## {r['name']}", '']
        lines.append(f"- MIDI: `{r['midi']}`")
        lines.append(f"- WAV preview: `{r['wav']}`")
        lines.append(f"- Notes/duration/density: {r['notes']} notes, {r['duration_sec']:.2f}s, {r['density_notes_per_sec']:.2f} notes/sec")
        lines.append(f"- Range: {r['range']}; median duration {r['median_dur_sec']:.3f}s; rest-event ratio {r['rest_ratio_events']:.2f}")
        lines.append(f"- Motion: avg abs interval {r['avg_abs_interval']:.2f} st; stepwise {r['stepwise_ratio']:.2f}; big leaps {r['big_leap_ratio']:.2f}; repeats {r['repeat_ratio']:.2f}")
        lines.append(f"- Pitch entropy: {r['pitch_entropy']:.2f}; top pitch classes: {r['top_pitch_classes']}")
        lines.append(f"- Harmonic: chord-tone {r['chord_tone']:.3f}; chord-or-approach {r['chord_or_approach']:.3f}")
        if r['non_tone_examples']:
            lines.append(f"- Off-grid examples: {'; '.join(r['non_tone_examples'])}")
        lines.append('')
        lines.append('| chord | tones | n | chord-tone | chord/approach | sample |')
        lines.append('|---|---|---:|---:|---:|---|')
        for s in r['sections']:
            lines.append(f"| {s['chord']} | {s['tones']} | {s['notes']} | {s['chord_tone']:.3f} | {s['chord_or_approach']:.3f} | {s['sample']} |")
        lines.append('')
    out.write_text('\n'.join(lines))
    print(out)
    for r in results:
        print(f"{r['name']}: chord-tone={r['chord_tone']:.3f} chord/approach={r['chord_or_approach']:.3f} range={r['range']} density={r['density_notes_per_sec']:.2f}/s stepwise={r['stepwise_ratio']:.2f} big_leap={r['big_leap_ratio']:.2f} wav={r['wav']}")


if __name__ == '__main__':
    main()
