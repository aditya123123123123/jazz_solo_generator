from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pretty_midi
from scipy.io import wavfile

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from src.generation.chord_utils import chord_tones, parse_chord

BASE = REPO / 'outputs' / 'auditions_v68_expanded_songs'
AUDIO = BASE / 'audio_wav'

def tones(sym):
    p = parse_chord(sym)
    return set(chord_tones(*p)) if p else set()

def approaches(sym):
    t = tones(sym)
    return {(x - 1) % 12 for x in t} | {(x + 1) % 12 for x in t}

def tune_name(mid: Path) -> str:
    return mid.name.split('_rhythm_')[0]

def synth(mid: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(str(mid))
    audio = pm.synthesize(fs=44100)
    if audio.size == 0:
        audio = np.zeros(44100, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    wavfile.write(str(out), 44100, np.int16(np.clip(audio / peak * .95, -1, 1) * 32767))

def metrics(js: Path, mid: Path):
    d = json.loads(js.read_text())
    pm = pretty_midi.PrettyMIDI(str(mid))
    solo = pm.instruments[0] if pm.instruments else None
    notes = sorted((solo.notes if solo else []), key=lambda n: (n.start, n.pitch))
    pitches = [n.pitch for n in notes]
    intervals = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    total = hit = either = 0
    reg_adj = repeat_adj = density_adj = 0
    reg_resampled = 0
    for sec in d.get('sections', []):
        ct = tones(sec.get('chord', ''))
        ap = approaches(sec.get('chord', ''))
        for pitch in [int(x) for x in sec.get('pitches', []) if 0 <= int(x) <= 127]:
            total += 1
            pc = pitch % 12
            hit += pc in ct
            either += (pc in ct) or (pc in ap)
        reg_adj += int(sec.get('register_continuity_adjusted', 0) or 0)
        reg_resampled += int(bool(sec.get('register_continuity_resampled', False)))
        repeat_adj += int(sec.get('repeat_guard_adjusted', 0) or 0)
        density_adj += int(sec.get('rhythm_density_calibration_adjusted', 0) or 0)
    dur = max((n.end for n in notes), default=0) or 1
    return dict(
        tune=tune_name(mid), notes=len(notes), duration=dur,
        chord_tone=hit / total if total else 0,
        chord_or_approach=either / total if total else 0,
        density=len(notes) / dur,
        avg_interval=sum(intervals) / len(intervals) if intervals else 0,
        big_leap=sum(i >= 10 for i in intervals) / len(intervals) if intervals else 0,
        repeat=sum(i == 0 for i in intervals) / len(intervals) if intervals else 0,
        register_adjustments=reg_adj,
        register_resampled_sections=reg_resampled,
        repeat_guard_adjusted=repeat_adj,
        density_calibration_adjusted=density_adj,
    )

rows = []
for mid in sorted(BASE.glob('*.mid')):
    wav = AUDIO / (tune_name(mid) + '.wav')
    synth(mid, wav)
    r = metrics(mid.with_suffix('.json'), mid)
    r['audio'] = str(wav.relative_to(REPO))
    rows.append(r)

notes = sum(r['notes'] for r in rows)
def w(k):
    return sum(r[k] * r['notes'] for r in rows) / notes if notes else 0

lines = [
    '# v6.8 Expanded Song Audition Comparison',
    '',
    'Same v6.8 checkpoint; expanded default generation set from 3 probes to 8 probes.',
    '',
    '| total tunes | notes | chord-tone | chord+approach | density | avg interval | big leap | repeat | reg edits | reg resampled | density edits | repeat edits |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    f"| {len(rows)} | {notes} | {w('chord_tone'):.3f} | {w('chord_or_approach'):.3f} | {w('density'):.2f} | {w('avg_interval'):.2f} | {w('big_leap'):.3f} | {w('repeat'):.3f} | {sum(r['register_adjustments'] for r in rows)} | {sum(r['register_resampled_sections'] for r in rows)} | {sum(r['density_calibration_adjusted'] for r in rows)} | {sum(r['repeat_guard_adjusted'] for r in rows)} |",
    '',
    '## Per tune',
    '',
    '| tune | notes | chord-tone | chord+approach | repeat | reg edits | reg resampled | density edits | audio |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---|',
]
for r in rows:
    lines.append(f"| {r['tune']} | {r['notes']} | {r['chord_tone']:.3f} | {r['chord_or_approach']:.3f} | {r['repeat']:.3f} | {r['register_adjustments']} | {r['register_resampled_sections']} | {r['density_calibration_adjusted']} | `{r['audio']}` |")

out = BASE / 'V68_EXPANDED_SONG_COMPARISON.md'
out.write_text('\n'.join(lines) + '\n')
print(out)
for line in lines[:16]:
    print(line)
print('rows', len(rows), 'wav', len(list(AUDIO.glob('*.wav'))))
