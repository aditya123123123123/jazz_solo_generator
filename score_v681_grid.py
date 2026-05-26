from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import pretty_midi
from scipy.io import wavfile

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from src.generation.chord_utils import chord_tones, parse_chord

BASE = REPO / 'outputs' / 'auditions_v681_grid'
VARIANTS = sorted(p.name for p in BASE.iterdir() if p.is_dir() and p.name.startswith('iivi_'))

def tones(sym):
    p = parse_chord(sym)
    return set(chord_tones(*p)) if p else set()

def approaches(sym):
    t = tones(sym)
    return {(x - 1) % 12 for x in t} | {(x + 1) % 12 for x in t}

def synth(mid, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(str(mid))
    audio = pm.synthesize(fs=44100)
    if audio.size == 0:
        audio = np.zeros(44100, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    wavfile.write(str(out), 44100, np.int16(np.clip(audio / peak * .95, -1, 1) * 32767))

def metrics(js, mid):
    d = json.loads(js.read_text())
    pm = pretty_midi.PrettyMIDI(str(mid))
    solo = pm.instruments[0] if pm.instruments else None
    notes = sorted((solo.notes if solo else []), key=lambda n: (n.start, n.pitch))
    pitches = [n.pitch for n in notes]
    intervals = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    total = hit = either = 0
    reg_adj = reg_resampled = repeat_adj = density_adj = 0
    low_density_sections = []
    for idx, sec in enumerate(d.get('sections', [])):
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
        gm = sec.get('generated_phrase_metrics') or {}
        n_notes = sec.get('n_notes') or 0
        actual = gm.get('num_notes') or len(sec.get('pitches') or [])
        if n_notes and actual / n_notes < 0.6:
            low_density_sections.append((idx, sec.get('chord'), actual, n_notes))
    dur = max((n.end for n in notes), default=0) or 1
    return dict(
        notes=len(notes), duration=dur,
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
        low_density_sections=low_density_sections,
    )

rows=[]
for v in VARIANTS:
    for mid in sorted((BASE / v).glob('*.mid')):
        tune = mid.name.split('_rhythm_')[0]
        wav = BASE / 'audio_wav' / v / (tune + '.wav')
        synth(mid, wav)
        r = metrics(mid.with_suffix('.json'), mid)
        r.update(variant=v, tune=tune, audio=str(wav.relative_to(REPO)))
        rows.append(r)

by=defaultdict(list)
for r in rows:
    by[r['variant']].append(r)
lines=['# v6.8.1 Inference Grid Comparison','', 'Settings: density calibration threshold 0.6, register resample cap >4 edits or >30%, one resample attempt, repeat guard max 2.','','| variant | notes | chord-tone | chord+approach | density | avg interval | big leap | repeat | reg edits | reg resampled | density edits | repeat edits |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
summary=[]
for v in VARIANTS:
    ss=by[v]
    notes=sum(x['notes'] for x in ss)
    def w(k): return sum(x[k]*x['notes'] for x in ss)/notes if notes else 0
    reg=sum(x['register_adjustments'] for x in ss)
    res=sum(x['register_resampled_sections'] for x in ss)
    den=sum(x['density_calibration_adjusted'] for x in ss)
    rep=sum(x['repeat_guard_adjusted'] for x in ss)
    lines.append(f"| {v} | {notes} | {w('chord_tone'):.3f} | {w('chord_or_approach'):.3f} | {w('density'):.2f} | {w('avg_interval'):.2f} | {w('big_leap'):.3f} | {w('repeat'):.3f} | {reg} | {res} | {den} | {rep} |")
    summary.append((v, notes, w('chord_tone'), w('chord_or_approach'), w('repeat'), reg, res, den, rep))
lines += ['','## Per tune','','| variant | tune | notes | chord-tone | chord+approach | repeat | reg edits | reg resampled | density edits | repeat edits | audio |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['variant']} | {r['tune']} | {r['notes']} | {r['chord_tone']:.3f} | {r['chord_or_approach']:.3f} | {r['repeat']:.3f} | {r['register_adjustments']} | {r['register_resampled_sections']} | {r['density_calibration_adjusted']} | {r['repeat_guard_adjusted']} | `{r['audio']}` |")

out=BASE/'V681_GRID_COMPARISON.md'
out.write_text('\n'.join(lines)+'\n')
print(out)
for l in lines[:14]: print(l)
print('rows', len(rows), 'wav', len(list((BASE/'audio_wav').rglob('*.wav'))))
