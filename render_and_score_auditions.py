from __future__ import annotations
import json, math, statistics as stats, subprocess
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pretty_midi
from scipy.io import wavfile
import sys
REPO=Path.cwd()
sys.path.insert(0,str(REPO))
from src.generation.chord_utils import chord_tones, parse_chord
BASE=REPO/'outputs'/'auditions_v68_compare'
AUDIO=BASE/'audio_m4a'
VERS=['v66','v67','v68']

def tones(sym):
    p=parse_chord(sym)
    return set(chord_tones(*p)) if p else set()

def approaches(sym):
    t=tones(sym); return {(x-1)%12 for x in t}|{(x+1)%12 for x in t}

def synth_to_m4a(mid: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    wav=out.with_suffix('.wav')
    pm=pretty_midi.PrettyMIDI(str(mid))
    audio=pm.synthesize(fs=22050)
    if audio.size==0: audio=np.zeros(22050, dtype=np.float32)
    peak=float(np.max(np.abs(audio))) or 1.0
    audio_i16=np.int16(np.clip(audio/peak*0.95, -1, 1)*32767)
    wavfile.write(str(wav), 22050, audio_i16)
    subprocess.run(['afconvert','-f','m4af','-d','aac','-b','192000',str(wav),str(out)], check=True)
    wav.unlink(missing_ok=True)

def metrics(json_path: Path, midi_path: Path):
    data=json.loads(json_path.read_text())
    pm=pretty_midi.PrettyMIDI(str(midi_path))
    solo_inst=pm.instruments[0] if pm.instruments else None
    solo_notes=sorted((solo_inst.notes if solo_inst else []), key=lambda n:(n.start,n.pitch))
    pitches=[n.pitch for n in solo_notes]
    intervals=[abs(b-a) for a,b in zip(pitches,pitches[1:])]
    durations=[n.end-n.start for n in solo_notes]
    sec_total=hit=either=0
    off=[]
    for sec in data.get('sections',[]):
        ct=tones(sec.get('chord','')); ap=approaches(sec.get('chord',''))
        for p in [int(x) for x in sec.get('pitches',[]) if 0<=int(x)<=127]:
            sec_total+=1
            pc=p%12
            hit += pc in ct
            either += (pc in ct) or (pc in ap)
            if pc not in ct and pc not in ap and len(off)<6:
                off.append(f"{sec.get('chord')}:{pretty_midi.note_number_to_name(p)}")
    pcs=Counter(p%12 for p in pitches)
    density=len(solo_notes)/(max((n.end for n in solo_notes), default=0) or 1)
    return {
      'notes': len(solo_notes),
      'duration': max((n.end for n in solo_notes), default=0),
      'range': f"{pretty_midi.note_number_to_name(min(pitches))}-{pretty_midi.note_number_to_name(max(pitches))}" if pitches else 'n/a',
      'density': density,
      'median_dur': stats.median(durations) if durations else 0,
      'avg_interval': sum(intervals)/len(intervals) if intervals else 0,
      'stepwise': sum(i<=2 for i in intervals)/len(intervals) if intervals else 0,
      'big_leap': sum(i>=10 for i in intervals)/len(intervals) if intervals else 0,
      'repeat': sum(i==0 for i in intervals)/len(intervals) if intervals else 0,
      'chord_tone': hit/sec_total if sec_total else 0,
      'chord_or_approach': either/sec_total if sec_total else 0,
      'off_examples': ', '.join(off),
    }

rows=[]
for v in VERS:
    for mid in sorted((BASE/v).glob('*.mid')):
        tune=mid.name.split('_rhythm_')[0]
        m4a=AUDIO/v/(tune+'.m4a')
        synth_to_m4a(mid,m4a)
        js=mid.with_suffix('.json')
        r=metrics(js,mid)
        r.update(version=v,tune=tune,midi=str(mid.relative_to(REPO)),json=str(js.relative_to(REPO)),audio=str(m4a.relative_to(REPO)))
        rows.append(r)

byv=defaultdict(list)
for r in rows: byv[r['version']].append(r)
lines=['# v6.8 Audition Comparison','','Matched generation settings: rhythm section=swing, phrase shaping, register continuity, phrase diversity max 3 with cadence preservation, bebop approach notes, target-contour section cadence.','','Audio previews are AAC `.m4a` synthesized from the generated MIDI. Listen by ear; metrics are guardrails only.','','## Aggregate','','| version | notes | chord-tone | chord+approach | density notes/sec | avg interval | big leap | repeat |','|---|---:|---:|---:|---:|---:|---:|---:|']
for v in VERS:
    ss=byv[v]; notes=sum(x['notes'] for x in ss)
    def w(k): return sum(x[k]*x['notes'] for x in ss)/notes if notes else 0
    lines.append(f"| {v} | {notes} | {w('chord_tone'):.3f} | {w('chord_or_approach'):.3f} | {w('density'):.2f} | {w('avg_interval'):.2f} | {w('big_leap'):.3f} | {w('repeat'):.3f} |")
lines += ['','## Per tune','','| version | tune | notes | dur | range | chord-tone | chord+approach | density | avg int | stepwise | big leap | audio |','|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['version']} | {r['tune']} | {r['notes']} | {r['duration']:.1f}s | {r['range']} | {r['chord_tone']:.3f} | {r['chord_or_approach']:.3f} | {r['density']:.2f} | {r['avg_interval']:.2f} | {r['stepwise']:.3f} | {r['big_leap']:.3f} | `{r['audio']}` |")
lines += ['','## Files','']
for v in VERS:
    lines.append(f"- {v} MIDI/JSON: `outputs/auditions_v68_compare/{v}/`")
    lines.append(f"- {v} audio: `outputs/auditions_v68_compare/audio_m4a/{v}/`")
report=BASE/'AUDITION_COMPARISON.md'
report.write_text('\n'.join(lines)+'\n')
print(report)
print('\n'.join(lines[:20]))
