from __future__ import annotations
import json, statistics as stats
from collections import defaultdict
from pathlib import Path
import sys, numpy as np, pretty_midi
from scipy.io import wavfile
REPO=Path.cwd(); sys.path.insert(0,str(REPO))
from src.generation.chord_utils import chord_tones, parse_chord
BASE=REPO/'outputs'/'auditions_v68_fix_test'
VARIANTS=["bias_anchor_s2_np05","bias_allbeat_s1_np025","bias_allbeat_s15_np05","timing_chord_color","timing_strict"]

def tones(sym):
    p=parse_chord(sym); return set(chord_tones(*p)) if p else set()
def approaches(sym):
    t=tones(sym); return {(x-1)%12 for x in t}|{(x+1)%12 for x in t}
def synth(mid,out):
    out.parent.mkdir(parents=True,exist_ok=True)
    pm=pretty_midi.PrettyMIDI(str(mid)); audio=pm.synthesize(fs=44100)
    if audio.size==0: audio=np.zeros(44100,dtype=np.float32)
    peak=float(np.max(np.abs(audio))) or 1.0
    wavfile.write(str(out),44100,np.int16(np.clip(audio/peak*.95,-1,1)*32767))
def metrics(js,mid):
    d=json.loads(js.read_text()); pm=pretty_midi.PrettyMIDI(str(mid)); solo=pm.instruments[0] if pm.instruments else None
    notes=sorted((solo.notes if solo else []),key=lambda n:(n.start,n.pitch)); pitches=[n.pitch for n in notes]
    intervals=[abs(b-a) for a,b in zip(pitches,pitches[1:])]
    total=hit=either=0
    for sec in d.get('sections',[]):
        ct=tones(sec.get('chord','')); ap=approaches(sec.get('chord',''))
        for p in [int(x) for x in sec.get('pitches',[]) if 0<=int(x)<=127]:
            total+=1; pc=p%12; hit += pc in ct; either += (pc in ct) or (pc in ap)
    dur=max((n.end for n in notes),default=0) or 1
    return dict(notes=len(notes), duration=dur, chord_tone=hit/total if total else 0, chord_or_approach=either/total if total else 0, density=len(notes)/dur, avg_interval=sum(intervals)/len(intervals) if intervals else 0, big_leap=sum(i>=10 for i in intervals)/len(intervals) if intervals else 0, repeat=sum(i==0 for i in intervals)/len(intervals) if intervals else 0)
rows=[]
for v in VARIANTS:
    for mid in sorted((BASE/v).glob('*.mid')):
        tune=mid.name.split('_rhythm_')[0]
        wav=BASE/'audio_wav'/v/(tune+'.wav')
        synth(mid,wav)
        r=metrics(mid.with_suffix('.json'),mid); r.update(variant=v,tune=tune,audio=str(wav.relative_to(REPO)))
        rows.append(r)
by=defaultdict(list)
for r in rows: by[r['variant']].append(r)
lines=['# v6.8 Fix Variant Comparison','','Base v6.8 earlier scored chord-tone 0.612 / chord+approach 0.881 on the matched audition set.','','| variant | notes | chord-tone | chord+approach | density | avg interval | big leap | repeat |','|---|---:|---:|---:|---:|---:|---:|---:|']
for v in VARIANTS:
    ss=by[v]; notes=sum(x['notes'] for x in ss)
    def w(k): return sum(x[k]*x['notes'] for x in ss)/notes if notes else 0
    lines.append(f"| {v} | {notes} | {w('chord_tone'):.3f} | {w('chord_or_approach'):.3f} | {w('density'):.2f} | {w('avg_interval'):.2f} | {w('big_leap'):.3f} | {w('repeat'):.3f} |")
lines += ['','## Per tune','','| variant | tune | notes | chord-tone | chord+approach | audio |','|---|---|---:|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['variant']} | {r['tune']} | {r['notes']} | {r['chord_tone']:.3f} | {r['chord_or_approach']:.3f} | `{r['audio']}` |")
out=BASE/'FIX_VARIANT_COMPARISON.md'; out.write_text('\n'.join(lines)+'\n')
print(out)
for l in lines[:9]: print(l)
print('rows',len(rows),'wav',len(list((BASE/'audio_wav').rglob('*.wav'))))
