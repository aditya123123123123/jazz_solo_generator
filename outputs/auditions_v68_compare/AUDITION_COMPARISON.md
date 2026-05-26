# v6.8 Audition Comparison

Matched generation settings: rhythm section=swing, phrase shaping, register continuity, phrase diversity max 3 with cadence preservation, bebop approach notes, target-contour section cadence.

Audio previews are WAV synthesized from the generated MIDI. Listen by ear; metrics are guardrails only.

## Aggregate

| version | notes | chord-tone | chord+approach | density notes/sec | avg interval | big leap | repeat |
|---|---:|---:|---:|---:|---:|---:|---:|
| v66 | 290 | 0.707 | 0.917 | 2.80 | 3.10 | 0.017 | 0.014 |
| v67 | 310 | 0.697 | 0.929 | 3.05 | 2.81 | 0.003 | 0.016 |
| v68 | 312 | 0.612 | 0.881 | 3.06 | 2.84 | 0.010 | 0.013 |

## Per tune

| version | tune | notes | dur | range | chord-tone | chord+approach | density | avg int | stepwise | big leap | audio |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| v66 | autumn_leaves | 112 | 45.7s | C3-C6 | 0.723 | 0.875 | 2.45 | 3.23 | 0.450 | 0.027 | `outputs/auditions_v68_compare/audio_wav/v66/autumn_leaves.wav` |
| v66 | blues_F | 135 | 44.1s | C3-D#5 | 0.674 | 0.941 | 3.06 | 3.01 | 0.485 | 0.015 | `outputs/auditions_v68_compare/audio_wav/v66/blues_F.wav` |
| v66 | ii_V_I_C | 43 | 14.8s | C3-C5 | 0.767 | 0.953 | 2.90 | 3.05 | 0.452 | 0.000 | `outputs/auditions_v68_compare/audio_wav/v66/ii_V_I_C.wav` |
| v67 | autumn_leaves | 126 | 42.5s | C3-C6 | 0.683 | 0.905 | 2.97 | 2.64 | 0.568 | 0.000 | `outputs/auditions_v68_compare/audio_wav/v67/autumn_leaves.wav` |
| v67 | blues_F | 139 | 46.6s | C3-G#5 | 0.719 | 0.957 | 2.98 | 2.96 | 0.522 | 0.007 | `outputs/auditions_v68_compare/audio_wav/v67/blues_F.wav` |
| v67 | ii_V_I_C | 45 | 12.9s | C4-A5 | 0.667 | 0.911 | 3.50 | 2.86 | 0.477 | 0.000 | `outputs/auditions_v68_compare/audio_wav/v67/ii_V_I_C.wav` |
| v68 | autumn_leaves | 128 | 42.1s | C3-F5 | 0.602 | 0.867 | 3.04 | 2.76 | 0.512 | 0.008 | `outputs/auditions_v68_compare/audio_wav/v68/autumn_leaves.wav` |
| v68 | blues_F | 140 | 43.5s | C3-C5 | 0.636 | 0.900 | 3.22 | 2.86 | 0.554 | 0.014 | `outputs/auditions_v68_compare/audio_wav/v68/blues_F.wav` |
| v68 | ii_V_I_C | 44 | 17.1s | C3-F5 | 0.568 | 0.864 | 2.57 | 2.98 | 0.558 | 0.000 | `outputs/auditions_v68_compare/audio_wav/v68/ii_V_I_C.wav` |

## Files

- v66 MIDI/JSON: `outputs/auditions_v68_compare/v66/`
- v66 audio: `outputs/auditions_v68_compare/audio_wav/v66/`
- v67 MIDI/JSON: `outputs/auditions_v68_compare/v67/`
- v67 audio: `outputs/auditions_v68_compare/audio_wav/v67/`
- v68 MIDI/JSON: `outputs/auditions_v68_compare/v68/`
- v68 audio: `outputs/auditions_v68_compare/audio_wav/v68/`
