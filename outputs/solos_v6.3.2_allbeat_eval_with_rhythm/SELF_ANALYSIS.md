# Self-analysis of rhythm-section MIDIs

Tempo assumed: 120 BPM. Checks use JSON section chords as ground truth.

| file | tracks | backing end | actual end | solo harmony | bass roots | piano bad tones |
|---|---|---:|---:|---:|---:|---:|
| autumn_leaves_with_rhythm_swing.mid | Solo:117, Piano:120, Bass:80, Drums:220 | 40.0 | 40.0 | 66.7% chord/color (43.6% strict) | 16/16 | 0 |
| blues_F_with_rhythm_swing.mid | Solo:167, Piano:144, Bass:96, Drums:264 | 48.0 | 48.0 | 73.1% chord/color (44.3% strict) | 24/24 | 0 |
| ii_V_I_C_with_rhythm_swing.mid | Solo:61, Piano:72, Bass:48, Drums:132 | 24.0 | 24.0 | 88.5% chord/color (54.1% strict) | 8/8 | 0 |

## Notes

- Bass roots hit the expected root pitch class at every chord boundary in all files.
- Piano comping notes are all chord tones or the added 9th shell tone for the active section.
- Solo note counts and timing are preserved from the original MIDIs; only Piano/Bass/Drums tracks were added.
- The rhythm section loops the JSON form enough times to cover the full generated solo, so the backing does not stop early.
