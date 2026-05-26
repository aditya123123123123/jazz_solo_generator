# v6.8 Fix Variant Comparison

Base v6.8 earlier scored chord-tone 0.612 / chord+approach 0.881 on the matched audition set.

| variant | notes | chord-tone | chord+approach | density | avg interval | big leap | repeat |
|---|---:|---:|---:|---:|---:|---:|---:|
| bias_anchor_s2_np05 | 281 | 0.612 | 0.918 | 3.28 | 2.68 | 0.011 | 0.018 |
| bias_allbeat_s1_np025 | 273 | 0.703 | 0.930 | 3.23 | 3.06 | 0.026 | 0.019 |
| bias_allbeat_s15_np05 | 273 | 0.751 | 0.934 | 3.12 | 2.98 | 0.007 | 0.022 |
| timing_chord_color | 311 | 0.573 | 0.905 | 3.69 | 1.88 | 0.003 | 0.380 |
| timing_strict | 311 | 0.573 | 0.905 | 3.69 | 1.63 | 0.013 | 0.558 |

## Per tune

| variant | tune | notes | chord-tone | chord+approach | audio |
|---|---|---:|---:|---:|---|
| bias_anchor_s2_np05 | autumn_leaves | 69 | 0.667 | 0.913 | `outputs/auditions_v68_fix_test/audio_wav/bias_anchor_s2_np05/autumn_leaves.wav` |
| bias_anchor_s2_np05 | blues_F | 157 | 0.605 | 0.949 | `outputs/auditions_v68_fix_test/audio_wav/bias_anchor_s2_np05/blues_F.wav` |
| bias_anchor_s2_np05 | ii_V_I_C | 55 | 0.564 | 0.836 | `outputs/auditions_v68_fix_test/audio_wav/bias_anchor_s2_np05/ii_V_I_C.wav` |
| bias_allbeat_s1_np025 | autumn_leaves | 67 | 0.657 | 0.910 | `outputs/auditions_v68_fix_test/audio_wav/bias_allbeat_s1_np025/autumn_leaves.wav` |
| bias_allbeat_s1_np025 | blues_F | 151 | 0.702 | 0.960 | `outputs/auditions_v68_fix_test/audio_wav/bias_allbeat_s1_np025/blues_F.wav` |
| bias_allbeat_s1_np025 | ii_V_I_C | 55 | 0.764 | 0.873 | `outputs/auditions_v68_fix_test/audio_wav/bias_allbeat_s1_np025/ii_V_I_C.wav` |
| bias_allbeat_s15_np05 | autumn_leaves | 68 | 0.735 | 0.912 | `outputs/auditions_v68_fix_test/audio_wav/bias_allbeat_s15_np05/autumn_leaves.wav` |
| bias_allbeat_s15_np05 | blues_F | 150 | 0.767 | 0.980 | `outputs/auditions_v68_fix_test/audio_wav/bias_allbeat_s15_np05/blues_F.wav` |
| bias_allbeat_s15_np05 | ii_V_I_C | 55 | 0.727 | 0.836 | `outputs/auditions_v68_fix_test/audio_wav/bias_allbeat_s15_np05/ii_V_I_C.wav` |
| timing_chord_color | autumn_leaves | 80 | 0.472 | 0.847 | `outputs/auditions_v68_fix_test/audio_wav/timing_chord_color/autumn_leaves.wav` |
| timing_chord_color | blues_F | 172 | 0.584 | 0.955 | `outputs/auditions_v68_fix_test/audio_wav/timing_chord_color/blues_F.wav` |
| timing_chord_color | ii_V_I_C | 59 | 0.679 | 0.839 | `outputs/auditions_v68_fix_test/audio_wav/timing_chord_color/ii_V_I_C.wav` |
| timing_strict | autumn_leaves | 80 | 0.472 | 0.847 | `outputs/auditions_v68_fix_test/audio_wav/timing_strict/autumn_leaves.wav` |
| timing_strict | blues_F | 172 | 0.584 | 0.955 | `outputs/auditions_v68_fix_test/audio_wav/timing_strict/blues_F.wav` |
| timing_strict | ii_V_I_C | 59 | 0.679 | 0.839 | `outputs/auditions_v68_fix_test/audio_wav/timing_strict/ii_V_I_C.wav` |
