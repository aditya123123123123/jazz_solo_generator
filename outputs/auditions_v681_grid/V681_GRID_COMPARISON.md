# v6.8.1 Inference Grid Comparison

Settings: density calibration threshold 0.6, register resample cap >4 edits or >30%, three resample attempts with best-attempt fallback, repeat guard max 2.

| variant | notes | chord-tone | chord+approach | density | avg interval | big leap | repeat | reg edits | reg resampled | density edits | repeat edits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iivi_s1p0_np0p25 | 326 | 0.687 | 0.929 | 4.33 | 2.79 | 0.000 | 0.003 | 111 | 14 | 13 | 0 |
| iivi_s1p0_np0p5 | 433 | 0.737 | 0.933 | 4.73 | 2.99 | 0.009 | 0.019 | 145 | 19 | 0 | 0 |
| iivi_s1p25_np0p25 | 377 | 0.740 | 0.928 | 4.23 | 3.14 | 0.016 | 0.011 | 166 | 21 | 0 | 0 |
| iivi_s1p25_np0p5 | 385 | 0.784 | 0.945 | 4.15 | 3.35 | 0.016 | 0.008 | 152 | 19 | 0 | 0 |
| iivi_s1p5_np0p25 | 258 | 0.760 | 0.938 | 3.78 | 2.96 | 0.004 | 0.016 | 93 | 19 | 0 | 0 |
| iivi_s1p5_np0p5 | 392 | 0.753 | 0.934 | 4.36 | 3.00 | 0.005 | 0.008 | 159 | 21 | 9 | 0 |
| iivi_s1p75_np0p25 | 387 | 0.760 | 0.907 | 4.61 | 3.11 | 0.008 | 0.008 | 131 | 19 | 12 | 0 |
| iivi_s1p75_np0p5 | 354 | 0.729 | 0.921 | 4.62 | 3.01 | 0.020 | 0.020 | 141 | 16 | 4 | 0 |

## Per tune

| variant | tune | notes | chord-tone | chord+approach | repeat | reg edits | reg resampled | density edits | repeat edits | audio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| iivi_s1p0_np0p25 | autumn_leaves | 64 | 0.781 | 0.922 | 0.016 | 0 | 0 | 13 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p0_np0p25/autumn_leaves.wav` |
| iivi_s1p0_np0p25 | blues_F | 206 | 0.660 | 0.961 | 0.000 | 98 | 11 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p0_np0p25/blues_F.wav` |
| iivi_s1p0_np0p25 | ii_V_I_C | 56 | 0.679 | 0.821 | 0.000 | 13 | 3 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p0_np0p25/ii_V_I_C.wav` |
| iivi_s1p0_np0p5 | autumn_leaves | 156 | 0.699 | 0.904 | 0.026 | 39 | 6 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p0_np0p5/autumn_leaves.wav` |
| iivi_s1p0_np0p5 | blues_F | 221 | 0.742 | 0.959 | 0.014 | 86 | 10 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p0_np0p5/blues_F.wav` |
| iivi_s1p0_np0p5 | ii_V_I_C | 56 | 0.821 | 0.911 | 0.018 | 20 | 3 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p0_np0p5/ii_V_I_C.wav` |
| iivi_s1p25_np0p25 | autumn_leaves | 99 | 0.768 | 0.909 | 0.020 | 55 | 7 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p25_np0p25/autumn_leaves.wav` |
| iivi_s1p25_np0p25 | blues_F | 222 | 0.707 | 0.950 | 0.009 | 94 | 11 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p25_np0p25/blues_F.wav` |
| iivi_s1p25_np0p25 | ii_V_I_C | 56 | 0.821 | 0.875 | 0.000 | 17 | 3 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p25_np0p25/ii_V_I_C.wav` |
| iivi_s1p25_np0p5 | autumn_leaves | 115 | 0.774 | 0.922 | 0.009 | 59 | 7 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p25_np0p5/autumn_leaves.wav` |
| iivi_s1p25_np0p5 | blues_F | 215 | 0.795 | 0.977 | 0.009 | 88 | 10 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p25_np0p5/blues_F.wav` |
| iivi_s1p25_np0p5 | ii_V_I_C | 55 | 0.764 | 0.873 | 0.000 | 5 | 2 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p25_np0p5/ii_V_I_C.wav` |
| iivi_s1p5_np0p25 | autumn_leaves | 97 | 0.773 | 0.969 | 0.000 | 40 | 6 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p5_np0p25/autumn_leaves.wav` |
| iivi_s1p5_np0p25 | blues_F | 106 | 0.745 | 0.962 | 0.029 | 49 | 10 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p5_np0p25/blues_F.wav` |
| iivi_s1p5_np0p25 | ii_V_I_C | 55 | 0.764 | 0.836 | 0.019 | 4 | 3 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p5_np0p25/ii_V_I_C.wav` |
| iivi_s1p5_np0p5 | autumn_leaves | 164 | 0.750 | 0.945 | 0.006 | 73 | 8 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p5_np0p5/autumn_leaves.wav` |
| iivi_s1p5_np0p5 | blues_F | 173 | 0.757 | 0.960 | 0.012 | 63 | 10 | 9 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p5_np0p5/blues_F.wav` |
| iivi_s1p5_np0p5 | ii_V_I_C | 55 | 0.745 | 0.818 | 0.000 | 23 | 3 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p5_np0p5/ii_V_I_C.wav` |
| iivi_s1p75_np0p25 | autumn_leaves | 158 | 0.734 | 0.861 | 0.006 | 80 | 7 | 5 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p75_np0p25/autumn_leaves.wav` |
| iivi_s1p75_np0p25 | blues_F | 174 | 0.770 | 0.954 | 0.006 | 40 | 9 | 7 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p75_np0p25/blues_F.wav` |
| iivi_s1p75_np0p25 | ii_V_I_C | 55 | 0.800 | 0.891 | 0.019 | 11 | 3 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p75_np0p25/ii_V_I_C.wav` |
| iivi_s1p75_np0p5 | autumn_leaves | 96 | 0.750 | 0.896 | 0.021 | 41 | 6 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p75_np0p5/autumn_leaves.wav` |
| iivi_s1p75_np0p5 | blues_F | 203 | 0.709 | 0.951 | 0.025 | 87 | 8 | 4 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p75_np0p5/blues_F.wav` |
| iivi_s1p75_np0p5 | ii_V_I_C | 55 | 0.764 | 0.855 | 0.000 | 13 | 2 | 0 | 0 | `outputs/auditions_v681_grid/audio_wav/iivi_s1p75_np0p5/ii_V_I_C.wav` |
