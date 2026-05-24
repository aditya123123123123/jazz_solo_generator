# v6.3 RunPod Launch Package

Commit: a8113c5
fix(harmonic): mask unknown chords from harmonic loss

Changes:
- Mask target_chord_id < 5 in harmonic_penalty
- Only apply loss on real chords (valid_strong_fraction ~0.58)
- Expected tone_pc.sum.mean ~3.8 on valid chords

Kill conditions implemented:
- NaN/Inf harm_l
- harm_l > 10x baseline
- valid_strong_fraction out of [0.40, 0.75]

See launch.sh and config_v6.3.yaml
