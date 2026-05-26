v6.8 expanded-song audition set

This is still the v6.8 checkpoint, not a new trained model version.

Checkpoint:
checkpoints/v6.8_curated_vocab_finetune_best.pt

Generation controls:
- all-beat chord-tone bias strength 1.5
- ii-V-I chord-tone bias override 1.5
- non-chord penalty 0.25
- rhythm density calibration min-ratio 0.6
- register-continuity resample cap: >4 edits or >30% of sounding notes, 3 attempts
- repeat guard max 2 consecutive identical sounding pitches
- swing rhythm section, phrase shaping, phrase diversity max 3, cadence enforcement, bebop approach notes

Default generation now includes 8 probes:
- ii_V_I_C
- blues_F
- autumn_leaves
- rhythm_changes_Bb
- all_the_things_you_are
- giant_steps_cycle
- minor_blues_C
- modal_so_what_Dm
