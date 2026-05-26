v6.8.1 selected audition set

Selected variant: iivi_s1p5_np0p25

Why this one: it is the best balanced v6.8.1 output. It improves raw/fixed v6.8 harmonic alignment while avoiding the over-busy behavior of the metric-max iivi_s1p25_np0p5 variant.

Generation controls:
- v6.8 curated vocab fine-tune checkpoint
- all-beat chord-tone bias strength 1.5
- ii-V-I chord-tone bias override 1.5
- non-chord penalty 0.25
- rhythm density calibration with min-ratio 0.6
- register-continuity resampling if >4 edits or >30% of sounding notes, up to 3 attempts, keeping the lowest-edit attempt
- repeat guard max 2 consecutive identical sounding pitches
- rhythm section swing, phrase shaping, phrase diversity max 3, cadence enforcement, bebop approach notes

Aggregate metrics:
- notes: 258
- chord-tone: 0.760
- chord+approach: 0.938
- density: 3.78
- avg interval: 2.96
- big leap: 0.004
- repeat: 0.016

Notable fix: Autumn Leaves opening changed from the previous dead Cm7 opening (1 distinct pitch, density 0.75) to a usable phrase (7 distinct pitches, density 2.0).
