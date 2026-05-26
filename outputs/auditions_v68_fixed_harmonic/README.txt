v6.8 fixed harmonic audition set

Issue: the raw v6.8 checkpoint generated plausible lines, but the first matched audition sample had weaker section-level harmonic alignment than v6.6/v6.7: chord-tone 0.612 and chord+approach 0.881.

Fix used for this handoff: keep the trained v6.8 checkpoint, but enable inference-time all-beat chord-tone steering with strength 1.5 and non-chord penalty 0.5, while preserving the same rhythm-section, phrase-shaping, register-continuity, phrase-diversity, bebop approach, and target-contour section cadence settings.

Fixed metrics on the matched 3-tune set: chord-tone 0.751, chord+approach 0.934, density 3.12 notes/sec, avg interval 2.98, big-leap ratio 0.007.

Listen by ear; metrics are guardrails, not the final judge.
