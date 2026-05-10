"""Standard jazz swing drum pattern generator.

Returns absolute-timed MIDI events on the drum channel:
  Ride cymbal (51): swing 8th notes (long=0.67 beat, short=0.33 beat) every beat
  Hi-hat      (42): beats 2 and 4
  Kick drum   (36): beat 1, velocity 70
  Snare drum  (38): beats 2 and 4, velocity 85

Event tuple: (start_sec, pitch_midi, duration_sec, velocity)
"""

RIDE  = 51
HIHAT = 42
KICK  = 36
SNARE = 38


def generate_drum_pattern(total_beats, tempo_bpm=120):
    """
    Returns MIDI note events for a standard jazz swing drum pattern.
    MIDI drum channel (channel 10, 0-indexed channel 9).
    """
    beat_dur = 60.0 / tempo_bpm
    events = []

    long_dur  = 0.67 * beat_dur
    short_dur = 0.33 * beat_dur
    short_off = 0.67 * beat_dur  # offset of swung 8th within the beat

    for beat in range(total_beats):
        t_beat = beat * beat_dur
        beat_in_bar = beat % 4  # 0=beat1, 1=beat2, 2=beat3, 3=beat4

        # Ride: continuous swing 8th-note stream
        events.append((t_beat,             RIDE, long_dur,  80))
        events.append((t_beat + short_off, RIDE, short_dur, 65))

        # Kick on beat 1
        if beat_in_bar == 0:
            events.append((t_beat, KICK, beat_dur * 0.5, 70))

        # Hi-hat (foot) and snare on beats 2 and 4
        if beat_in_bar in (1, 3):
            events.append((t_beat, HIHAT, beat_dur * 0.5, 60))
            events.append((t_beat, SNARE, beat_dur * 0.5, 85))

    events.sort(key=lambda e: e[0])
    return events
