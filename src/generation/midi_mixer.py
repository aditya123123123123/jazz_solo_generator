"""Combine melody, walking bass, and drum streams into a multitrack MIDI file.

Track layout:
  Track 0: tempo meta
  Track 1: melody  — channel 0, program 65 (alto sax)
  Track 2: bass    — channel 1, program 32 (acoustic bass)
  Track 3: drums   — channel 9 (no program change required)

Event formats accepted by mix_to_midi:
  solo_events: list of (start_sec, pitch_midi, duration_sec, velocity)
  bass_events: list of (pitch_midi, duration_sec, is_rest)         (sequential)
  drum_events: list of (start_sec, pitch_midi, duration_sec, velocity)
"""
import mido

TICKS_PER_BEAT = 480


def _sec_to_tick(seconds, tempo_bpm):
    return int(round(seconds * tempo_bpm / 60.0 * TICKS_PER_BEAT))


def _bass_to_absolute(bass_events, velocity=75):
    """Convert (pitch, dur, is_rest) stream to (start, pitch, dur, vel)."""
    out = []
    t = 0.0
    for pitch, dur, is_rest in bass_events:
        if not is_rest:
            out.append((t, pitch, dur, velocity))
        t += dur
    return out


def _write_events(track, events, channel, tempo_bpm):
    """Append note_on/note_off messages (delta-time) for a list of absolute events."""
    msgs = []
    for start, pitch, dur, vel in events:
        on_tick  = _sec_to_tick(start,       tempo_bpm)
        off_tick = _sec_to_tick(start + dur, tempo_bpm)
        if off_tick <= on_tick:
            off_tick = on_tick + 1
        msgs.append((on_tick,  1, pitch, vel))   # 1 = on  (sort key tiebreak: off before on)
        msgs.append((off_tick, 0, pitch, 0))

    msgs.sort(key=lambda m: (m[0], m[1]))

    last_tick = 0
    for tick, kind, pitch, vel in msgs:
        delta = tick - last_tick
        if kind == 1:
            track.append(mido.Message("note_on",  note=pitch, velocity=vel, time=delta, channel=channel))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0,   time=delta, channel=channel))
        last_tick = tick


def mix_to_midi(solo_events, bass_events, drum_events, output_path, tempo_bpm=120):
    """
    Combines three event streams into a single multitrack MIDI file.
    Track 0: melody (channel 0, instrument 65 = alto sax)
    Track 1: bass   (channel 1, instrument 32 = acoustic bass)
    Track 2: drums  (channel 9, no instrument change)
    """
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    # --- Tempo meta track -------------------------------------------------
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    mid.tracks.append(meta)

    # --- Melody (alto sax, channel 0) -------------------------------------
    melody = mido.MidiTrack()
    melody.append(mido.Message("program_change", program=65, channel=0, time=0))
    _write_events(melody, list(solo_events), channel=0, tempo_bpm=tempo_bpm)
    mid.tracks.append(melody)

    # --- Bass (acoustic bass, channel 1) ----------------------------------
    bass = mido.MidiTrack()
    bass.append(mido.Message("program_change", program=32, channel=1, time=0))
    _write_events(bass, _bass_to_absolute(bass_events), channel=1, tempo_bpm=tempo_bpm)
    mid.tracks.append(bass)

    # --- Drums (channel 9 — GM drum kit) ----------------------------------
    drums = mido.MidiTrack()
    _write_events(drums, list(drum_events), channel=9, tempo_bpm=tempo_bpm)
    mid.tracks.append(drums)

    mid.save(str(output_path))


# ---------------------------------------------------------------------------
# Smoke test: ii-V-I at 160 BPM
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from src.generation.walking_bass import generate_walking_bass
    from src.generation.drum_pattern import generate_drum_pattern

    progression = [("Dm7", 4), ("G7", 4), ("Cj7", 8)]
    tempo = 160

    bass = generate_walking_bass(progression, tempo_bpm=tempo)
    total_beats = sum(b for _, b in progression)
    drums = generate_drum_pattern(total_beats, tempo_bpm=tempo)

    out = Path(__file__).resolve().parents[2] / "outputs" / "test_rhythm_section.mid"
    out.parent.mkdir(parents=True, exist_ok=True)

    mix_to_midi([], bass, drums, str(out), tempo_bpm=tempo)

    duration = total_beats * 60.0 / tempo
    bass_notes = sum(1 for _, _, r in bass if not r)

    print(f"Wrote: {out}")
    print(f"  Solo notes : 0")
    print(f"  Bass notes : {bass_notes}")
    print(f"  Drum events: {len(drums)}")
    print(f"  Duration   : {duration:.2f} sec  ({total_beats} beats @ {tempo} BPM)")
