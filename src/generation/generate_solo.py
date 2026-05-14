"""
End-to-end jazz solo generation pipeline.

For each chord in a progression:
  1. PhrasePlanner  (chord sequence → phrase token)
  2. NoteExecutor   (chord + phrase token → note sequence)
  3. Concatenate notes, export to MIDI.
"""
import argparse
import json
import re
from pathlib import Path

import pretty_midi
import torch

from src.models.note_executor import NoteExecutor
from src.models.phrase_planner import PhrasePlanner
from src.tokenization import ArtistTokenizer, ChordTokenizer, NoteTokenizer, PhraseTokenizer

_REPO = Path(__file__).parents[2]

# ---------------------------------------------------------------------------
# Chord notation normalizer: lead-sheet → WJazzD
# ---------------------------------------------------------------------------
# WJazzD uses "X-7" for minor-7 chords.  Half-dim (Xm7b5) is already correct.
_MINOR7_RE = re.compile(r"^([A-G][#b]?)m7(?!b5)")

def normalize_chord(symbol: str) -> str:
    """Translate standard lead-sheet notation to WJazzD vocabulary notation."""
    return _MINOR7_RE.sub(r"\1-7", symbol)


# ---------------------------------------------------------------------------
# Pitch clamping: octave-transpose into [48, 84] (C3–C6)
# ---------------------------------------------------------------------------
PITCH_LO = 48
PITCH_HI = 84

def clamp_pitch(pitch: int, lo: int = PITCH_LO, hi: int = PITCH_HI) -> int:
    """Transpose pitch by octaves until it falls within [lo, hi]."""
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


CHECKPOINTS      = _REPO / "checkpoints"
SOLOS_DIR        = _REPO / "outputs" / "solos"
BEAT_DURATION    = 0.5    # seconds per beat at 120 BPM
N_NOTES          = 16     # notes generated per chord section
MIN_NOTE_DUR     = 0.05   # seconds — clamp very short decoded durations
REST_THRESH_LONG = 0.8    # notes longer than this get a rest inserted after
REST_SHORTEN     = 0.35   # how much to shorten a long note before the rest
BOUNDARY_DUR_MIN = 0.3    # threshold for boundary rest insertion
BOUNDARY_REST    = 0.25   # duration of injected boundary rest


# ---------------------------------------------------------------------------
# PHRASE_00 remapping — steers fallback phrase tokens toward family-appropriate ones
# ---------------------------------------------------------------------------

def remap_phrase_fallback(phrase_int: int, chord_wjazz: str, phrase_tok) -> int:
    """Replace PHRASE_00 with a chord-family-appropriate phrase cluster."""
    if phrase_tok.decode(phrase_int) != "PHRASE_00":
        return phrase_int
    if any(x in chord_wjazz for x in ("m7b5", "o7", "-7", "-6")):
        return phrase_tok.encode("PHRASE_18")   # minor family
    if any(x in chord_wjazz for x in ("j7", "maj")):
        return phrase_tok.encode("PHRASE_52")   # major family
    if chord_wjazz.endswith("7"):
        return phrase_tok.encode("PHRASE_57")   # dominant family
    return phrase_tok.encode("PHRASE_18")       # default


# ---------------------------------------------------------------------------
# Rest injection — musical breathing within and between sections
# ---------------------------------------------------------------------------

def _inject_rests(section_events: list) -> list:
    """Insert rests after long notes (>0.8 s) to create musical breathing."""
    result = []
    for pitch, dur, is_rest in section_events:
        if not is_rest and dur > REST_THRESH_LONG:
            result.append((pitch, dur - REST_SHORTEN, False))
            result.append((0, REST_SHORTEN, True))
        else:
            result.append((pitch, dur, is_rest))
    return result

PROGRESSIONS = {
    "ii_V_I_C": [
        ("Dm7", 4), ("G7", 4), ("Cj7", 8), ("Cj7", 8),
    ],
    "blues_F": [
        ("F7", 4), ("Bb7", 4), ("F7", 4), ("F7", 4),
        ("Bb7", 4), ("Bb7", 4), ("F7", 4), ("D7", 4),
        ("Gm7", 4), ("C7", 4), ("F7", 4), ("C7", 4),
    ],
    "autumn_leaves": [
        ("Cm7", 4), ("F7", 4), ("Bbj7", 4), ("Ebj7", 4),
        ("Am7b5", 4), ("D7", 4), ("Gm7", 8), ("Gm7", 8),
    ],
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(chord_tok, artist_tok):
    planner = PhrasePlanner(
        chord_vocab_size=chord_tok.vocab_size,
        artist_vocab_size=artist_tok.vocab_size,
    )
    planner.load_state_dict(
        torch.load(CHECKPOINTS / "phrase_planner_best.pt", map_location="cpu")
    )
    planner.eval()

    executor = NoteExecutor(
        chord_vocab_size=chord_tok.vocab_size,
        artist_vocab_size=artist_tok.vocab_size,
        dur_vocab_size=19,
        dropout=0.2,
    )
    executor.load_state_dict(
        torch.load(CHECKPOINTS / "note-executor-v5-best.pt", map_location="cpu"), strict=False
    )
    executor.eval()

    return planner, executor


# ---------------------------------------------------------------------------
# Per-chord generation helpers
# ---------------------------------------------------------------------------

def _plan_phrase(planner, chord_ids, artist_id, fallback=3, temperature=1.0):
    with torch.no_grad():
        tokens = planner.generate(chord_ids, artist_id, max_len=8, temperature=temperature)
    return tokens[0] if tokens else fallback


def _generate_notes(executor, chord_ids, phrase_id, artist_id,
                    prefix_pitch=None, prefix_dur=None, prefix_rest=None,
                    temperature=1.0, window=8):
    with torch.no_grad():
        raw = executor.generate(
            chord_ids, phrase_id, artist_id, n_notes=N_NOTES,
            prefix_pitch=prefix_pitch,
            prefix_dur=prefix_dur,
            prefix_rest=prefix_rest,
            temperature=temperature,
            window=window,
        )
    return raw


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_solo(progression, artist_name="Charlie Parker",
                  chord_tok=None, note_tok=None,
                  phrase_tok=None, artist_tok=None,
                  planner=None, executor=None,
                  window=16, temperature=1.0):
    """
    progression : list of (chord_str, beats)
    Returns (note_events, chord_summaries, unknown_chords)
      note_events : list of (pitch_midi, dur_sec, is_rest)
      chord_summaries : per-chord metadata
      unknown_chords  : chord symbols not in vocab
    """
    artist_id = torch.tensor([artist_tok.encode(artist_name)], dtype=torch.long)

    unknown_chords  = []
    note_events     = []
    chord_summaries = []

    # Rolling cross-chord context: last WINDOW note tokens from previous section
    ctx_pitch: list = []
    ctx_dur:   list = []
    ctx_rest:  list = []

    for chord_str, beats in progression:
        wjazz_str = normalize_chord(chord_str)
        chord_id  = chord_tok.encode(wjazz_str)
        if chord_id == chord_tok.UNK:
            unknown_chords.append(chord_str)

        chord_ids = torch.tensor([[chord_id]], dtype=torch.long)

        # ---- PhrasePlanner + PHRASE_00 remapping ----
        phrase_int_raw = _plan_phrase(planner, chord_ids, artist_id, temperature=temperature)
        phrase_int     = remap_phrase_fallback(phrase_int_raw, wjazz_str, phrase_tok)
        phrase_remapped = phrase_int != phrase_int_raw
        phrase_label = phrase_tok.decode(phrase_int)
        phrase_id    = torch.tensor([phrase_int], dtype=torch.long)

        # ---- NoteExecutor (with cross-chord prefix) ----
        pfx_p = ctx_pitch[-window:] or None
        pfx_d = ctx_dur[-window:]   or None
        pfx_r = ctx_rest[-window:]  or None
        raw_notes = _generate_notes(
            executor, chord_ids, phrase_id, artist_id,
            prefix_pitch=pfx_p, prefix_dur=pfx_d, prefix_rest=pfx_r,
            temperature=temperature,
            window=window,
        )

        section_events = []
        pitches_midi   = []
        n_clamped      = 0
        dur_fallback   = 0
        for p_tok, d_tok, r_tok in raw_notes:
            # Accumulate rolling token context for next chord section
            ctx_pitch.append(p_tok)
            ctx_dur.append(d_tok)
            ctx_rest.append(r_tok)
            raw_pitch  = note_tok.decode_pitch(p_tok)
            pitch_midi = clamp_pitch(raw_pitch) if 0 <= raw_pitch <= 127 else raw_pitch
            if pitch_midi != raw_pitch:
                n_clamped += 1
            try:
                _d = note_tok.decode_duration(d_tok, 120)
            except ValueError:
                _d = 0.25  # fallback for special/invalid duration tokens
                dur_fallback += 1
            dur_sec  = max(_d, MIN_NOTE_DUR)
            is_rest  = bool(r_tok)
            section_events.append((pitch_midi, dur_sec, is_rest))
            if 0 <= pitch_midi <= 127:
                pitches_midi.append(pitch_midi)

        # --- Rest injection (within section) ---
        section_events = _inject_rests(section_events)

        # --- Boundary rest (between sections) ---
        if note_events and section_events:
            last_p, last_d, last_r = section_events[-1]
            if not last_r and last_d < BOUNDARY_DUR_MIN:
                shortened = max(last_d - BOUNDARY_REST + 0.05, MIN_NOTE_DUR)
                section_events[-1] = (last_p, shortened, last_r)
                section_events.append((0, BOUNDARY_REST, True))

        note_events.extend(section_events)
        chord_summaries.append({
            "chord":           chord_str,
            "wjazz":           wjazz_str,
            "beats":           beats,
            "in_vocab":        chord_id != chord_tok.UNK,
            "phrase_token":    phrase_label,
            "phrase_remapped": phrase_remapped,
            "phrase_original": phrase_tok.decode(phrase_int_raw) if phrase_remapped else None,
            "n_notes":         16,           # always 16 model-generated notes
            "n_clamped":       n_clamped,
            "dur_fallback":    dur_fallback,
            "pitches":         pitches_midi,
        })

    return note_events, chord_summaries, unknown_chords


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

def export_midi(note_events, output_path, tempo=120):
    """Write note_events to a MIDI file. Returns total duration in seconds."""
    pm    = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    piano = pretty_midi.Instrument(program=0, name="Piano")

    t = 0.0
    for pitch_midi, dur_sec, is_rest in note_events:
        if not is_rest and 0 <= pitch_midi <= 127 and dur_sec > 0:
            piano.notes.append(pretty_midi.Note(
                velocity=64,
                pitch=int(pitch_midi),
                start=t,
                end=t + dur_sec,
            ))
        t += dur_sec

    pm.instruments.append(piano)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
    return t


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(note_events, summaries, output_path, name=None, tempo_bpm=120):
    """Save note events and per-chord metadata as JSON."""
    output_path = Path(output_path)
    data = {
        "name":        name or output_path.stem,
        "tempo_bpm":   tempo_bpm,
        "total_notes": len(note_events),
        "notes": [
            {"pitch": int(p), "duration_sec": round(float(d), 4), "is_rest": bool(r)}
            for p, d, r in note_events
        ],
        "sections": [
            {
                "chord":          s["chord"],
                "beats":          s["beats"],
                "phrase":         s["phrase_token"],
                "n_notes":        s["n_notes"],
                "clamped":        s.get("n_clamped", 0),
                "distinct_pitches": len(set(s["pitches"])),
                "dur_fallback":   s.get("dur_fallback", 0),
            }
            for s in summaries
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _is_plausible(pitches, min_distinct=3, lo=48, hi=96):
    """Rough musicality check: diverse enough pitches in a reasonable range."""
    if len(pitches) < 4:
        return False, "too few notes"
    in_range = [p for p in pitches if lo <= p <= hi]
    distinct = len(set(pitches))
    if len(in_range) / len(pitches) < 0.7:
        return False, f"out-of-range pitches ({len(pitches)-len(in_range)}/{len(pitches)})"
    if distinct < min_distinct:
        return False, f"only {distinct} distinct pitches"
    return True, "OK"


def report(name, summaries, unknown_chords, path, total_dur, note_events):
    sounding = [(p, d, r) for p, d, r in note_events if not r and 0 <= p <= 127]
    all_pitches = [p for p, d, r in sounding]
    total_clamped  = sum(s.get("n_clamped", 0) for s in summaries)
    total_fallback = sum(s.get("dur_fallback", 0) for s in summaries)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  File        : {path}")
    print(f"  Total notes : {len(sounding)}")
    print(f"  Total dur   : {total_dur:.1f}s")
    if all_pitches:
        print(f"  Pitch range : {min(all_pitches)} – {max(all_pitches)} MIDI  "
              f"({pretty_midi.note_number_to_name(min(all_pitches))}–"
              f"{pretty_midi.note_number_to_name(max(all_pitches))})")
    print(f"  Clamped     : {total_clamped} notes")
    print(f"  Dur fallback: {total_fallback}")
    if unknown_chords:
        print(f"  ⚠ Unknown   : {', '.join(unknown_chords)}")
    else:
        print(f"  Vocab       : all chords found")
    print()
    print(f"  {'Chord':<10} {'Beats':>5}  {'Phrase':<12}  {'Notes':>5}  {'Clamped':>7}  {'Distinct':>8}  {'DurFall':>7}  Plausible")
    print(f"  {'-'*80}")
    for s in summaries:
        distinct = len(set(s["pitches"]))
        ok, reason = _is_plausible(s["pitches"])
        flag = "✓" if ok else f"✗ ({reason})"
        print(f"  {s['chord']:<10} {s['beats']:>5}  {s['phrase_token']:<12}  "
              f"{s['n_notes']:>5}  {s.get('n_clamped',0):>7}  {distinct:>8}  "
              f"{s.get('dur_fallback',0):>7}  {flag}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate jazz solos with chord progressions.")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=SOLOS_DIR, dest="out_dir")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    chord_tok  = ChordTokenizer.from_json()
    note_tok   = NoteTokenizer.from_json()
    phrase_tok = PhraseTokenizer()
    artist_tok = ArtistTokenizer()

    print("Loading models …")
    planner, executor = load_models(chord_tok, artist_tok)
    print("Models loaded.")

    for name, progression in PROGRESSIONS.items():
        note_events, summaries, unknowns = generate_solo(
            progression,
            chord_tok=chord_tok, note_tok=note_tok,
            phrase_tok=phrase_tok, artist_tok=artist_tok,
            planner=planner, executor=executor,
            window=args.window, temperature=args.temperature,
        )
        stem = f"{name}__w{args.window}_t{args.temperature:.1f}"
        midi_out = args.out_dir / f"{stem}.mid"
        json_out = args.out_dir / f"{stem}.json"
        total_dur = export_midi(note_events, midi_out)
        export_json(note_events, summaries, json_out, name=name)
        report(name, summaries, unknowns, midi_out, total_dur, note_events)

    print(f"\nFiles saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
