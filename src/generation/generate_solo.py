"""
End-to-end jazz solo generation pipeline.

For each chord in a progression:
  1. PhrasePlanner  (chord sequence → phrase token)
  2. NoteExecutor   (chord + phrase token → note sequence)
  3. Concatenate notes, export to MIDI.
"""
import argparse
import json
import logging
import random
import re
from pathlib import Path

import pretty_midi
import torch

from src.generation.phrase_features import PhraseFeature, load_phrase_features, phrase_feature_to_dict
from src.generation.rhythm_section import generate_rhythm_section
from src.models.note_executor import NoteExecutor
from src.models.phrase_planner import PhrasePlanner
from src.tokenization import ArtistTokenizer, ChordTokenizer, NoteTokenizer, PhraseTokenizer

logger = logging.getLogger(__name__)

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

def _load_note_executor_with_compat(path, chord_tok, artist_tok, note_tok, dropout=0.2):
    import math
    path = Path(path)
    if not path.exists():
        legacy = CHECKPOINTS / "note-executor-v5-best.pt"
        if legacy.exists():
            logger.info("note_executor checkpoint: %s missing, falling back to %s",
                        path.name, legacy.name)
            path = legacy
        else:
            raise FileNotFoundError(
                f"No note_executor checkpoint found at {path} or {legacy}"
            )
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    ckpt_dur_vocab = state["dur_embed.weight"].shape[0]
    has_tempo = "tempo_embed.weight" in state

    if ckpt_dur_vocab == 19 and not has_tempo:
        logger.info("note_executor checkpoint: legacy v5 detected "
                    "(dur_vocab=19, no tempo_embed) — using compat path")
        executor = NoteExecutor(
            chord_vocab_size=chord_tok.vocab_size,
            artist_vocab_size=artist_tok.vocab_size,
            dur_vocab_size=19,
            dropout=dropout,
        )
        result = executor.load_state_dict(state, strict=False)
        expected_missing = {
            "tempo_embed.weight",
            "tempo_embed.bias",
            "phrase_pos_embed.weight",
        }
        missing = set(result.missing_keys)
        unexpected = set(result.unexpected_keys)
        if missing != expected_missing or unexpected:
            raise RuntimeError(
                f"compat load: unexpected key mismatch — missing={missing}, "
                f"unexpected={unexpected}"
            )
        with torch.no_grad():
            executor.tempo_embed.weight.zero_()
            executor.tempo_embed.bias.zero_()
            executor.phrase_pos_embed.weight.zero_()
        logger.info("phrase_pos_embed: zero-initialized (legacy checkpoint)")
        executor.eval()

        legacy_bins_path = _REPO / "data" / "processed" / "duration_bins.v5.legacy.json"
        with open(legacy_bins_path) as f:
            edges = json.load(f)["bin_edges"]
        DUR_OFFSET = 4
        def decode_dur(token, tempo_bpm):
            idx = int(token) - DUR_OFFSET
            if 0 <= idx < len(edges) - 1:
                return math.sqrt(edges[idx] * edges[idx + 1])
            return MIN_NOTE_DUR
        return executor, decode_dur

    logger.info("note_executor checkpoint: current v6 (dur_vocab=%d, tempo_embed=%s)",
                ckpt_dur_vocab, has_tempo)
    executor = NoteExecutor(
        chord_vocab_size=chord_tok.vocab_size,
        artist_vocab_size=artist_tok.vocab_size,
        dur_vocab_size=ckpt_dur_vocab,
        dropout=dropout,
    )
    if "phrase_pos_embed.weight" in state:
        executor.load_state_dict(state, strict=True)
    else:
        result = executor.load_state_dict(state, strict=False)
        missing = set(result.missing_keys)
        unexpected = set(result.unexpected_keys)
        if missing != {"phrase_pos_embed.weight"} or unexpected:
            raise RuntimeError(
                f"compat load: unexpected key mismatch — missing={missing}, "
                f"unexpected={unexpected}"
            )
        with torch.no_grad():
            executor.phrase_pos_embed.weight.zero_()
        logger.info("phrase_pos_embed: zero-initialized (legacy checkpoint)")
    executor.eval()
    def decode_dur(token, tempo_bpm):
        return note_tok.decode_duration(int(token), tempo_bpm=tempo_bpm)
    return executor, decode_dur


def load_models(chord_tok, artist_tok, note_tok, note_executor_checkpoint=None):
    planner = PhrasePlanner(
        chord_vocab_size=chord_tok.vocab_size,
        artist_vocab_size=artist_tok.vocab_size,
    )
    planner.load_state_dict(
        torch.load(CHECKPOINTS / "phrase_planner_best.pt", map_location="cpu")
    )
    planner.eval()

    executor_path = note_executor_checkpoint or CHECKPOINTS / "note_executor_best.pt"
    executor, decode_dur = _load_note_executor_with_compat(
        executor_path, chord_tok, artist_tok, note_tok, dropout=0.2,
    )
    return planner, executor, decode_dur


# ---------------------------------------------------------------------------
# Per-chord generation helpers
# ---------------------------------------------------------------------------

def _plan_phrase(planner, chord_ids, artist_id, fallback=3, temperature=0.8):
    with torch.no_grad():
        tokens = planner.generate(chord_ids, artist_id, max_len=8, temperature=temperature)
    return tokens[0] if tokens else fallback


def _plan_phrases(planner, model_chord_ids, artist_id, n_sections, fallback=3, temperature=0.8):
    """Plan phrase tokens once over the full progression, then pad/repeat."""
    full_chord_ids = torch.tensor([model_chord_ids], dtype=torch.long)
    with torch.no_grad():
        tokens = planner.generate(
            full_chord_ids,
            artist_id,
            max_len=n_sections + 4,
            temperature=temperature,
        )
    tokens = [int(t) for t in (tokens or []) if int(t) >= 3]
    if not tokens:
        tokens = [fallback]
    while len(tokens) < n_sections:
        tokens.extend(tokens)
    return tokens[:n_sections]


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _phrase_shaped_n_notes(feature: PhraseFeature | None) -> int:
    if feature is None:
        return N_NOTES
    return int(_clamp(round(feature.median_num_notes), 8, 28))


def _phrase_shaped_rest_boost(base_rest_boost: float, feature: PhraseFeature | None) -> float:
    if feature is None:
        return base_rest_boost
    # Treat ~20% rest as neutral; scale gently and clamp to avoid destructive output.
    return float(_clamp(base_rest_boost + (feature.median_rest_ratio - 0.20) * 4.0, 0.0, 4.0))


def _section_contour(pitches: list[int]) -> str:
    if len(pitches) < 2:
        return "flat"
    first, last = pitches[0], pitches[-1]
    pitch_range = max(pitches) - min(pitches)
    if pitch_range <= 2 or abs(last - first) <= 2:
        return "flat"
    mid = len(pitches) // 2
    if max(pitches) in pitches[max(0, mid - 1): min(len(pitches), mid + 2)]:
        if pitches[mid] > first and pitches[mid] > last:
            return "arch"
    return "ascending" if last > first else "descending"


def _generated_phrase_metrics(section_events: list, beats: float) -> dict:
    total_dur = sum(float(d) for _p, d, _r in section_events)
    rest_dur = sum(float(d) for _p, d, r in section_events if r)
    sounding = [(int(p), float(d)) for p, d, r in section_events if not r and 0 <= int(p) <= 127]
    pitches = [p for p, _d in sounding]
    return {
        "contour": _section_contour(pitches),
        "num_notes": len(sounding),
        "density": (len(sounding) / float(beats)) if beats else 0.0,
        "rest_ratio": (rest_dur / total_dur) if total_dur else 0.0,
        "pitch_range": (max(pitches) - min(pitches)) if pitches else 0,
        "final_pitch": pitches[-1] if pitches else None,
    }


def _apply_register_shape(section_events: list, contour: str | None) -> list:
    """Gentle octave shaping only; keeps pitch classes/chord-tone bias intact."""
    if contour not in {"ascending", "descending", "arch"}:
        return section_events
    shaped = []
    n = max(1, len(section_events) - 1)
    for i, (pitch, dur, is_rest) in enumerate(section_events):
        if is_rest or not (0 <= int(pitch) <= 127):
            shaped.append((pitch, dur, is_rest))
            continue
        frac = i / n
        target = 0
        if contour == "ascending":
            target = -6 if frac < 0.33 else (6 if frac > 0.66 else 0)
        elif contour == "descending":
            target = 6 if frac < 0.33 else (-6 if frac > 0.66 else 0)
        elif contour == "arch":
            target = 6 if 0.33 <= frac <= 0.66 else -6
        new_pitch = int(pitch)
        if target > 0 and new_pitch + 12 <= PITCH_HI:
            new_pitch += 12
        elif target < 0 and new_pitch - 12 >= PITCH_LO:
            new_pitch -= 12
        shaped.append((new_pitch, dur, is_rest))
    return shaped


def _generate_notes(executor, chord_ids, phrase_id, artist_id,
                    prefix_pitch=None, prefix_dur=None, prefix_rest=None,
                    temperature=0.8, window=8,
                    duration_temperature=1.6, rest_boost=1.8,
                    final_note_chord_tone_boost=3.0,
                    chord_tone_bias=False,
                    chord_tone_bias_strength=2.0,
                    non_chord_penalty=0.0,
                    strong_beat_only=True,
                    is_final_segment=False,
                    active_chord_symbol=None,
                    n_notes=N_NOTES):
    with torch.no_grad():
        raw = executor.generate(
            chord_ids, phrase_id, artist_id, n_notes=n_notes,
            prefix_pitch=prefix_pitch,
            prefix_dur=prefix_dur,
            prefix_rest=prefix_rest,
            temperature=temperature,
            window=window,
            duration_temperature=duration_temperature,
            rest_boost=rest_boost,
            final_note_chord_tone_boost=final_note_chord_tone_boost,
            chord_tone_bias=chord_tone_bias,
            chord_tone_bias_strength=chord_tone_bias_strength,
            non_chord_penalty=non_chord_penalty,
            strong_beat_only=strong_beat_only,
            is_final_segment=is_final_segment,
            active_chord_symbol=active_chord_symbol,
        )
    return raw


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_solo(progression, artist_name="Charlie Parker",
                  chord_tok=None, note_tok=None,
                  phrase_tok=None, artist_tok=None,
                  planner=None, executor=None,
                  window=8, temperature=0.8,
                  decode_dur=None, tempo_bpm=120.0,
                  duration_temperature=1.6, rest_boost=1.8,
                  chord_shuffled=False, chord_zeroed=False,
                  chord_perturb_seed=42,
                  final_cadence_boost=3.0,
                  chord_tone_bias=False,
                  chord_tone_bias_strength=2.0,
                  non_chord_penalty=0.0,
                  strong_beat_only=True,
                  phrase_features: dict[str, PhraseFeature] | None = None,
                  phrase_shaping=False):
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

    original_chord_ids = [
        chord_tok.encode(normalize_chord(chord_str))
        for chord_str, _ in progression
    ]
    model_chord_ids = list(original_chord_ids)
    chord_condition = "normal"
    if chord_shuffled:
        rng = random.Random(chord_perturb_seed)
        rng.shuffle(model_chord_ids)
        chord_condition = "chord_shuffled"
    elif chord_zeroed:
        model_chord_ids = [getattr(chord_tok, "UNK", 0)] * len(model_chord_ids)
        chord_condition = "chord_zeroed"

    phrase_plan_ints = _plan_phrases(
        planner,
        model_chord_ids,
        artist_id,
        len(progression),
        fallback=phrase_tok.encode("PHRASE_00"),
        temperature=temperature,
    )
    phrase_plan_labels = []
    for i, raw_phrase in enumerate(phrase_plan_ints):
        chord_for_remap = chord_tok.decode(model_chord_ids[i])
        phrase_plan_labels.append(phrase_tok.decode(remap_phrase_fallback(raw_phrase, chord_for_remap, phrase_tok)))

    # Rolling cross-chord context: last WINDOW note tokens from previous section
    ctx_pitch: list = []
    ctx_dur:   list = []
    ctx_rest:  list = []

    for section_idx, (chord_str, beats) in enumerate(progression):
        wjazz_str = normalize_chord(chord_str)
        chord_id  = chord_tok.encode(wjazz_str)
        if chord_id == chord_tok.UNK:
            unknown_chords.append(chord_str)

        model_chord_id = model_chord_ids[section_idx]
        model_chord_str = chord_tok.decode(model_chord_id)
        chord_ids = torch.tensor([[model_chord_id]], dtype=torch.long)

        # ---- PhrasePlanner + PHRASE_00 remapping ----
        phrase_int_raw = phrase_plan_ints[section_idx]
        phrase_int     = remap_phrase_fallback(phrase_int_raw, model_chord_str, phrase_tok)
        phrase_remapped = phrase_int != phrase_int_raw
        phrase_label = phrase_tok.decode(phrase_int)
        phrase_id    = torch.tensor([phrase_int], dtype=torch.long)
        feature = phrase_features.get(phrase_label) if phrase_features else None
        section_n_notes = _phrase_shaped_n_notes(feature) if phrase_shaping else N_NOTES
        section_rest_boost = _phrase_shaped_rest_boost(rest_boost, feature) if phrase_shaping else rest_boost

        # ---- NoteExecutor (with cross-chord prefix) ----
        pfx_p = ctx_pitch[-window:] or None
        pfx_d = ctx_dur[-window:]   or None
        pfx_r = ctx_rest[-window:]  or None
        raw_notes = _generate_notes(
            executor, chord_ids, phrase_id, artist_id,
            prefix_pitch=pfx_p, prefix_dur=pfx_d, prefix_rest=pfx_r,
            temperature=temperature,
            window=window,
            duration_temperature=duration_temperature,
            rest_boost=section_rest_boost,
            final_note_chord_tone_boost=final_cadence_boost,
            chord_tone_bias=chord_tone_bias,
            chord_tone_bias_strength=chord_tone_bias_strength,
            non_chord_penalty=non_chord_penalty,
            strong_beat_only=strong_beat_only,
            is_final_segment=section_idx == len(progression) - 1,
            active_chord_symbol=chord_str,
            n_notes=section_n_notes,
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
                _d = decode_dur(d_tok, tempo_bpm) if decode_dur is not None else note_tok.decode_duration(d_tok, tempo_bpm)
            except ValueError:
                _d = 0.25
                dur_fallback += 1
            dur_sec  = max(_d, MIN_NOTE_DUR)
            is_rest  = bool(r_tok)
            section_events.append((pitch_midi, dur_sec, is_rest))
            if 0 <= pitch_midi <= 127:
                pitches_midi.append(pitch_midi)

        # --- Rest injection (within section) ---
        section_events = _inject_rests(section_events)
        if phrase_shaping and feature is not None:
            section_events = _apply_register_shape(section_events, feature.contour)
        generated_metrics = _generated_phrase_metrics(section_events, beats)

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
            "model_chord":     model_chord_str,
            "model_chord_id":  int(model_chord_id),
            "chord_condition": chord_condition,
            "beats":           beats,
            "in_vocab":        chord_id != chord_tok.UNK,
            "phrase_token":    phrase_label,
            "phrase_plan":     phrase_plan_labels,
            "phrase_features": phrase_feature_to_dict(feature),
            "generated_phrase_metrics": generated_metrics,
            "phrase_shaping":  bool(phrase_shaping),
            "section_rest_boost": section_rest_boost,
            "phrase_remapped": phrase_remapped,
            "phrase_original": phrase_tok.decode(phrase_int_raw) if phrase_remapped else None,
            "n_notes":         section_n_notes,
            "n_clamped":       n_clamped,
            "dur_fallback":    dur_fallback,
            "pitches":         pitches_midi,
        })

    return note_events, chord_summaries, unknown_chords


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

def export_midi(note_events, output_path, tempo=120, rhythm_instruments=None):
    """Write note_events to a MIDI file. Returns total duration in seconds.

    rhythm_instruments: optional list of pretty_midi.Instrument objects (piano,
    bass, drums from generate_rhythm_section). When None, behavior matches the
    original solo-only export exactly.
    """
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
    if rhythm_instruments:
        pm.instruments.extend(rhythm_instruments)
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
    sections = []
    for s in summaries:
        section = {
            "chord":          s["chord"],
            "model_chord":    s.get("model_chord", s["chord"]),
            "chord_condition": s.get("chord_condition", "normal"),
            "beats":          s["beats"],
            "phrase":         s["phrase_token"],
            "n_notes":        s["n_notes"],
            "clamped":        s.get("n_clamped", 0),
            "distinct_pitches": len(set(s["pitches"])),
            "dur_fallback":   s.get("dur_fallback", 0),
            "pitches":         s.get("pitches", []),
        }
        for optional_key in (
            "phrase_features",
            "generated_phrase_metrics",
            "phrase_shaping",
            "section_rest_boost",
        ):
            if optional_key in s:
                section[optional_key] = s.get(optional_key)
        section.update({
            "phrase_remapped": s.get("phrase_remapped", False),
            "phrase_original": s.get("phrase_original", None),
        })
        sections.append(section)

    data = {
        "name":        name or output_path.stem,
        "tempo_bpm":   tempo_bpm,
        "total_notes": len(note_events),
        "notes": [
            {"pitch": int(p), "duration_sec": round(float(d), 4), "is_rest": bool(r)}
            for p, d, r in note_events
        ],
        "sections": sections,
    }
    if summaries and "phrase_plan" in summaries[0]:
        # Preserve legacy JSON shape when callers provide old-style summaries.
        data = {
            "name": data["name"],
            "tempo_bpm": data["tempo_bpm"],
            "total_notes": data["total_notes"],
            "phrase_plan": summaries[0].get("phrase_plan", []),
            "notes": data["notes"],
            "sections": data["sections"],
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

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate jazz solos with chord progressions.")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--out-dir", type=Path, default=SOLOS_DIR, dest="out_dir")
    parser.add_argument("--note-executor-checkpoint", type=Path, default=None,
                        help="Override NoteExecutor checkpoint path.")
    parser.add_argument("--chord-shuffled", action="store_true",
                        help="Shuffle chord token IDs before model conditioning.")
    parser.add_argument("--chord-zeroed", action="store_true",
                        help="Replace model chord token IDs with UNK before conditioning.")
    parser.add_argument("--with-rhythm-section", action="store_true",
                        help="Include rule-based piano/bass/drums tracks.")
    parser.add_argument("--rhythm-style", default="swing",
                        choices=["swing", "bossa", "ballad", "latin", "funk"],
                        help="Rhythm-section style (default: swing).")
    parser.add_argument("--rhythm-seed", type=int, default=42,
                        help="Seed for rhythm-section humanization RNG.")
    parser.add_argument("--duration-temperature", type=float, default=1.6,
                        dest="duration_temperature",
                        help="Temperature for duration sampling (default 1.6).")
    parser.add_argument("--rest-boost", type=float, default=1.8,
                        dest="rest_boost",
                        help="Additive logit boost on is_rest_head's rest class (default 1.8).")
    parser.add_argument("--final-cadence-boost", type=float, default=3.0,
                        dest="final_cadence_boost",
                        help="Additive pitch-logit boost for final chord-tone cadence notes (default 3.0).")
    parser.add_argument("--no-final-cadence-boost", action="store_const",
                        const=0.0, dest="final_cadence_boost",
                        help="Disable final cadence chord-tone boost.")
    parser.add_argument("--chord-tone-bias", action="store_true",
                        help="Enable inference-time chord-tone steering for generated pitches.")
    parser.add_argument("--chord-tone-bias-strength", type=float, default=2.0,
                        dest="chord_tone_bias_strength",
                        help="Logit boost added to active chord-tone pitch classes when --chord-tone-bias is enabled.")
    parser.add_argument("--non-chord-penalty", type=float, default=0.0,
                        dest="non_chord_penalty",
                        help="Optional logit penalty subtracted from non-chord-tone pitch classes under --chord-tone-bias.")
    parser.add_argument("--all-beat-chord-tone-bias", action="store_false",
                        dest="strong_beat_only",
                        help="Apply chord-tone bias to every generated position instead of only 0/4/8/12 anchors.")
    parser.add_argument("--phrase-shaping", action="store_true",
                        help="Use phrase-cluster feature medians to shape note count/rests/register.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.chord_shuffled and args.chord_zeroed:
        raise SystemExit("--chord-shuffled and --chord-zeroed are mutually exclusive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    chord_tok  = ChordTokenizer.from_json()
    note_tok   = NoteTokenizer.from_json()
    phrase_tok = PhraseTokenizer()
    artist_tok = ArtistTokenizer()

    print("Loading models …")
    planner, executor, decode_dur = load_models(
        chord_tok, artist_tok, note_tok,
        note_executor_checkpoint=args.note_executor_checkpoint,
    )
    print("Models loaded.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    # Rhythm-section tempo is derived from the solo's own internal grid so
    # the two streams cannot desync. BEAT_DURATION is seconds-per-beat.
    tempo_bpm = 60.0 / BEAT_DURATION
    phrase_features = load_phrase_features()


    for name, progression in PROGRESSIONS.items():
        note_events, summaries, unknowns = generate_solo(
            progression,
            chord_tok=chord_tok, note_tok=note_tok,
            phrase_tok=phrase_tok, artist_tok=artist_tok,
            planner=planner, executor=executor,
            window=args.window, temperature=args.temperature,
            decode_dur=decode_dur, tempo_bpm=tempo_bpm,
            duration_temperature=args.duration_temperature,
            rest_boost=args.rest_boost,
            chord_shuffled=args.chord_shuffled,
            chord_zeroed=args.chord_zeroed,
            chord_perturb_seed=args.rhythm_seed,
            final_cadence_boost=args.final_cadence_boost,
            chord_tone_bias=args.chord_tone_bias,
            chord_tone_bias_strength=args.chord_tone_bias_strength,
            non_chord_penalty=args.non_chord_penalty,
            strong_beat_only=args.strong_beat_only,
            phrase_features=phrase_features,
            phrase_shaping=args.phrase_shaping,
        )

        if args.with_rhythm_section:
            stem = f"{name}_rhythm_{args.rhythm_style}"
        elif args.window != 8 or args.temperature != 0.8:
            stem = f"{name}__w{args.window}_t{args.temperature:.1f}"
        else:
            stem = name

        if args.duration_temperature != 1.6 or args.rest_boost != 1.8:
            stem += f"_dt{args.duration_temperature:.1f}_rb{args.rest_boost:.1f}"
        if args.chord_shuffled:
            stem += "_chord_shuffled"
        elif args.chord_zeroed:
            stem += "_chord_zeroed"

        midi_out = args.out_dir / f"{stem}.mid"
        json_out = args.out_dir / f"{stem}.json"

        rhythm = None
        if args.with_rhythm_section:
            rhythm = generate_rhythm_section(
                progression, tempo_bpm=tempo_bpm,
                style=args.rhythm_style, seed=args.rhythm_seed,
            )

        total_dur = export_midi(note_events, midi_out, rhythm_instruments=rhythm)
        export_json(note_events, summaries, json_out, name=name)
        report(name, summaries, unknowns, midi_out, total_dur, note_events)

    print(f"\nFiles saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
