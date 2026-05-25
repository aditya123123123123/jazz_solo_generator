"""Phrase-cluster feature loading for phrase-aware generation.

Aggregates ``data/processed/phrase_clusters_all.csv`` by ``phrase_token`` into
interpretable median/modal statistics usable at inference and evaluation time.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.generation.chord_utils import chord_tones, parse_chord

_REPO = Path(__file__).parents[2]
DEFAULT_PHRASE_FEATURES_PATH = _REPO / "data" / "processed" / "phrase_clusters_all.csv"


@dataclass(frozen=True)
class PhraseFeature:
    token: str
    median_length_beats: float
    median_num_notes: int
    median_density: float
    median_pitch_mean: float
    median_pitch_range: float
    median_rest_ratio: float
    contour: str
    starts_on_downbeat: bool
    final_chord_tone_rate: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _mode(values: list[Any]) -> Any:
    counts = Counter(values)
    # Counter preserves first-seen order for ties in most_common().
    return counts.most_common(1)[0][0]


def load_phrase_features(path: Path | str | None = None) -> dict[str, PhraseFeature]:
    """Load aggregate phrase-cluster features keyed by ``PHRASE_XX`` token."""
    csv_path = Path(path) if path is not None else DEFAULT_PHRASE_FEATURES_PATH
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    phrase_key_to_token: dict[tuple[str, str], str] = {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = row.get("phrase_token", "").strip()
            if token:
                groups[token].append(row)
                phrase_key_to_token[(str(row.get("solo_id", "")), str(row.get("phrase_number", "")))] = token

    final_chord_tone_rates = _load_final_chord_tone_rates(csv_path, phrase_key_to_token)

    features: dict[str, PhraseFeature] = {}
    for token in sorted(groups):
        rows = groups[token]

        def med_float(column: str) -> float:
            return float(statistics.median(float(r[column]) for r in rows))

        median_num_notes = int(round(med_float("num_notes")))
        starts_values = [_as_bool(r["starts_on_downbeat"]) for r in rows]
        features[token] = PhraseFeature(
            token=token,
            median_length_beats=med_float("length_beats"),
            median_num_notes=median_num_notes,
            median_density=med_float("note_density"),
            median_pitch_mean=med_float("pitch_mean"),
            median_pitch_range=med_float("pitch_range"),
            median_rest_ratio=med_float("rest_ratio"),
            contour=str(_mode([r["contour"] for r in rows])),
            starts_on_downbeat=bool(_mode(starts_values)),
            final_chord_tone_rate=final_chord_tone_rates.get(token, 0.0),
        )
    return features


def _load_final_chord_tone_rates(csv_path: Path, phrase_key_to_token: dict[tuple[str, str], str]) -> dict[str, float]:
    """Estimate how often each phrase cluster ends on its active chord.

    The cluster CSV has no cadence column, but the sibling phrase JSON contains
    per-note chords. For each training phrase we carry the latest explicit chord
    symbol forward through blank chord cells, then score whether the final note
    pitch class belongs to that active chord's shell tones. The aggregate rate is
    a deterministic tie-breaker for phrase-diversity replacements.
    """
    json_path = csv_path.with_name("phrases_all.json")
    if not json_path.exists():
        return {}

    counts: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    with json_path.open() as f:
        phrases = json.load(f)

    for phrase in phrases:
        token = phrase_key_to_token.get((str(phrase.get("solo_id", "")), str(phrase.get("phrase_number", ""))))
        if not token:
            continue
        notes = phrase.get("notes") or []
        active_chord = None
        final_pitch = None
        for note in notes:
            chord = str(note.get("chord") or "").strip()
            if chord:
                active_chord = chord
            if note.get("pitch") is not None:
                final_pitch = int(note["pitch"])
        if active_chord is None or final_pitch is None:
            continue
        parsed = parse_chord(active_chord)
        if parsed is None:
            continue
        counts[token] += 1
        if final_pitch % 12 in set(chord_tones(*parsed)):
            hits[token] += 1
    return {token: hits[token] / count for token, count in counts.items() if count}


def phrase_feature_to_dict(feature: PhraseFeature | None) -> dict[str, Any] | None:
    if feature is None:
        return None
    return feature.to_json_dict()
