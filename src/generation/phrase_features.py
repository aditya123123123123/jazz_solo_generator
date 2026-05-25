"""Phrase-cluster feature loading for phrase-aware generation.

Aggregates ``data/processed/phrase_clusters_all.csv`` by ``phrase_token`` into
interpretable median/modal statistics usable at inference and evaluation time.
"""
from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = row.get("phrase_token", "").strip()
            if token:
                groups[token].append(row)

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
        )
    return features


def phrase_feature_to_dict(feature: PhraseFeature | None) -> dict[str, Any] | None:
    if feature is None:
        return None
    return feature.to_json_dict()
