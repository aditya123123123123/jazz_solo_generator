import pytest

from src.eval.conditioning_ablation import (
    CONDITIONS,
    EXPECTED_V6_PITCH_CE,
    _build_comparison_rows,
    _check_v6_baseline,
)

# Mac-mini v6.1.0 vs v6 comparison run (2026-05-20), full-precision pitch CE.
VERIFIED_CANDIDATE_PITCH = {
    "baseline": 2.0211238167723833,
    "phrase_zeroed": 2.1957900536303616,
    "chord_zeroed": 2.1855179691801268,
    "artist_zeroed": 2.041961451574248,
    "chord_shuffled": 2.189636052871237,
    "phrase_shuffled": 2.1479893059146646,
}

VERIFIED_BASELINE_PITCH = {
    "baseline": 5.000074639612315,
    "phrase_zeroed": 4.9680421838955,
    "chord_zeroed": 4.938775111217888,
    "artist_zeroed": 5.000241128765807,
    "chord_shuffled": 5.000536176623131,
    "phrase_shuffled": 4.999380542307484,
}

VERIFIED_DELTA_VS_CANDIDATE_BASELINE = {
    "phrase_zeroed": 0.1747,
    "chord_zeroed": 0.1644,
    "artist_zeroed": 0.0208,
    "chord_shuffled": 0.1685,
    "phrase_shuffled": 0.1269,
}


def _rows_from_pitch_map(pitch_by_condition):
    rows = []
    baseline_pitch = pitch_by_condition["baseline"]
    for condition in CONDITIONS:
        pitch = pitch_by_condition[condition]
        delta = None if condition == "baseline" else pitch - baseline_pitch
        rows.append({"condition": condition, "pitch_CE": pitch})
    return rows


def test_check_v6_baseline_accepts_reference_midpoints():
    rows = [
        {"condition": condition, "pitch_CE": (lo + hi) / 2}
        for condition, (lo, hi) in EXPECTED_V6_PITCH_CE.items()
    ]
    _check_v6_baseline(rows)


def test_build_comparison_rows_match_verified_v61_table():
    candidate_rows = _rows_from_pitch_map(VERIFIED_CANDIDATE_PITCH)
    baseline_rows = _rows_from_pitch_map(VERIFIED_BASELINE_PITCH)
    comparison = _build_comparison_rows(candidate_rows, baseline_rows)

    assert len(comparison) == len(CONDITIONS)
    for row in comparison:
        condition = row["condition"]
        assert row["candidate_pitch_ce"] == pytest.approx(VERIFIED_CANDIDATE_PITCH[condition])
        assert row["baseline_pitch_ce"] == pytest.approx(VERIFIED_BASELINE_PITCH[condition])
        assert row["delta_candidate_minus_baseline"] == pytest.approx(
            VERIFIED_CANDIDATE_PITCH[condition] - VERIFIED_BASELINE_PITCH[condition]
        )
        if condition == "baseline":
            assert row["delta_vs_candidate_baseline"] is None
        else:
            assert row["delta_vs_candidate_baseline"] == pytest.approx(
                VERIFIED_DELTA_VS_CANDIDATE_BASELINE[condition], abs=1e-4
            )

