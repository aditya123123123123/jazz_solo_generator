import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PHRASES_PATH = Path(__file__).parents[2] / "data" / "processed" / "phrases_all.json"
CSV_PATH = Path(__file__).parents[2] / "data" / "processed" / "phrase_features_all.csv"
OUTPUTS_DIR = Path(__file__).parents[2] / "outputs"
BEAT_DURATION = 0.5  # seconds per beat, assuming 120 BPM


def compute_phrase_features(phrase):
    notes = sorted(phrase["notes"], key=lambda x: x["onset"])
    n = len(notes)

    if n == 0:
        return {
            "solo_id": phrase["solo_id"],
            "performer": phrase["performer"],
            "phrase_number": phrase["phrase_number"],
            "num_notes": 0,
            "phrase_duration": 0.0,
            "length_beats": 0.0,
            "note_density": 0.0,
            "pitch_mean": 0.0,
            "pitch_range": 0,
            "pitch_std": 0.0,
            "rest_ratio": 0.0,
            "contour": "flat",
            "starts_on_downbeat": False,
        }

    pitches = [note["pitch"] for note in notes]
    onsets = [note["onset"] for note in notes]
    durations = [note["duration"] for note in notes]

    phrase_duration = onsets[-1] + durations[-1] - onsets[0]
    length_beats = phrase_duration / BEAT_DURATION
    note_density = n / length_beats if length_beats > 0 else 0.0

    pitch_mean = sum(pitches) / n
    pitch_range = max(pitches) - min(pitches) if n > 1 else 0
    pitch_std = (
        math.sqrt(sum((p - pitch_mean) ** 2 for p in pitches) / n) if n > 1 else 0.0
    )

    total_silence = sum(
        max(0.0, onsets[i + 1] - (onsets[i] + durations[i])) for i in range(n - 1)
    )
    rest_ratio = total_silence / phrase_duration if phrase_duration > 0 else 0.0

    if n <= 1:
        contour = "flat"
    else:
        first_pitch, last_pitch = pitches[0], pitches[-1]
        if last_pitch > first_pitch + 3:
            contour = "ascending"
        elif last_pitch < first_pitch - 3:
            contour = "descending"
        else:
            peak_idx = pitches.index(max(pitches))
            third = n / 3
            contour = "arch" if third <= peak_idx < 2 * third else "flat"

    starts_on_downbeat = notes[0]["beat"] == 1

    return {
        "solo_id": phrase["solo_id"],
        "performer": phrase["performer"],
        "phrase_number": phrase["phrase_number"],
        "num_notes": n,
        "phrase_duration": phrase_duration,
        "length_beats": length_beats,
        "note_density": note_density,
        "pitch_mean": pitch_mean,
        "pitch_range": pitch_range,
        "pitch_std": pitch_std,
        "rest_ratio": rest_ratio,
        "contour": contour,
        "starts_on_downbeat": starts_on_downbeat,
    }


def compute_all_features(phrases):
    return [compute_phrase_features(p) for p in phrases]


def print_summary_stats(df):
    numeric_cols = [
        "length_beats", "num_notes", "note_density", "pitch_mean",
        "pitch_range", "pitch_std", "rest_ratio", "phrase_duration",
    ]
    summary = df.groupby("performer")[numeric_cols].agg(["mean", "std"])
    print(summary.to_string())


def save_plots(df):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    artists = ["Charlie Parker", "Miles Davis"]
    colors = ["steelblue", "coral"]

    for feature, xlabel, filename in [
        ("length_beats", "Phrase Length (beats)", "phrase_length_distribution.png"),
        ("note_density", "Note Density (notes/beat)", "note_density_distribution.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for artist, color in zip(artists, colors):
            data = df.loc[df["performer"] == artist, feature]
            ax.hist(data, bins=30, alpha=0.6, label=artist, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(xlabel + " Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / filename, dpi=150)
        plt.close(fig)
        print(f"  {filename}")


if __name__ == "__main__":
    with open(PHRASES_PATH) as f:
        phrases = json.load(f)

    rows = compute_all_features(phrases)
    df = pd.DataFrame(rows, columns=[
        "solo_id", "performer", "phrase_number", "length_beats", "num_notes",
        "note_density", "pitch_mean", "pitch_range", "pitch_std", "rest_ratio",
        "contour", "starts_on_downbeat", "phrase_duration",
    ])

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} rows to {CSV_PATH}")

    print("\nSummary stats per performer:")
    print_summary_stats(df)

    print("\nPlots saved to outputs/:")
    save_plots(df)
