import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.load_wjazzd import load_db

ARTISTS = ["Charlie Parker", "Miles Davis"]
OUTPUT_PATH = Path(__file__).parents[2] / "data" / "processed" / "phrases.json"


def assign_relative_idx(melody):
    melody = melody.sort_values(["melid", "eventid"]).copy()
    melody["rel_idx"] = melody.groupby("melid").cumcount()
    return melody


def extract_phrases(melody, sections, solo_info, beats):
    target_melids = set(solo_info.loc[solo_info["performer"].isin(ARTISTS), "melid"])

    melody = melody[melody["melid"].isin(target_melids)].copy()
    sections = sections[sections["melid"].isin(target_melids)].copy()

    melody = assign_relative_idx(melody)

    beats_clean = beats.drop_duplicates(subset=["melid", "bar", "beat"])
    melody = melody.merge(
        beats_clean[["melid", "bar", "beat", "chord"]],
        on=["melid", "bar", "beat"],
        how="left",
    )

    melody = melody.merge(
        solo_info[["melid", "performer", "title"]],
        on="melid",
        how="left",
    )

    phrases = []
    for melid, mel_group in melody.groupby("melid"):
        secs = sections[sections["melid"] == melid]
        for _, sec in secs.iterrows():
            start, end, phrase_num = int(sec["start"]), int(sec["end"]), int(sec["value"])
            mask = (mel_group["rel_idx"] >= start) & (mel_group["rel_idx"] <= end)
            notes_df = mel_group[mask]
            if notes_df.empty:
                continue

            first = notes_df.iloc[0]
            notes = [
                {
                    "eventid": int(r["eventid"]),
                    "pitch": int(r["pitch"]),
                    "duration": float(r["duration"]),
                    "onset": float(r["onset"]),
                    "bar": int(r["bar"]),
                    "beat": int(r["beat"]),
                    "chord": r["chord"] if pd.notna(r["chord"]) else None,
                }
                for _, r in notes_df.iterrows()
            ]

            phrases.append(
                {
                    "solo_id": int(melid),
                    "performer": first["performer"],
                    "title": first["title"],
                    "phrase_number": phrase_num,
                    "notes": notes,
                }
            )

    return phrases


if __name__ == "__main__":
    melody, sections, solo_info, beats = load_db()
    phrases = extract_phrases(melody, sections, solo_info, beats)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(phrases, f, indent=2)

    counts = Counter(p["performer"] for p in phrases)
    for artist in ARTISTS:
        print(f"{artist}: {counts.get(artist, 0)} phrases")
    print(f"Total: {len(phrases)} phrases saved to {OUTPUT_PATH}")
