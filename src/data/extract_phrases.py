import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.load_wjazzd import load_db

ARTISTS = ["Charlie Parker", "Miles Davis"]
OUTPUT_PATH = Path(__file__).parents[2] / "data" / "processed" / "phrases.json"

TARGET_PERFORMERS = [
    "John Coltrane",
    "Miles Davis",
    "Charlie Parker",
    "Sonny Rollins",
    "David Liebman",
    "Wayne Shorter",
    "Steve Coleman",
    "Michael Brecker",
    "Clifford Brown",
    "Woody Shaw",
    "Paul Desmond",
    "Louis Armstrong",
    "Lee Konitz",
    "Joe Lovano",
    "Joe Henderson",
    "J.J. Johnson",
    "Don Byas",
    "Chet Baker",
    "Wynton Marsalis",
    "Lester Young",
    "Kenny Dorham",
    "Chris Potter",
    "Bob Berg",
    "Benny Goodman",
    "Benny Carter",
    "Zoot Sims",
    "Steve Lacy",
    "Stan Getz",
    "Sonny Stitt",
    "Roy Eldridge",
    "Phil Woods",
    "Milt Jackson",
    "Lionel Hampton",
    "Johnny Dodds",
    "Gerry Mulligan",
    "Freddie Hubbard",
    "Fats Navarro",
    "Eric Dolphy",
    "Don Ellis",
    "Dizzy Gillespie",
    "Dickie Wells",
    "Dexter Gordon",
    "David Murray",
    "Coleman Hawkins",
    "Branford Marsalis",
    "Art Pepper",
    "Sidney Bechet",
    "Pepper Adams",
    "Ornette Coleman",
    "Kid Ory",
    "Joshua Redman",
    "Herbie Hancock",
    "Cannonball Adderley",
    "Bix Beiderbecke",
    "Ben Webster",
]
EXPANDED_OUTPUT_PATH = Path(__file__).parents[2] / "data" / "processed" / "phrases_expanded.json"
ALL_OUTPUT_PATH = Path(__file__).parents[2] / "data" / "processed" / "phrases_all.json"


def assign_relative_idx(melody):
    melody = melody.sort_values(["melid", "eventid"]).copy()
    melody["rel_idx"] = melody.groupby("melid").cumcount()
    return melody


def extract_phrases(melody, sections, solo_info, beats, performers=None):
    if performers is None:
        performers = ARTISTS
    target_melids = set(solo_info.loc[solo_info["performer"].isin(performers), "melid"])

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
    phrases = extract_phrases(melody, sections, solo_info, beats, performers=TARGET_PERFORMERS)

    ALL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALL_OUTPUT_PATH, "w") as f:
        json.dump(phrases, f, indent=2)

    phrase_counts = Counter(p["performer"] for p in phrases)
    solos_per_performer = {}
    for p in phrases:
        solos_per_performer.setdefault(p["performer"], set()).add(p["solo_id"])

    total_solos = sum(len(v) for v in solos_per_performer.values())

    print(f"{'Performer':<25} {'Phrases':>7}  {'Solos':>5}")
    print("-" * 42)
    for performer in TARGET_PERFORMERS:
        n_phrases = phrase_counts.get(performer, 0)
        n_solos = len(solos_per_performer.get(performer, set()))
        print(f"{performer:<25} {n_phrases:>7}  {n_solos:>5}")
    print("-" * 42)
    print(f"{'Total':<25} {len(phrases):>7}  {total_solos:>5}")
    print(f"\nPerformers included: {len(TARGET_PERFORMERS)}")
    print(f"Saved to {ALL_OUTPUT_PATH}")
