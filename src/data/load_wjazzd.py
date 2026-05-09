import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parents[2] / "data" / "raw" / "wjazzd" / "WJazzD.sqlite"


def load_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    melody = pd.read_sql_query(
        "SELECT eventid, melid, onset, pitch, duration, bar, beat FROM melody ORDER BY melid, eventid",
        conn,
    )
    sections = pd.read_sql_query(
        "SELECT melid, type, start, end, value FROM sections WHERE type='PHRASE'",
        conn,
    )
    solo_info = pd.read_sql_query(
        "SELECT melid, performer, title, key, avgtempo FROM solo_info",
        conn,
    )
    beats = pd.read_sql_query(
        "SELECT melid, bar, beat, chord FROM beats",
        conn,
    )
    conn.close()
    return melody, sections, solo_info, beats


def filter_by_performer(melody, solo_info, performer):
    melids = solo_info.loc[solo_info["performer"] == performer, "melid"]
    return melody[melody["melid"].isin(melids)]


if __name__ == "__main__":
    melody, sections, solo_info, beats = load_db()
    print(f"melody:             {len(melody):>8,} rows")
    print(f"sections (PHRASE):  {len(sections):>8,} rows")
    print(f"solo_info:          {len(solo_info):>8,} rows, {solo_info['performer'].nunique()} performers")
    print(f"beats:              {len(beats):>8,} rows")
    for name in ["Charlie Parker", "Miles Davis"]:
        n = len(filter_by_performer(melody, solo_info, name))
        print(f"  {name}: {n:,} notes")
