"""One-shot backfill: join WJazzD `solo_info.avgtempo` onto each phrase
in `data/processed/phrases_all.json` and write
`data/processed/phrases_all_with_tempo.json`.

The phrase-extraction step dropped tempo. The note-duration tokenizer needs
beat-relative durations (`duration_sec * tempo / 60`) to avoid the geometric-
binning collapse measured on the seconds-based vocabulary. This script
restores tempo at the phrase level so the downstream tokenizer + dataset
loader can compute beats per note at training time.

Run once:
    python3 scripts/backfill_tempo.py
"""
import json
import sqlite3
import statistics
from pathlib import Path

_REPO  = Path(__file__).resolve().parents[1]
DB     = _REPO / "data" / "raw" / "wjazzd" / "WJazzD.sqlite"
SRC    = _REPO / "data" / "processed" / "phrases_all.json"
DST    = _REPO / "data" / "processed" / "phrases_all_with_tempo.json"


def load_tempo_by_melid(db_path: Path) -> dict[int, float]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT melid, avgtempo FROM solo_info").fetchall()
    finally:
        conn.close()
    return {melid: tempo for melid, tempo in rows}


def main():
    if not DB.exists():
        raise FileNotFoundError(f"WJazzD DB not found: {DB}")
    if not SRC.exists():
        raise FileNotFoundError(f"phrases_all.json not found: {SRC}")

    tempo_by_melid = load_tempo_by_melid(DB)
    print(f"Loaded tempos for {len(tempo_by_melid)} solos from {DB.name}")

    with open(SRC) as f:
        phrases = json.load(f)
    print(f"Loaded {len(phrases):,} phrases from {SRC.name}")

    missing_solo_ids = set()
    bad_tempo_solo_ids = set()
    out = []
    for p in phrases:
        sid = p["solo_id"]
        tempo = tempo_by_melid.get(sid)
        if tempo is None:
            missing_solo_ids.add(sid)
            continue
        if tempo <= 0:
            bad_tempo_solo_ids.add(sid)
            continue
        # Insert tempo_bpm before "notes" so the field order reads naturally.
        new = {k: v for k, v in p.items() if k != "notes"}
        new["tempo_bpm"] = float(tempo)
        new["notes"] = p["notes"]
        out.append(new)

    if missing_solo_ids:
        raise RuntimeError(
            f"{len(missing_solo_ids)} solo_id(s) missing from solo_info: "
            f"{sorted(missing_solo_ids)[:10]}{'...' if len(missing_solo_ids) > 10 else ''}"
        )
    if bad_tempo_solo_ids:
        raise RuntimeError(
            f"{len(bad_tempo_solo_ids)} solo_id(s) have non-positive avgtempo: "
            f"{sorted(bad_tempo_solo_ids)[:10]}{'...' if len(bad_tempo_solo_ids) > 10 else ''}"
        )

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w") as f:
        json.dump(out, f)

    tempos      = [p["tempo_bpm"] for p in out]
    unique_solos = {p["solo_id"] for p in out}
    print()
    print(f"phrases read         : {len(phrases):,}")
    print(f"phrases with tempo   : {len(out):,}  ({len(unique_solos)} unique solos)")
    print(f"tempo min / median / max : "
          f"{min(tempos):.1f} / {statistics.median(tempos):.1f} / {max(tempos):.1f} BPM")
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
