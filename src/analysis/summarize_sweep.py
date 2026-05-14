import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


FILENAME_RE = re.compile(r"^(?P<progression>.+)__w(?P<window>\d+)_t(?P<temp>\d+\.\d+)\.json$")
QUALITY_RE = re.compile(r"^[A-G][#b]?(.*)$")

COLS = [
    "progression",
    "window",
    "temp",
    "total_notes",
    "avg_distinct_pitches",
    "total_clamped",
    "total_dur_fallback",
    "avg_pitch_range",
    "avg_ic_entropy",
    "avg_rest_ratio",
    "unique_phrases",
    "phrase_diversity_ratio",
    "chord_conditional_phrase_entropy",
]


def parse_filename(name):
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return m.group("progression"), int(m.group("window")), float(m.group("temp"))


def chord_quality(chord):
    m = QUALITY_RE.match(chord)
    return m.group(1) if m else chord


def conditional_phrase_entropy(sections):
    by_quality = defaultdict(Counter)
    for s in sections:
        by_quality[chord_quality(s["chord"])][s["phrase"]] += 1
    total = sum(sum(c.values()) for c in by_quality.values())
    if total == 0:
        return 0.0
    weighted = 0.0
    for c in by_quality.values():
        n = sum(c.values())
        h = -sum((v / n) * math.log2(v / n) for v in c.values() if v > 0)
        weighted += (n / total) * h
    return weighted


def summarize(path):
    parsed = parse_filename(path.name)
    if parsed is None:
        return None
    progression, window, temp = parsed
    data = json.loads(path.read_text())
    sections = data["sections"]
    n = len(sections)
    if n == 0:
        return None
    phrases = [s["phrase"] for s in sections]
    unique_phrases = len(set(phrases))
    return {
        "progression": progression,
        "window": window,
        "temp": temp,
        "total_notes": data["total_notes"],
        "avg_distinct_pitches": round(sum(s["distinct_pitches"] for s in sections) / n, 2),
        "total_clamped": sum(s["clamped"] for s in sections),
        "total_dur_fallback": sum(s["dur_fallback"] for s in sections),
        "avg_pitch_range": round(sum(s["pitch_range"] for s in sections) / n, 2),
        "avg_ic_entropy": round(sum(s["interval_class_entropy"] for s in sections) / n, 4),
        "avg_rest_ratio": round(sum(s["rest_ratio"] for s in sections) / n, 4),
        "unique_phrases": unique_phrases,
        "phrase_diversity_ratio": round(unique_phrases / n, 4),
        "chord_conditional_phrase_entropy": round(conditional_phrase_entropy(sections), 4),
    }


def warnings_for(row):
    out = ""
    if row["total_notes"] > 0 and row["total_clamped"] > 0.10 * row["total_notes"]:
        out += " ⚠CLAMP"
    if row["phrase_diversity_ratio"] < 0.5:
        out += " ⚠COLLAPSE"
    return out


def print_table(rows):
    header = "| " + " | ".join(COLS) + " | warnings |"
    sep = "|" + "|".join(["---"] * (len(COLS) + 1)) + "|"
    print(header)
    print(sep)
    for r in rows:
        cells = [str(r[c]) for c in COLS]
        print("| " + " | ".join(cells) + " |" + warnings_for(r) + " |")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r[c] for c in COLS})


def main():
    parser = argparse.ArgumentParser(description="Summarize sweep results.")
    parser.add_argument("--solos-dir", default="outputs/solos_sweep")
    args = parser.parse_args()

    solos_dir = Path(args.solos_dir)
    rows = []
    for p in sorted(solos_dir.glob("*.json")):
        row = summarize(p)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r["progression"], r["window"], r["temp"]))

    print_table(rows)
    csv_path = solos_dir / "summary.csv"
    write_csv(rows, csv_path)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
