import argparse
import json
import math
from pathlib import Path
from statistics import pstdev


def fold_interval_classes(pitches):
    classes = []
    for i in range(1, len(pitches)):
        delta = abs(pitches[i] - pitches[i - 1])
        m = delta % 12
        classes.append(min(m, 12 - m))
    return classes


def ic_entropy(pitches):
    classes = fold_interval_classes(pitches)
    if not classes:
        return 0.0
    counts = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    total = len(classes)
    return -sum((v / total) * math.log2(v / total) for v in counts.values())


def enrich_section(section, notes_slice):
    n_total = len(notes_slice)
    pitches = [n["pitch"] for n in notes_slice if not n["is_rest"]]
    durations = [n["duration_sec"] for n in notes_slice]

    if pitches:
        p_mean = sum(pitches) / len(pitches)
        p_std = pstdev(pitches) if len(pitches) > 1 else 0.0
        p_range = max(pitches) - min(pitches)
    else:
        p_mean = 0.0
        p_std = 0.0
        p_range = 0

    section["pitch_mean"] = round(p_mean, 2)
    section["pitch_std"] = round(p_std, 2)
    section["pitch_range"] = p_range
    section["interval_class_entropy"] = round(ic_entropy(pitches), 4)
    section["rest_ratio"] = round(
        sum(1 for n in notes_slice if n["is_rest"]) / n_total, 4
    ) if n_total else 0.0
    section["avg_duration_sec"] = round(
        sum(durations) / n_total, 4
    ) if n_total else 0.0
    return section


def enrich_file(path):
    data = json.loads(path.read_text())
    notes = data["notes"]
    sections = data["sections"]
    idx = 0
    for sec in sections:
        n = sec["n_notes"]
        enrich_section(sec, notes[idx : idx + n])
        idx += n
    path.write_text(json.dumps(data, indent=2))
    return sections


def print_table(name, sections):
    print(f"\n### {name}\n")
    print("| chord | phrase | n_notes | pitch_range | ic_entropy | rest_ratio |")
    print("|---|---|---:|---:|---:|---:|")
    for s in sections:
        print(
            f"| {s['chord']:<8} "
            f"| {s['phrase']:<12} "
            f"| {s['n_notes']:>7} "
            f"| {s['pitch_range']:>11} "
            f"| {s['interval_class_entropy']:>10.4f} "
            f"| {s['rest_ratio']:>10.4f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Enrich solo JSONs with per-section analytics.")
    parser.add_argument("--solos-dir", default="outputs/solos")
    args = parser.parse_args()

    solos_dir = Path(args.solos_dir)
    for path in sorted(solos_dir.glob("*.json")):
        sections = enrich_file(path)
        print_table(path.stem, sections)
    print()


if __name__ == "__main__":
    main()
