import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from src.data.note_dataset import NoteWindowDataset, collate_note_window
from src.models.note_executor import NoteExecutor
from src.tokenization import ArtistTokenizer, ChordTokenizer


CONDITIONS = [
    "baseline",
    "phrase_zeroed",
    "chord_zeroed",
    "artist_zeroed",
    "chord_shuffled",
    "phrase_shuffled",
]

EXPECTED_V6_PITCH_CE = {
    "baseline":        (4.9997, 5.0040),
    "phrase_zeroed":   (4.9658, 4.9740),
    "chord_zeroed":    (4.9335, 4.9420),
    "artist_zeroed":   (4.9981, 5.0061),
    "chord_shuffled":  (4.9974, 5.0054),
    "phrase_shuffled": (4.9982, 5.0062),
}


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_with_tempo(batch):
    out = collate_note_window(batch)
    out["tempo_bpm"] = torch.stack([b["tempo_bpm"] for b in batch])
    return out


def _load_model(checkpoint_path, device):
    chord_tok = ChordTokenizer.from_json()
    artist_tok = ArtistTokenizer()
    model = NoteExecutor(
        chord_vocab_size=chord_tok.vocab_size,
        artist_vocab_size=artist_tok.vocab_size,
        dropout=0.1,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _mutate_batch(batch, condition, generator):
    mutated = {k: v.clone() for k, v in batch.items()}
    if condition == "baseline":
        return mutated
    if condition == "phrase_zeroed":
        mutated["phrase_id"].zero_()
    elif condition == "chord_zeroed":
        mutated["chord_ids"].zero_()
    elif condition == "artist_zeroed":
        mutated["artist_id"].zero_()
    elif condition == "chord_shuffled":
        perm = torch.randperm(mutated["chord_ids"].size(0), generator=generator)
        mutated["chord_ids"] = mutated["chord_ids"][perm]
        mutated["chord_mask"] = mutated["chord_mask"][perm]
    elif condition == "phrase_shuffled":
        perm = torch.randperm(mutated["phrase_id"].size(0), generator=generator)
        mutated["phrase_id"] = mutated["phrase_id"][perm]
    else:
        raise ValueError(f"unknown condition: {condition}")
    return mutated


@torch.no_grad()
def evaluate_condition(model, loader, condition, device, seed):
    ce = nn.CrossEntropyLoss(label_smoothing=0.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sums = {"pitch_CE": 0.0, "dur_CE": 0.0, "rest_CE": 0.0, "total": 0.0}
    n_batches = 0

    for batch in loader:
        batch = _mutate_batch(batch, condition, generator)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")
        ):
            p_logits, d_logits, r_logits = model(
                batch["chord_ids"],
                batch["phrase_id"],
                batch["artist_id"],
                batch["ctx_pitch"],
                batch["ctx_dur"],
                batch["ctx_rest"],
                batch["pos_in_phrase"],
                tempo_bpm=batch["tempo_bpm"],
                src_key_padding_mask=~batch["chord_mask"],
            )
            pitch_l = ce(p_logits, batch["target_pitch"])
            dur_l = ce(d_logits, batch["target_dur"])
            rest_l = ce(r_logits, batch["target_rest"])
            total = pitch_l + dur_l + rest_l

        sums["pitch_CE"] += pitch_l.item()
        sums["dur_CE"] += dur_l.item()
        sums["rest_CE"] += rest_l.item()
        sums["total"] += total.item()
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


def _format_markdown(rows):
    lines = [
        "| condition | pitch_CE | dur_CE | rest_CE | total | delta_pitch_vs_baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["delta_pitch_vs_baseline"]
        delta_s = "--" if delta is None else f"{delta:+.4f}"
        lines.append(
            f"| {row['condition']} | {row['pitch_CE']:.4f} | {row['dur_CE']:.4f} | "
            f"{row['rest_CE']:.4f} | {row['total']:.4f} | {delta_s} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "conditioning_ablation.csv"
    md_path = out_dir / "conditioning_ablation.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "pitch_CE",
                "dur_CE",
                "rest_CE",
                "total",
                "delta_pitch_vs_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path.write_text(_format_markdown(rows))
    return csv_path, md_path


def _check_v6_baseline(rows):
    failures = []
    for row in rows:
        condition = row["condition"]
        if condition not in EXPECTED_V6_PITCH_CE:
            continue
        lo, hi = EXPECTED_V6_PITCH_CE[condition]
        value = row["pitch_CE"]
        if not (lo < value < hi):
            failures.append(f"{condition}: pitch_CE={value:.4f}, expected {lo:.4f} < x < {hi:.4f}")
    if failures:
        joined = "\n  ".join(failures)
        raise AssertionError(f"v6 baseline ablation reproduction failed:\n  {joined}")


def _build_val_loader(args, device):
    print(f"Loading cache: {args.cache}")
    full_ds = NoteWindowDataset.from_cache(args.cache)
    n_train = int(0.8 * len(full_ds))
    n_val = len(full_ds) - n_train
    _, val_ds = random_split(full_ds, [n_train, n_val])
    val_subset = Subset(val_ds, range(min(args.subset_size, len(val_ds))))
    return DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_with_tempo,
    )


def run_ablation(checkpoint_path, loader, device, seed):
    print(f"Loading checkpoint: {checkpoint_path}")
    model = _load_model(checkpoint_path, device)
    rows = []
    baseline_pitch = None
    for i, condition in enumerate(CONDITIONS):
        metrics = evaluate_condition(model, loader, condition, device, seed + i)
        if condition == "baseline":
            baseline_pitch = metrics["pitch_CE"]
            delta = None
        else:
            delta = metrics["pitch_CE"] - baseline_pitch
        row = {
            "condition": condition,
            "pitch_CE": metrics["pitch_CE"],
            "dur_CE": metrics["dur_CE"],
            "rest_CE": metrics["rest_CE"],
            "total": metrics["total"],
            "delta_pitch_vs_baseline": delta,
        }
        rows.append(row)
        delta_s = "--" if delta is None else f"{delta:+.4f}"
        print(
            f"{condition:15s} pitch={row['pitch_CE']:.4f} dur={row['dur_CE']:.4f} "
            f"rest={row['rest_CE']:.4f} total={row['total']:.4f} delta={delta_s}"
        )
    return rows


def _build_comparison_rows(candidate_rows, baseline_rows):
    candidate_by_cond = {row["condition"]: row for row in candidate_rows}
    baseline_by_cond = {row["condition"]: row for row in baseline_rows}
    candidate_baseline_pitch = candidate_by_cond["baseline"]["pitch_CE"]
    rows = []
    for condition in CONDITIONS:
        candidate_pitch = candidate_by_cond[condition]["pitch_CE"]
        baseline_pitch = baseline_by_cond[condition]["pitch_CE"]
        delta_vs_candidate_baseline = None
        if condition != "baseline":
            delta_vs_candidate_baseline = candidate_pitch - candidate_baseline_pitch
        rows.append(
            {
                "condition": condition,
                "candidate_pitch_ce": candidate_pitch,
                "baseline_pitch_ce": baseline_pitch,
                "delta_candidate_minus_baseline": candidate_pitch - baseline_pitch,
                "delta_vs_candidate_baseline": delta_vs_candidate_baseline,
            }
        )
    return rows


def _format_comparison_markdown(rows):
    lines = [
        "| condition | candidate_pitch_ce | baseline_pitch_ce | "
        "delta_candidate_minus_baseline | delta_vs_candidate_baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta_vs = row["delta_vs_candidate_baseline"]
        delta_vs_s = "--" if delta_vs is None else f"{delta_vs:+.4f}"
        lines.append(
            f"| {row['condition']} | {row['candidate_pitch_ce']:.4f} | "
            f"{row['baseline_pitch_ce']:.4f} | "
            f"{row['delta_candidate_minus_baseline']:+.4f} | {delta_vs_s} |"
        )
    return "\n".join(lines) + "\n"


def _write_comparison_outputs(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "conditioning_ablation_comparison.csv"
    md_path = out_dir / "conditioning_ablation_comparison.md"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "candidate_pitch_ce",
                "baseline_pitch_ce",
                "delta_candidate_minus_baseline",
                "delta_vs_candidate_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(_format_comparison_markdown(rows))
    return csv_path, md_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run NoteExecutor conditioning ablations.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help="Optional reference checkpoint for side-by-side comparison output.",
    )
    parser.add_argument("--cache", type=Path, default=_REPO / "data" / "processed" / "notes_v6_cache.pt")
    parser.add_argument("--subset-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--check-v6-baseline", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    device = _select_device()
    print(f"Device: {device}")

    # Match the 2026-05-18 ablation ordering: seed, instantiate model, load
    # weights, then random_split. The model init consumes RNG before splitting.
    torch.manual_seed(args.seed)
    loader = _build_val_loader(args, device)

    candidate_rows = run_ablation(args.checkpoint, loader, device, args.seed)

    if args.check_v6_baseline:
        _check_v6_baseline(candidate_rows)
        print("v6 baseline ablation reproduction passed.")

    csv_path, md_path = _write_outputs(candidate_rows, args.out_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    if args.baseline_checkpoint is not None:
        baseline_rows = run_ablation(args.baseline_checkpoint, loader, device, args.seed)
        comparison_rows = _build_comparison_rows(candidate_rows, baseline_rows)
        cmp_csv, cmp_md = _write_comparison_outputs(comparison_rows, args.out_dir)
        print(f"Wrote {cmp_csv}")
        print(f"Wrote {cmp_md}")


if __name__ == "__main__":
    main()
