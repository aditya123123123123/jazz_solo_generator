import math
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, random_split

sys.stdout.reconfigure(line_buffering=True)

from src.data.note_dataset import NoteWindowDataset, collate_note_window
from src.models.note_executor import NoteExecutor
from src.tokenization import ArtistTokenizer, ChordTokenizer, PhraseTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS           = 40
BATCH_SIZE       = 512
LR               = 5e-4
WEIGHT_DECAY     = 0.01
ETA_MIN          = 1e-5
WARMUP_FRAC      = 0.05
LABEL_SMOOTHING  = 0.0
PATIENCE         = 4
NUM_WORKERS      = 4
SEED             = 42
GRAD_CLIP        = 1.0
GEN_EVERY_EPOCHS = 2
CKPT_DIR         = Path("checkpoints")
BEST_PATH        = CKPT_DIR / "v6.1.0_best.pt"
LATEST_PATH      = CKPT_DIR / "v6.1.0_latest.pt"
WANDB_PROJECT    = "jazz-solo-generator"
WANDB_NAME       = "note-executor-v6.1.0-literal-harmonic-finetune"
HARMONIC_WEIGHT  = 0.15
STRONG_BEAT_POS  = frozenset({0, 4, 8, 12})
DEFAULT_CACHE    = Path("data/processed/notes_v6_cache.pt")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train NoteExecutor v6")
parser.add_argument("--epochs",           type=int, default=EPOCHS,
                    help=f"Number of training epochs (default {EPOCHS})")
parser.add_argument("--dry-run-batches",  type=int, default=0, dest="dry_run_batches",
                    help="Limit train+val to N batches per epoch (0 = no limit)")
parser.add_argument("--resume",           type=Path, default=None,
                    help="Checkpoint to load model weights from before training")
parser.add_argument("--lr",               type=float, default=LR,
                    help=f"Learning rate (default {LR})")
parser.add_argument("--cache",            type=Path, default=DEFAULT_CACHE,
                    help=f"NoteWindowDataset cache path (default {DEFAULT_CACHE})")
parser.add_argument("--run-name",         type=str, default=WANDB_NAME,
                    help=f"wandb run name (default {WANDB_NAME})")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def collate_with_tempo(batch):
    out = collate_note_window(batch)
    out["tempo_bpm"] = torch.stack([b["tempo_bpm"] for b in batch])
    return out


def _entropy(logits: torch.Tensor) -> float:
    mean = logits.detach().float().mean(0)
    p = torch.softmax(mean, dim=-1)
    return -(p * p.clamp_min(1e-12).log()).sum().item()


def _build_scheduler(opt, epochs, steps_per_epoch, lr):
    total_steps  = epochs * steps_per_epoch
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cos + (1.0 - cos) * (ETA_MIN / lr)

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _save_checkpoint(path, model, opt, epoch, val_loss):
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "epoch":           epoch,
        "val_loss":        val_loss,
        "vocab_version":   "v6",
        "dur_vocab_size":  18,
    }, path)


_ROOT_PC = {
    "C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
    "C#": 1, "D#": 3, "G#": 8, "A#": 10, "Cb": 11,
}
_ROOTS_SORTED = sorted(_ROOT_PC.keys(), key=len, reverse=True)


def _chord_to_pcs(symbol: str) -> set[int]:
    """Return chord-tone pitch classes for a WJazzD chord symbol."""
    if not symbol or symbol in {"NC", "", "<PAD>", "<UNK>", "<BOS>", "<EOS>"}:
        return set()
    root = None
    rest = ""
    for r in _ROOTS_SORTED:
        if symbol.startswith(r):
            root = _ROOT_PC[r]
            rest = symbol[len(r):]
            break
    if root is None:
        return set()
    if "m7b5" in rest:
        intervals = [0, 3, 6, 10]
    elif "o7" in rest:
        intervals = [0, 3, 6, 9]
    elif "-7" in rest:
        intervals = [0, 3, 7, 10]
    elif "j7" in rest:
        intervals = [0, 4, 7, 11]
    elif rest.startswith("7"):
        intervals = [0, 4, 7, 10]
    elif "-" in rest:
        intervals = [0, 3, 7]
    elif "6" in rest:
        intervals = [0, 4, 7, 9]
    elif "j" in rest:
        intervals = [0, 4, 7, 11]
    else:
        intervals = [0, 4, 7, 10]
    return {(root + interval) % 12 for interval in intervals}


def build_chord_tone_tensor(chord_tok) -> torch.Tensor:
    """Return (chord_vocab, 12) tensor; 1 where pitch class is a chord tone."""
    out = torch.zeros(chord_tok.vocab_size, 12)
    for token_id, symbol in chord_tok._inv.items():
        for pc in _chord_to_pcs(symbol):
            out[token_id, pc] = 1.0
    return out


def harmonic_penalty(
    pitch_logits:      torch.Tensor,
    pos_in_phrase:     torch.Tensor,
    chord_ids:         torch.Tensor,
    chord_tone_tensor: torch.Tensor,
) -> torch.Tensor:
    # v6.1.0 literal port: v5b and v6 chord_ids/pos_in_phrase share semantics,
    # so this is a true regression test.
    """Penalty = -log P(chord tone) at v5b literal strong-position anchors."""
    strong = torch.zeros(pos_in_phrase.size(0), dtype=torch.bool, device=pos_in_phrase.device)
    for p in STRONG_BEAT_POS:
        strong |= (pos_in_phrase == p)
    if not strong.any():
        return pitch_logits.sum() * 0.0

    logits = pitch_logits[strong]
    c_ids = chord_ids[strong, 0]
    tone_pc = chord_tone_tensor[c_ids]

    pc_map = torch.zeros(logits.size(1), dtype=torch.long, device=logits.device)
    pc_map[4:] = torch.arange(128, device=logits.device) % 12
    tone_tok = tone_pc[:, pc_map]
    tone_tok[:, :4] = 0

    probs = F.softmax(logits, dim=-1)
    chord_tone_prob = (probs * tone_tok).sum(dim=-1)
    return -chord_tone_prob.clamp(min=1e-8).log().mean()


def _load_model_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)

    src_chord_embed = state["chord_embed.weight"].detach().cpu()
    model_chord_embed = model.chord_embed.weight.detach().cpu()
    assert torch.equal(model_chord_embed, src_chord_embed), (
        "resume failed: chord_embed weights differ"
    )
    return ckpt


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, opt, scheduler, scaler, ce, device,
                 chord_tone_tensor, epoch, global_step, dry_run_batches):
    model.train()
    n_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        if dry_run_batches and batch_idx >= dry_run_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
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
            dur_l   = ce(d_logits, batch["target_dur"])
            rest_l  = ce(r_logits, batch["target_rest"])
            harm_l  = harmonic_penalty(
                p_logits, batch["pos_in_phrase"], batch["chord_ids"], chord_tone_tensor
            )
            total   = pitch_l + dur_l + rest_l + HARMONIC_WEIGHT * harm_l

        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        scheduler.step()

        cur_lr     = scheduler.get_last_lr()[0]
        h_dur      = _entropy(d_logits)
        h_pitch    = _entropy(p_logits)
        global_step += 1

        wandb.log({
            "train/loss":               total.item(),
            "train/pitch_loss":         pitch_l.item(),
            "train/dur_loss":           dur_l.item(),
            "train/rest_loss":          rest_l.item(),
            "train/harmonic_loss":      harm_l.item(),
            "train/lr":                 cur_lr,
            "train/dur_token_entropy":  h_dur,
            "train/pitch_token_entropy": h_pitch,
        }, step=global_step)
        print(
            f"epoch {epoch:3d} batch {batch_idx:5d}/{n_batches} "
            f"loss={total.item():.4f} pitch={pitch_l.item():.4f} "
            f"dur={dur_l.item():.4f} rest={rest_l.item():.4f} "
            f"harm={harm_l.item():.4f} "
            f"lr={cur_lr:.2e} H_dur={h_dur:.3f} H_pitch={h_pitch:.3f}"
        )

    return global_step


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(model, loader, device, ce, chord_tone_tensor, dry_run_batches):
    model.eval()
    sums = {"loss": 0.0, "pitch": 0.0, "dur": 0.0, "rest": 0.0, "harmonic": 0.0}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if dry_run_batches and i >= dry_run_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                p, d, r = model(
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
                pl = ce(p, batch["target_pitch"])
                dl = ce(d, batch["target_dur"])
                rl = ce(r, batch["target_rest"])
                hl = harmonic_penalty(p, batch["pos_in_phrase"], batch["chord_ids"], chord_tone_tensor)
                tot = pl + dl + rl + HARMONIC_WEIGHT * hl
            sums["loss"]  += tot.item()
            sums["pitch"] += pl.item()
            sums["dur"]   += dl.item()
            sums["rest"]  += rl.item()
            sums["harmonic"] += hl.item()
            n += 1

    return {k: v / max(1, n) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Generation sample: ii-V-I in C (D-7 → G7 → Cj7)
# ---------------------------------------------------------------------------

def _sanity_generate_iiVI(model, chord_tok, phrase_tok, artist_tok, tempo, device):
    model.eval()
    chord_ids = torch.tensor(
        [[chord_tok.encode("D-7"), chord_tok.encode("G7"), chord_tok.encode("Cj7")]],
        dtype=torch.long, device=device,
    )
    phrase_id = torch.tensor(
        [phrase_tok.encode("PHRASE_00")], dtype=torch.long, device=device
    )
    artist_id = torch.tensor(
        [artist_tok.encode("Charlie Parker")], dtype=torch.long, device=device
    )
    notes = model.generate(
        chord_ids=chord_ids, phrase_id=phrase_id, artist_id=artist_id,
        n_notes=16, tempo_bpm=float(tempo),
    )
    lines = [f"ii-V-I in C @ {tempo:.1f} BPM  (pitch tok / dur tok / rest)"]
    for i, (p, d, r) in enumerate(notes):
        lines.append(f"  {i:2d}: pitch={p:3d}  dur={d:2d}  rest={r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parser.parse_args()
    epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    chord_tok  = ChordTokenizer.from_json()
    artist_tok = ArtistTokenizer()
    phrase_tok = PhraseTokenizer()
    chord_tone_tensor = build_chord_tone_tensor(chord_tok).to(device)

    print(f"Loading v6 cache: {args.cache}")
    full_ds = NoteWindowDataset.from_cache(args.cache)
    print(f"Loaded cache: {len(full_ds):,} samples")

    torch.manual_seed(SEED)
    n_train  = int(0.8 * len(full_ds))
    n_val    = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    print(f"Split: {n_train:,} train  /  {n_val:,} val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_with_tempo, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_with_tempo,
    )

    model = NoteExecutor(
        chord_vocab_size  = chord_tok.vocab_size,
        artist_vocab_size = artist_tok.vocab_size,
        dropout           = 0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    if args.resume:
        _load_model_checkpoint(model, args.resume, device)
        print(f"Loaded model weights from {args.resume}")

    ce        = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = _build_scheduler(opt, epochs, len(train_loader), args.lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    gen_tempo = float(full_ds[0]["tempo_bpm"].item())

    wandb.init(
        project = WANDB_PROJECT,
        name    = args.run_name,
        config  = dict(
            epochs=epochs, batch_size=BATCH_SIZE, lr=args.lr,
            weight_decay=WEIGHT_DECAY, eta_min=ETA_MIN,
            warmup_frac=WARMUP_FRAC, label_smoothing=LABEL_SMOOTHING,
            harmonic_weight=HARMONIC_WEIGHT,
            strong_beat_source="pos_in_phrase_literal_v5b",
            target_chord_source="chord_ids_col0_literal_v5b",
            patience=PATIENCE, grad_clip=GRAD_CLIP,
            d_model=256, nhead=8, num_encoder_layers=4,
            num_decoder_layers=4, dim_feedforward=512, dropout=0.1,
            dur_vocab_size=18, vocab_version="v6",
            n_train=n_train, n_val=n_val, cache=str(args.cache),
            resume=str(args.resume) if args.resume else None,
        ),
    )

    best_val          = float("inf")
    epochs_no_improve = 0
    global_step       = 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        global_step = _train_epoch(
            model, train_loader, opt, scheduler, scaler, ce, device,
            chord_tone_tensor, epoch, global_step, args.dry_run_batches,
        )

        val_metrics = _validate(
            model, val_loader, device, ce, chord_tone_tensor, args.dry_run_batches
        )

        val_log = {
            "val/loss":       val_metrics["loss"],
            "val/pitch_loss": val_metrics["pitch"],
            "val/dur_loss":   val_metrics["dur"],
            "val/rest_loss":  val_metrics["rest"],
            "val/harmonic_loss": val_metrics["harmonic"],
        }
        wandb.log(val_log, step=global_step)
        print(
            f"[epoch {epoch:3d}] val  loss={val_metrics['loss']:.4f} "
            f"pitch={val_metrics['pitch']:.4f} dur={val_metrics['dur']:.4f} "
            f"rest={val_metrics['rest']:.4f} harm={val_metrics['harmonic']:.4f}"
        )

        _save_checkpoint(LATEST_PATH, model, opt, epoch, val_metrics["loss"])
        if val_metrics["loss"] < best_val:
            best_val          = val_metrics["loss"]
            epochs_no_improve = 0
            _save_checkpoint(BEST_PATH, model, opt, epoch, best_val)
            print(f"  ↳ new best val={best_val:.4f}, saved to {BEST_PATH}")
        else:
            epochs_no_improve += 1
            print(f"  no improvement ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

        if epoch % GEN_EVERY_EPOCHS == 0:
            text = _sanity_generate_iiVI(
                model, chord_tok, phrase_tok, artist_tok, gen_tempo, device
            )
            print(text)
            wandb.log({"gen/text": text}, step=global_step)

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
