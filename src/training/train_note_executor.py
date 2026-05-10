from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, random_split

from src.data.note_dataset import NoteWindowDataset, collate_note_window
from src.models.note_executor import NoteExecutor
from src.tokenization import ArtistTokenizer, ChordTokenizer, NoteTokenizer, PhraseTokenizer

_REPO = Path(__file__).parents[2]
CHECKPOINT_DIR   = _REPO / "checkpoints"
BEST_CHECKPOINT  = CHECKPOINT_DIR / "note_executor_best.pt"

EPOCHS          = 80
PATIENCE        = 20
BATCH_SIZE      = 64
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
SEED            = 42
HARMONIC_WEIGHT = 0.15
BEAT_DURATION   = 0.5

STRONG_BEAT_POS = frozenset({0, 4, 8, 12})

# ---------------------------------------------------------------------------
# Chord-tone tensor (vocab_size, 12) — used for harmonic awareness loss
# ---------------------------------------------------------------------------

_ROOT_PC = {
    'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11,
    'C#': 1, 'D#': 3, 'G#': 8, 'A#': 10, 'Cb': 11,
}
_ROOTS_SORTED = sorted(_ROOT_PC.keys(), key=len, reverse=True)


def _chord_to_pcs(symbol: str) -> set:
    """Return set of pitch classes (0-11) for a WJazzD chord symbol."""
    if not symbol or symbol in {'NC', '', '<PAD>', '<UNK>', '<BOS>', '<EOS>'}:
        return set()
    root = None
    for r in _ROOTS_SORTED:
        if symbol.startswith(r):
            root = _ROOT_PC[r]
            rest = symbol[len(r):]
            break
    if root is None:
        return set()
    if 'm7b5' in rest:      ivs = [0, 3, 6, 10]
    elif 'o7' in rest:      ivs = [0, 3, 6, 9]
    elif '-7' in rest:      ivs = [0, 3, 7, 10]
    elif 'j7' in rest:      ivs = [0, 4, 7, 11]
    elif rest.startswith('7'): ivs = [0, 4, 7, 10]
    elif '-' in rest:       ivs = [0, 3, 7]
    elif '6' in rest:       ivs = [0, 4, 7, 9]
    elif 'j' in rest:       ivs = [0, 4, 7, 11]
    else:                   ivs = [0, 4, 7, 10]   # default dom7
    return {(root + i) % 12 for i in ivs}


def build_chord_tone_tensor(chord_tok) -> torch.Tensor:
    """Return (vocab_size, 12) float tensor; 1 where pitch class is a chord tone."""
    vs  = chord_tok.vocab_size
    out = torch.zeros(vs, 12)
    for token_id, symbol in chord_tok._inv.items():
        for pc in _chord_to_pcs(symbol):
            out[token_id, pc] = 1.0
    return out


def harmonic_penalty(
    pitch_logits:       torch.Tensor,   # (B, pitch_vocab)
    pos_in_phrase:      torch.Tensor,   # (B,)
    chord_ids:          torch.Tensor,   # (B, Lc)
    chord_tone_tensor:  torch.Tensor,   # (chord_vocab, 12) on same device
) -> torch.Tensor:
    """Penalty = -log P(chord tone) at strong beat positions."""
    strong = torch.zeros(pos_in_phrase.size(0), dtype=torch.bool, device=pos_in_phrase.device)
    for p in STRONG_BEAT_POS:
        strong |= (pos_in_phrase == p)
    if not strong.any():
        return pitch_logits.sum() * 0.0   # keeps grad graph alive, value 0

    logits  = pitch_logits[strong]                            # (Bs, 132)
    c_ids   = chord_ids[strong, 0]                            # (Bs,)  first chord token
    tone_pc = chord_tone_tensor[c_ids]                        # (Bs, 12)

    # Map each pitch token to its pitch class
    pc_map      = torch.zeros(logits.size(1), dtype=torch.long, device=logits.device)
    pc_map[4:]  = torch.arange(128, device=logits.device) % 12
    tone_tok    = tone_pc[:, pc_map]                          # (Bs, 132)
    tone_tok[:, :4] = 0                                       # special tokens never count

    probs           = F.softmax(logits, dim=-1)               # (Bs, 132)
    chord_tone_prob = (probs * tone_tok).sum(dim=-1)          # (Bs,)
    return -chord_tone_prob.clamp(min=1e-8).log().mean()


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def _step(batch, model, device, chord_tone_tensor, pitch_fn, dur_fn, rest_fn):
    chord_ids     = batch["chord_ids"].to(device)
    chord_mask    = batch["chord_mask"].to(device)
    phrase_id     = batch["phrase_id"].to(device)
    artist_id     = batch["artist_id"].to(device)
    ctx_pitch     = batch["ctx_pitch"].to(device)
    ctx_dur       = batch["ctx_dur"].to(device)
    ctx_rest      = batch["ctx_rest"].to(device)
    t_pitch       = batch["target_pitch"].to(device)
    t_dur         = batch["target_dur"].to(device)
    t_rest        = batch["target_rest"].to(device)
    pos_in_phrase = batch["pos_in_phrase"].to(device)

    p_logits, d_logits, r_logits = model(
        chord_ids, phrase_id, artist_id,
        ctx_pitch, ctx_dur, ctx_rest, pos_in_phrase,
        src_key_padding_mask=~chord_mask,
    )
    ce   = pitch_fn(p_logits, t_pitch) + dur_fn(d_logits, t_dur) + rest_fn(r_logits, t_rest)
    harm = harmonic_penalty(p_logits, pos_in_phrase, chord_ids,
                            chord_tone_tensor.to(device))
    return ce + HARMONIC_WEIGHT * harm


def _train_epoch(model, loader, optimizer, device, chord_tone_tensor,
                 pitch_fn, dur_fn, rest_fn):
    model.train()
    total = 0.0
    for batch in loader:
        loss = _step(batch, model, device, chord_tone_tensor, pitch_fn, dur_fn, rest_fn)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _val_epoch(model, loader, device, chord_tone_tensor, pitch_fn, dur_fn, rest_fn):
    model.eval()
    total = 0.0
    for batch in loader:
        total += _step(batch, model, device, chord_tone_tensor, pitch_fn, dur_fn, rest_fn).item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Sanity generation (unchanged chord set, now with cross-chord context)
# ---------------------------------------------------------------------------

def _sanity_generate(model, chord_tok, note_tok, phrase_tok, artist_tok, device):
    model.eval()
    for chord_str in ("Cj7", "F7"):
        chord_id = chord_tok.encode(chord_str)
        if chord_id != chord_tok.UNK:
            break

    print(f"\nSanity generation: chord={chord_str!r}, phrase=PHRASE_18 (Parker style)")
    chord_ids = torch.tensor([[chord_id]], dtype=torch.long, device=device)
    phrase_id = torch.tensor([phrase_tok.encode("PHRASE_18")], dtype=torch.long, device=device)
    artist_id = torch.tensor([artist_tok.encode("Charlie Parker")], dtype=torch.long, device=device)

    notes = model.generate(chord_ids, phrase_id, artist_id, n_notes=16)
    print(f"\n  {'#':>2}  {'pitch':>5}  {'beats':>8}  {'rest':>4}")
    for i, (p_tok, d_tok, r_tok) in enumerate(notes):
        midi  = note_tok.decode_pitch(p_tok)
        beats = note_tok.decode_duration(d_tok) / BEAT_DURATION
        print(f"  {i+1:>2}  {midi:>5}  {beats:>8.2f}  {r_tok:>4}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    chord_tok  = ChordTokenizer.from_json()
    note_tok   = NoteTokenizer.from_json()
    artist_tok = ArtistTokenizer()
    phrase_tok = PhraseTokenizer()

    chord_tone_tensor = build_chord_tone_tensor(chord_tok)

    print("Building NoteWindowDataset (cross-phrase context) …")
    full_ds = NoteWindowDataset(chord_tok, note_tok, artist_tok, phrase_tok)
    print(f"Total samples: {len(full_ds):,}")

    torch.manual_seed(SEED)
    n_train = int(0.8 * len(full_ds))
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    print(f"Train: {n_train:,}  Val: {n_val:,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_note_window, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_note_window, num_workers=0,
    )

    model = NoteExecutor(
        chord_vocab_size=chord_tok.vocab_size,
        artist_vocab_size=artist_tok.vocab_size,
        dropout=0.25,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    pitch_fn  = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    dur_fn    = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    rest_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)

    wandb.init(
        project="jazz-solo-generator",
        name="note-executor-v4",
        config={
            "d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 512,
            "dropout": 0.25, "label_smoothing": 0.1, "harmonic_weight": HARMONIC_WEIGHT,
            "lr": LR, "scheduler": "cosine", "T_max": EPOCHS, "eta_min": 1e-5,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "patience": PATIENCE,
            "cross_phrase_context": True, "pos_in_phrase_embed": True,
            "train_samples": n_train, "val_samples": n_val,
        },
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss    = float("inf")
    best_epoch       = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device,
                                  chord_tone_tensor, pitch_fn, dur_fn, rest_fn)
        val_loss   = _val_epoch(model, val_loader, device,
                                chord_tone_tensor, pitch_fn, dur_fn, rest_fn)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": lr}, step=epoch)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_CHECKPOINT)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    torch.save(model.state_dict(), CHECKPOINT_DIR / "note_executor_final.pt")
    print(f"\nBest val_loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {BEST_CHECKPOINT}")

    model.load_state_dict(torch.load(BEST_CHECKPOINT, map_location=device))
    _sanity_generate(model, chord_tok, note_tok, phrase_tok, artist_tok, device)
    wandb.finish()


if __name__ == "__main__":
    main()
