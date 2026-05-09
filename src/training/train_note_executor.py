from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, random_split

from src.data.note_dataset import NoteWindowDataset, collate_note_window
from src.models.note_executor import NoteExecutor
from src.tokenization import ArtistTokenizer, ChordTokenizer, NoteTokenizer, PhraseTokenizer

_REPO = Path(__file__).parents[2]
CHECKPOINT_DIR = _REPO / "checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "note_executor_best.pt"

EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
BEAT_DURATION = 0.5  # seconds per beat at 120 BPM


def _train_epoch(model, loader, optimizer, pitch_fn, dur_fn, rest_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        chord_ids = batch["chord_ids"].to(device)
        chord_mask = batch["chord_mask"].to(device)
        phrase_id = batch["phrase_id"].to(device)
        artist_id = batch["artist_id"].to(device)
        ctx_pitch = batch["ctx_pitch"].to(device)
        ctx_dur = batch["ctx_dur"].to(device)
        ctx_rest = batch["ctx_rest"].to(device)
        t_pitch = batch["target_pitch"].to(device)
        t_dur = batch["target_dur"].to(device)
        t_rest = batch["target_rest"].to(device)

        p_logits, d_logits, r_logits = model(
            chord_ids, phrase_id, artist_id,
            ctx_pitch, ctx_dur, ctx_rest,
            src_key_padding_mask=~chord_mask,
        )
        loss = pitch_fn(p_logits, t_pitch) + dur_fn(d_logits, t_dur) + rest_fn(r_logits, t_rest)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _val_epoch(model, loader, pitch_fn, dur_fn, rest_fn, device):
    model.eval()
    total = 0.0
    for batch in loader:
        chord_ids = batch["chord_ids"].to(device)
        chord_mask = batch["chord_mask"].to(device)
        phrase_id = batch["phrase_id"].to(device)
        artist_id = batch["artist_id"].to(device)
        ctx_pitch = batch["ctx_pitch"].to(device)
        ctx_dur = batch["ctx_dur"].to(device)
        ctx_rest = batch["ctx_rest"].to(device)
        t_pitch = batch["target_pitch"].to(device)
        t_dur = batch["target_dur"].to(device)
        t_rest = batch["target_rest"].to(device)

        p_logits, d_logits, r_logits = model(
            chord_ids, phrase_id, artist_id,
            ctx_pitch, ctx_dur, ctx_rest,
            src_key_padding_mask=~chord_mask,
        )
        loss = pitch_fn(p_logits, t_pitch) + dur_fn(d_logits, t_dur) + rest_fn(r_logits, t_rest)
        total += loss.item()
    return total / len(loader)


def _sanity_generate(model, chord_tok, note_tok, phrase_tok, artist_tok, device):
    model.eval()
    # Try Cmaj7 in WJazzD notation ("Cj7"), fall back to F7 if not in vocab
    for chord_str in ("Cj7", "Cmaj7", "F7"):
        chord_id = chord_tok.encode(chord_str)
        if chord_id != chord_tok.UNK:
            break

    print(f"\nSanity generation: chord={chord_str!r}, phrase=PHRASE_18 (Parker style)")
    chord_ids = torch.tensor([[chord_id]], dtype=torch.long, device=device)
    phrase_id  = torch.tensor([phrase_tok.encode("PHRASE_18")], dtype=torch.long, device=device)
    artist_id  = torch.tensor([artist_tok.encode("Charlie Parker")], dtype=torch.long, device=device)

    generated = model.generate(chord_ids, phrase_id, artist_id, n_notes=16)

    print(f"\nFirst 16 notes  (pitch_midi, duration_beats, velocity, is_rest):")
    print(f"  {'#':>2}  {'pitch':>5}  {'dur_beats':>9}  {'vel':>3}  {'rest':>4}")
    print("  " + "-" * 35)
    for i, (p_tok, d_tok, r_tok) in enumerate(generated):
        pitch_midi  = note_tok.decode_pitch(p_tok)
        dur_seconds = note_tok.decode_duration(d_tok)
        dur_beats   = dur_seconds / BEAT_DURATION
        velocity    = 64  # no velocity in corpus; fixed default
        print(f"  {i+1:>2}  {pitch_midi:>5}  {dur_beats:>9.2f}  {velocity:>3}  {r_tok:>4}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    chord_tok  = ChordTokenizer.from_json()
    note_tok   = NoteTokenizer.from_json()
    artist_tok = ArtistTokenizer()
    phrase_tok = PhraseTokenizer()

    print("Building NoteWindowDataset …")
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pitch_fn = nn.CrossEntropyLoss(ignore_index=0)
    dur_fn   = nn.CrossEntropyLoss(ignore_index=0)
    rest_fn  = nn.CrossEntropyLoss()

    wandb.init(
        project="jazz-solo-generator",
        name="note-executor-v1",
        config={
            "d_model": 256, "nhead": 8, "num_layers": 4,
            "dim_feedforward": 512, "dropout": 0.1,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "patience": PATIENCE,
            "train_samples": n_train, "val_samples": n_val,
        },
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = _train_epoch(model, train_loader, optimizer,
                                  pitch_fn, dur_fn, rest_fn, device)
        val_loss   = _val_epoch(model, val_loader, pitch_fn, dur_fn, rest_fn, device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_CHECKPOINT)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {BEST_CHECKPOINT}")

    model.load_state_dict(torch.load(BEST_CHECKPOINT, map_location=device))
    _sanity_generate(model, chord_tok, note_tok, phrase_tok, artist_tok, device)

    wandb.finish()


if __name__ == "__main__":
    main()
