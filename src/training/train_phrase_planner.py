from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, random_split

from src.data.collate import collate_phrase_planner
from src.data.datasets import PhrasePlannerDataset
from src.models.phrase_planner import PhrasePlanner
from src.tokenization import ArtistTokenizer, ChordTokenizer, PhraseTokenizer

_REPO = Path(__file__).parents[2]
CHECKPOINT_DIR = _REPO / "checkpoints"

EPOCHS = 60
PATIENCE = 15
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42


def _train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        chord_ids = batch["chord_ids"].to(device)
        phrase_ids = batch["phrase_ids"].to(device)
        artist_id = batch["artist_id"].to(device)
        chord_mask = batch["chord_mask"].to(device)
        phrase_mask = batch["phrase_mask"].to(device)

        tgt_in = phrase_ids[:, :-1]
        tgt_out = phrase_ids[:, 1:]
        src_pad = ~chord_mask
        tgt_pad = ~phrase_mask[:, :-1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=device
        )

        logits = model(
            chord_ids, artist_id, tgt_in,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
        )
        loss = loss_fn(logits.reshape(-1, model.phrase_vocab_size), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _val_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for batch in loader:
        chord_ids = batch["chord_ids"].to(device)
        phrase_ids = batch["phrase_ids"].to(device)
        artist_id = batch["artist_id"].to(device)
        chord_mask = batch["chord_mask"].to(device)
        phrase_mask = batch["phrase_mask"].to(device)

        tgt_in = phrase_ids[:, :-1]
        tgt_out = phrase_ids[:, 1:]
        src_pad = ~chord_mask
        tgt_pad = ~phrase_mask[:, :-1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=device
        )

        logits = model(
            chord_ids, artist_id, tgt_in,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
        )
        loss = loss_fn(logits.reshape(-1, model.phrase_vocab_size), tgt_out.reshape(-1))
        total += loss.item()
    return total / len(loader)


def _generate_demo(model, dataset, chord_tok, phrase_tok, device):
    model.eval()
    results = []

    # Find one Parker and one Davis sample in dataset by scanning
    parker_sample = None
    davis_sample = None
    for i in range(len(dataset)):
        s = dataset[i]
        artist = s["artist_id"].item()
        if artist == 1 and parker_sample is None:
            parker_sample = s
        if artist == 2 and davis_sample is None:
            davis_sample = s
        if parker_sample and davis_sample:
            break

    for desc, sample in [("Parker solo", parker_sample), ("Davis solo", davis_sample)]:
        chord_ids = sample["chord_ids"].unsqueeze(0).to(device)
        artist_id = sample["artist_id"].unsqueeze(0).to(device)
        tokens = model.generate(chord_ids, artist_id)
        labels = [phrase_tok.decode(t) for t in tokens]
        results.append((desc, str(tokens), " ".join(labels)))
        print(f"\n{desc}: {' '.join(labels)}")

    # 12-bar blues in F
    blues_chords = ["F7", "Bb7", "F7", "Bb7", "F7", "C7", "Bb7", "F7"]
    chord_ids = torch.tensor(
        [[chord_tok.encode(c) for c in blues_chords]], dtype=torch.long, device=device
    )
    artist_id = torch.tensor([1], dtype=torch.long, device=device)  # Parker style
    tokens = model.generate(chord_ids, artist_id)
    labels = [phrase_tok.decode(t) for t in tokens]
    results.append(("12-bar blues (F, Parker style)", str(tokens), " ".join(labels)))
    print(f"\n12-bar blues: {' '.join(labels)}")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    chord_tok = ChordTokenizer.from_json()
    artist_tok = ArtistTokenizer()
    phrase_tok = PhraseTokenizer()

    dataset = PhrasePlannerDataset(chord_tok, None, artist_tok, phrase_tok)
    torch.manual_seed(SEED)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"Train: {n_train} solos  Val: {n_val} solos")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_phrase_planner
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_phrase_planner
    )

    model = PhrasePlanner(
        chord_vocab_size=chord_tok.vocab_size,
        artist_vocab_size=artist_tok.vocab_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    wandb.init(
        project="jazz-solo-generator",
        name="phrase-planner-v3-reclustered",
        config={
            "d_model": 128, "nhead": 4, "num_layers": 4,
            "dropout": 0.3, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS,
            "train_solos": n_train, "val_solos": n_val,
        },
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = _val_epoch(model, val_loader, loss_fn, device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "phrase_planner_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    torch.save(model.state_dict(), CHECKPOINT_DIR / "phrase_planner_final.pt")
    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {CHECKPOINT_DIR}/")

    # Load best weights for generation
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "phrase_planner_best.pt", map_location=device))
    results = _generate_demo(model, dataset, chord_tok, phrase_tok, device)

    table = wandb.Table(columns=["description", "phrase_token_ids", "phrase_labels"])
    for row in results:
        table.add_data(*row)
    wandb.log({"generated_phrases": table})
    wandb.finish()


if __name__ == "__main__":
    main()
