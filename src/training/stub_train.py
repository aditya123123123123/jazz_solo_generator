import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.collate import collate_note_executor, collate_phrase_planner
from src.data.datasets import NoteExecutorDataset, PhrasePlannerDataset
from src.tokenization import ArtistTokenizer, ChordTokenizer, NoteTokenizer, PhraseTokenizer

D_MODEL = 32
NHEAD = 2
NUM_LAYERS = 2
DIM_FF = 64
BATCH_SIZE = 4


def sinusoidal_pe(x):
    B, L, D = x.shape
    pos = torch.arange(L, device=x.device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, D, 2, device=x.device).float() * (-math.log(10000.0) / D))
    pe = torch.zeros(L, D, device=x.device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return x + pe.unsqueeze(0)


def causal_mask(size, device):
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)


def run_phrase_planner_stub(chord_tok, phrase_tok, artist_tok):
    print("\n=== PhrasePlanner stub ===")
    dataset = PhrasePlannerDataset(chord_tok, None, artist_tok, phrase_tok)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_phrase_planner, shuffle=False)
    batch = next(iter(loader))

    chord_embed = nn.Embedding(chord_tok.vocab_size, D_MODEL, padding_idx=0)
    artist_embed = nn.Embedding(artist_tok.vocab_size, D_MODEL)
    phrase_embed = nn.Embedding(phrase_tok.vocab_size, D_MODEL, padding_idx=0)
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(D_MODEL, NHEAD, DIM_FF, batch_first=True, dropout=0.0),
        num_layers=NUM_LAYERS,
    )
    decoder = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(D_MODEL, NHEAD, DIM_FF, batch_first=True, dropout=0.0),
        num_layers=NUM_LAYERS,
    )
    phrase_head = nn.Linear(D_MODEL, phrase_tok.vocab_size)

    chord_ids = batch["chord_ids"]
    phrase_ids = batch["phrase_ids"]
    artist_id = batch["artist_id"]
    chord_mask = batch["chord_mask"]

    src = chord_embed(chord_ids) + artist_embed(artist_id).unsqueeze(1)
    print(f"  Embeddings (chord + artist broadcast): {src.shape}")

    src = sinusoidal_pe(src)
    print(f"  After positional encoding:             {src.shape}")

    src_pad_mask = ~chord_mask
    memory = encoder(src, src_key_padding_mask=src_pad_mask)
    print(f"  Encoder output (memory):               {memory.shape}")

    tgt_in = phrase_ids[:, :-1]
    tgt_emb = phrase_embed(tgt_in)
    tgt_emb = sinusoidal_pe(tgt_emb)
    tgt_sz = tgt_in.size(1)
    tgt_mask = causal_mask(tgt_sz, src.device)

    out = decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_pad_mask)
    print(f"  Decoder output:                        {out.shape}")

    logits = phrase_head(out)
    print(f"  Logits:                                {logits.shape}")

    target = phrase_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, phrase_tok.vocab_size), target.reshape(-1), ignore_index=0)
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()
    print("  Gradients: OK")


def run_note_executor_stub(chord_tok, note_tok, artist_tok, phrase_tok):
    print("\n=== NoteExecutor stub ===")
    dataset = NoteExecutorDataset(chord_tok, note_tok, artist_tok, phrase_tok)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_note_executor, shuffle=False)
    batch = next(iter(loader))

    chord_embed = nn.Embedding(chord_tok.vocab_size, D_MODEL, padding_idx=0)
    phrase_embed = nn.Embedding(phrase_tok.vocab_size, D_MODEL)
    artist_embed = nn.Embedding(artist_tok.vocab_size, D_MODEL)
    pitch_embed = nn.Embedding(note_tok.pitch_vocab_size, D_MODEL, padding_idx=0)
    dur_embed = nn.Embedding(note_tok.dur_vocab_size, D_MODEL, padding_idx=0)

    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(D_MODEL, NHEAD, DIM_FF, batch_first=True, dropout=0.0),
        num_layers=NUM_LAYERS,
    )
    decoder = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(D_MODEL, NHEAD, DIM_FF, batch_first=True, dropout=0.0),
        num_layers=NUM_LAYERS,
    )
    pitch_head = nn.Linear(D_MODEL, note_tok.pitch_vocab_size)
    dur_head = nn.Linear(D_MODEL, note_tok.dur_vocab_size)

    chord_ids = batch["chord_ids"]
    pitch_ids = batch["pitch_ids"]
    dur_ids = batch["dur_ids"]
    phrase_id = batch["phrase_id"]
    artist_id = batch["artist_id"]
    chord_mask = batch["chord_mask"]

    phrase_emb = phrase_embed(phrase_id).unsqueeze(1)
    artist_emb = artist_embed(artist_id).unsqueeze(1)
    chord_emb = chord_embed(chord_ids)
    src = torch.cat([phrase_emb, artist_emb, chord_emb], dim=1)
    print(f"  Encoder input [phrase|artist|chords]:  {src.shape}")

    src = sinusoidal_pe(src)
    print(f"  After positional encoding:             {src.shape}")

    # Extend chord_mask with True for the two prepended tokens
    B = chord_mask.size(0)
    prefix_mask = torch.ones(B, 2, dtype=torch.bool, device=chord_mask.device)
    src_mask = torch.cat([prefix_mask, chord_mask], dim=1)
    memory = encoder(src, src_key_padding_mask=~src_mask)
    print(f"  Encoder output (memory):               {memory.shape}")

    tgt_in_pitch = pitch_ids[:, :-1]
    tgt_in_dur = dur_ids[:, :-1]
    tgt_emb = pitch_embed(tgt_in_pitch) + dur_embed(tgt_in_dur)
    tgt_emb = sinusoidal_pe(tgt_emb)
    print(f"  Decoder target embedding:              {tgt_emb.shape}")

    tgt_sz = tgt_emb.size(1)
    tgt_mask = causal_mask(tgt_sz, src.device)
    out = decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=~src_mask)
    print(f"  Decoder output:                        {out.shape}")

    pitch_logits = pitch_head(out)
    dur_logits = dur_head(out)
    print(f"  Pitch logits:                          {pitch_logits.shape}")
    print(f"  Duration logits:                       {dur_logits.shape}")

    target_pitch = pitch_ids[:, 1:]
    target_dur = dur_ids[:, 1:]
    loss = (
        F.cross_entropy(pitch_logits.reshape(-1, note_tok.pitch_vocab_size), target_pitch.reshape(-1), ignore_index=0)
        + F.cross_entropy(dur_logits.reshape(-1, note_tok.dur_vocab_size), target_dur.reshape(-1), ignore_index=0)
    )
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()
    print("  Gradients: OK")


if __name__ == "__main__":
    chord_tok = ChordTokenizer.from_json()
    note_tok = NoteTokenizer.from_json()
    artist_tok = ArtistTokenizer()
    phrase_tok = PhraseTokenizer()

    run_phrase_planner_stub(chord_tok, phrase_tok, artist_tok)
    run_note_executor_stub(chord_tok, note_tok, artist_tok, phrase_tok)

    print("\nEND-TO-END STUB PASSED")
