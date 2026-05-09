import math

import torch
import torch.nn as nn

# Note: the WJazzD corpus does not include per-note velocity.
# This model predicts pitch, quantised duration, and is_rest (gap flag).
# Velocity is omitted; callers should supply a fixed default (e.g. 64).


class NoteExecutor(nn.Module):
    def __init__(
        self,
        chord_vocab_size:  int   = 196,
        phrase_vocab_size: int   = 67,
        artist_vocab_size: int   = 16,
        pitch_vocab_size:  int   = 132,
        dur_vocab_size:    int   = 19,
        d_model:           int   = 256,
        nhead:             int   = 8,
        num_encoder_layers:int   = 4,
        num_decoder_layers:int   = 4,
        dim_feedforward:   int   = 512,
        dropout:           float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pitch_vocab_size = pitch_vocab_size
        self.dur_vocab_size   = dur_vocab_size

        # Encoder embeddings (chord context + artist bias at position 0)
        self.chord_embed  = nn.Embedding(chord_vocab_size,  d_model, padding_idx=0)
        self.artist_embed = nn.Embedding(artist_vocab_size, d_model)

        # Phrase prefix embedding – broadcast-added to every decoder position
        # so the phrase type conditions the full note generation, not just position 0
        self.phrase_embed = nn.Embedding(phrase_vocab_size, d_model)

        # Decoder embeddings (note history)
        self.pitch_embed  = nn.Embedding(pitch_vocab_size, d_model, padding_idx=0)
        self.dur_embed    = nn.Embedding(dur_vocab_size,   d_model, padding_idx=0)
        self.is_rest_embed = nn.Embedding(2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.pitch_head  = nn.Linear(d_model, pitch_vocab_size)
        self.dur_head    = nn.Linear(d_model, dur_vocab_size)
        self.is_rest_head = nn.Linear(d_model, 2)

    def _sinusoidal_pe(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, D, 2, device=x.device).float() * (-math.log(10000.0) / D)
        )
        pe = torch.zeros(L, D, device=x.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.unsqueeze(0)

    def encode(
        self,
        chord_ids: torch.Tensor,
        artist_id: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        src = self.chord_embed(chord_ids)
        print(f"  [NoteExecutor] chord_embed:  {src.shape}")
        src[:, 0, :] = src[:, 0, :] + self.artist_embed(artist_id)
        src = self._sinusoidal_pe(src)
        print(f"  [NoteExecutor] after PE:     {src.shape}")
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        print(f"  [NoteExecutor] encoder out:  {memory.shape}")
        return memory

    def forward(
        self,
        chord_ids: torch.Tensor,
        phrase_id: torch.Tensor,
        artist_id: torch.Tensor,
        ctx_pitch: torch.Tensor,
        ctx_dur:   torch.Tensor,
        ctx_rest:  torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ):
        memory = self.encode(chord_ids, artist_id, src_key_padding_mask)

        # Combine note embeddings; inject phrase bias at every position
        note_emb = (
            self.pitch_embed(ctx_pitch)
            + self.dur_embed(ctx_dur)
            + self.is_rest_embed(ctx_rest)
        )
        print(f"  [NoteExecutor] note_emb:     {note_emb.shape}")
        phrase_bias = self.phrase_embed(phrase_id).unsqueeze(1)  # (B, 1, d_model)
        note_emb = note_emb + phrase_bias                        # broadcast to all positions
        note_emb = self._sinusoidal_pe(note_emb)

        tgt_len = note_emb.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=ctx_pitch.device)
        out = self.decoder(
            note_emb, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        print(f"  [NoteExecutor] decoder out:  {out.shape}")

        # Predict next note from last context position
        last = out[:, -1, :]
        pitch_logits  = self.pitch_head(last)
        dur_logits    = self.dur_head(last)
        rest_logits   = self.is_rest_head(last)
        print(f"  [NoteExecutor] pitch_logits: {pitch_logits.shape}  dur: {dur_logits.shape}  rest: {rest_logits.shape}")
        return pitch_logits, dur_logits, rest_logits

    @torch.no_grad()
    def generate(
        self,
        chord_ids: torch.Tensor,
        phrase_id: torch.Tensor,
        artist_id: torch.Tensor,
        n_notes: int = 16,
        window:  int = 8,
        bos_pitch: int = 1,
        bos_dur:   int = 1,
    ) -> list:
        self.eval()
        device = chord_ids.device
        memory = self.encode(chord_ids, artist_id, src_key_padding_mask=None)

        pitch_hist = [bos_pitch]
        dur_hist   = [bos_dur]
        rest_hist  = [0]
        generated  = []

        for _ in range(n_notes):
            ctx_p = pitch_hist[-window:]
            ctx_d = dur_hist[-window:]
            ctx_r = rest_hist[-window:]
            pad = window - len(ctx_p)
            ctx_p = [0] * pad + ctx_p
            ctx_d = [0] * pad + ctx_d
            ctx_r = [0] * pad + ctx_r

            cp = torch.tensor([ctx_p], dtype=torch.long, device=device)
            cd = torch.tensor([ctx_d], dtype=torch.long, device=device)
            cr = torch.tensor([ctx_r], dtype=torch.long, device=device)

            note_emb = (
                self.pitch_embed(cp)
                + self.dur_embed(cd)
                + self.is_rest_embed(cr)
            )
            phrase_bias = self.phrase_embed(phrase_id).unsqueeze(1)
            note_emb = note_emb + phrase_bias
            note_emb = self._sinusoidal_pe(note_emb)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(window, device=device)
            out = self.decoder(note_emb, memory, tgt_mask=tgt_mask)
            last = out[:, -1, :]

            next_pitch = self.pitch_head(last).argmax(-1).item()
            next_dur   = self.dur_head(last).argmax(-1).item()
            next_rest  = self.is_rest_head(last).argmax(-1).item()

            generated.append((next_pitch, next_dur, next_rest))
            pitch_hist.append(next_pitch)
            dur_hist.append(next_dur)
            rest_hist.append(next_rest)

        return generated


if __name__ == "__main__":
    torch.manual_seed(0)
    B, Lc, W = 4, 12, 8
    chord_ids = torch.randint(4, 196, (B, Lc))
    phrase_id = torch.randint(3, 67,  (B,))
    artist_id = torch.randint(1, 16,  (B,))
    ctx_pitch = torch.randint(4, 132, (B, W))
    ctx_dur   = torch.randint(3, 19,  (B, W))
    ctx_rest  = torch.randint(0, 2,   (B, W))

    model = NoteExecutor()
    model.eval()
    with torch.no_grad():
        p_logits, d_logits, r_logits = model(chord_ids, phrase_id, artist_id,
                                              ctx_pitch, ctx_dur, ctx_rest)
    print(f"\nForward pass OK.")
    print(f"  pitch_logits: {p_logits.shape}")   # (4, 132)
    print(f"  dur_logits:   {d_logits.shape}")    # (4, 19)
    print(f"  rest_logits:  {r_logits.shape}")    # (4, 2)
    assert p_logits.shape == (B, 132)
    assert d_logits.shape == (B, 19)
    assert r_logits.shape == (B, 2)
    print("Shape assertions passed.")
