import math

import torch
import torch.nn as nn

# The WJazzD corpus has no per-note velocity.
# This model predicts pitch, quantised duration, and is_rest (gap flag).


class NoteExecutor(nn.Module):
    def __init__(
        self,
        chord_vocab_size:   int   = 196,
        phrase_vocab_size:  int   = 67,
        artist_vocab_size:  int   = 16,
        pitch_vocab_size:   int   = 132,
        dur_vocab_size:     int   = 19,
        d_model:            int   = 256,
        nhead:              int   = 8,
        num_encoder_layers: int   = 4,
        num_decoder_layers: int   = 4,
        dim_feedforward:    int   = 512,
        dropout:            float = 0.1,
        max_pos_in_phrase:  int   = 16,   # positions 0-15 within a phrase
    ):
        super().__init__()
        self.d_model          = d_model
        self.pitch_vocab_size = pitch_vocab_size
        self.dur_vocab_size   = dur_vocab_size

        # ---- Encoder (chord context) ----
        self.chord_embed  = nn.Embedding(chord_vocab_size,  d_model, padding_idx=0)
        self.artist_embed = nn.Embedding(artist_vocab_size, d_model)

        # ---- Decoder conditioning ----
        # Phrase embedding broadcast-added to EVERY decoder position
        self.phrase_embed = nn.Embedding(phrase_vocab_size, d_model)
        # Learned position-in-phrase (0-15) — tells the model WHERE in the phrase it is
        self.pos_embed    = nn.Embedding(max_pos_in_phrase, d_model)

        # ---- Decoder note embeddings ----
        self.pitch_embed   = nn.Embedding(pitch_vocab_size, d_model, padding_idx=0)
        self.dur_embed     = nn.Embedding(dur_vocab_size,   d_model, padding_idx=0)
        self.is_rest_embed = nn.Embedding(2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.pitch_head   = nn.Linear(d_model, pitch_vocab_size)
        self.dur_head     = nn.Linear(d_model, dur_vocab_size)
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

    def encode(self, chord_ids, artist_id, src_key_padding_mask=None):
        src = self.chord_embed(chord_ids)
        src[:, 0, :] = src[:, 0, :] + self.artist_embed(artist_id)
        src = self._sinusoidal_pe(src)
        return self.encoder(src, src_key_padding_mask=src_key_padding_mask)

    def forward(
        self,
        chord_ids:     torch.Tensor,
        phrase_id:     torch.Tensor,
        artist_id:     torch.Tensor,
        ctx_pitch:     torch.Tensor,
        ctx_dur:       torch.Tensor,
        ctx_rest:      torch.Tensor,
        pos_in_phrase: torch.Tensor,           # (B,)  0-15
        src_key_padding_mask: torch.Tensor = None,
    ):
        memory = self.encode(chord_ids, artist_id, src_key_padding_mask)

        note_emb = (
            self.pitch_embed(ctx_pitch)
            + self.dur_embed(ctx_dur)
            + self.is_rest_embed(ctx_rest)
        )
        # Broadcast phrase token and position-in-phrase to every context position
        phrase_bias = self.phrase_embed(phrase_id).unsqueeze(1)     # (B, 1, D)
        pos_bias    = self.pos_embed(pos_in_phrase).unsqueeze(1)    # (B, 1, D)
        note_emb = note_emb + phrase_bias + pos_bias
        note_emb = self._sinusoidal_pe(note_emb)

        tgt_len  = note_emb.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=ctx_pitch.device)
        out = self.decoder(
            note_emb, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        last = out[:, -1, :]
        return self.pitch_head(last), self.dur_head(last), self.is_rest_head(last)

    @torch.no_grad()
    def generate(
        self,
        chord_ids:   torch.Tensor,
        phrase_id:   torch.Tensor,
        artist_id:   torch.Tensor,
        n_notes:     int  = 16,
        window:      int  = 8,
        prefix_pitch: list = None,   # cross-chord context from previous section
        prefix_dur:   list = None,
        prefix_rest:  list = None,
    ) -> list:
        self.eval()
        device = chord_ids.device
        memory = self.encode(chord_ids, artist_id, src_key_padding_mask=None)

        # Seed history: use cross-chord prefix if provided, else empty
        pitch_hist = list(prefix_pitch or [])
        dur_hist   = list(prefix_dur   or [])
        rest_hist  = list(prefix_rest  or [])

        generated = []
        for pos in range(n_notes):
            # Build exactly `window` context tokens (left-padded with PAD)
            ctx_p = pitch_hist[-window:]
            ctx_d = dur_hist[-window:]
            ctx_r = rest_hist[-window:]
            pad   = window - len(ctx_p)
            ctx_p = [0] * pad + ctx_p
            ctx_d = [0] * pad + ctx_d
            ctx_r = [0] * pad + ctx_r

            cp     = torch.tensor([ctx_p], dtype=torch.long, device=device)
            cd     = torch.tensor([ctx_d], dtype=torch.long, device=device)
            cr     = torch.tensor([ctx_r], dtype=torch.long, device=device)
            pos_id = torch.tensor([min(pos, 15)], dtype=torch.long, device=device)

            note_emb = (
                self.pitch_embed(cp)
                + self.dur_embed(cd)
                + self.is_rest_embed(cr)
            )
            phrase_bias = self.phrase_embed(phrase_id).unsqueeze(1)
            pos_bias    = self.pos_embed(pos_id).unsqueeze(1)
            note_emb = note_emb + phrase_bias + pos_bias
            note_emb = self._sinusoidal_pe(note_emb)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(window, device=device)
            out  = self.decoder(note_emb, memory, tgt_mask=tgt_mask)
            last = out[:, -1, :]

            pitch_probs = torch.softmax(self.pitch_head(last) / 0.9, dim=-1)
            next_pitch  = torch.multinomial(pitch_probs, num_samples=1).item()
            dur_probs   = torch.softmax(self.dur_head(last) / 0.8, dim=-1)
            next_dur    = torch.multinomial(dur_probs, num_samples=1).item()
            next_rest   = self.is_rest_head(last).argmax(-1).item()

            generated.append((next_pitch, next_dur, next_rest))
            pitch_hist.append(next_pitch)
            dur_hist.append(next_dur)
            rest_hist.append(next_rest)

        return generated


if __name__ == "__main__":
    torch.manual_seed(0)
    B, Lc, W = 4, 12, 8
    chord_ids     = torch.randint(4, 196, (B, Lc))
    phrase_id     = torch.randint(3, 67,  (B,))
    artist_id     = torch.randint(1, 16,  (B,))
    ctx_pitch     = torch.randint(4, 132, (B, W))
    ctx_dur       = torch.randint(3, 19,  (B, W))
    ctx_rest      = torch.randint(0, 2,   (B, W))
    pos_in_phrase = torch.randint(0, 16,  (B,))

    model = NoteExecutor()
    model.eval()
    with torch.no_grad():
        p_logits, d_logits, r_logits = model(
            chord_ids, phrase_id, artist_id,
            ctx_pitch, ctx_dur, ctx_rest, pos_in_phrase
        )
    print(f"\nForward pass OK.")
    print(f"  pitch_logits: {p_logits.shape}")
    print(f"  dur_logits:   {d_logits.shape}")
    print(f"  rest_logits:  {r_logits.shape}")
    assert p_logits.shape == (B, 132)
    assert d_logits.shape == (B, 19)
    assert r_logits.shape == (B, 2)
    print("Shape assertions passed.")
