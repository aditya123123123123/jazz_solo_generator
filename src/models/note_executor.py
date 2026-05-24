import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.generation.chord_utils import chord_tones, parse_chord

N_NOTES = 16


def _nucleus_sample(logits: torch.Tensor, top_p: float = 0.92, temperature: float = 0.95) -> int:
    """Top-p (nucleus) sampling with temperature."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = (cumsum - sorted_probs) > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    return sorted_idx[torch.multinomial(sorted_probs, 1)].item()

# The WJazzD corpus has no per-note velocity.
# This model predicts pitch, quantised duration, and is_rest (gap flag).


class NoteExecutor(nn.Module):
    def __init__(
        self,
        chord_vocab_size:   int   = 196,
        phrase_vocab_size:  int   = 67,
        artist_vocab_size:  int   = 16,
        pitch_vocab_size:   int   = 132,
        dur_vocab_size:     int   = 18,
        d_model:            int   = 256,
        nhead:              int   = 8,
        num_encoder_layers: int   = 4,
        num_decoder_layers: int   = 4,
        dim_feedforward:    int   = 512,
        dropout:            float = 0.1,
        max_pos_in_phrase:  int   = N_NOTES,   # positions 0-15 within a phrase
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
        # Context-token phrase positions. Zero initialization preserves legacy
        # v6.1.0 behavior until the next finetune learns this table.
        with torch.random.fork_rng(devices=[]):
            self.phrase_pos_embed = nn.Embedding(N_NOTES, d_model)
        nn.init.zeros_(self.phrase_pos_embed.weight)

        # ---- Decoder note embeddings ----
        self.pitch_embed   = nn.Embedding(pitch_vocab_size, d_model, padding_idx=0)
        self.dur_embed     = nn.Embedding(dur_vocab_size,   d_model, padding_idx=0)
        self.is_rest_embed = nn.Embedding(2, d_model)

        # Tempo conditioning: scalar BPM → d_model bias added to every
        # decoder position. Normalized by /200 (≈ corpus median tempo)
        # so the input stays in a small range.
        self.tempo_embed   = nn.Linear(1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

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
        phrase_position: Optional[torch.Tensor] = None,  # (B, W) context positions
        tempo_bpm:     Optional[torch.Tensor] = None,   # (B,) or (B,1) float
        src_key_padding_mask: torch.Tensor = None,
    ):
        memory = self.encode(chord_ids, artist_id, src_key_padding_mask)

        note_emb = (
            self.pitch_embed(ctx_pitch)
            + self.dur_embed(ctx_dur)
            + self.is_rest_embed(ctx_rest)
        )
        if phrase_position is None:
            phrase_position = pos_in_phrase.unsqueeze(1).expand_as(ctx_pitch)
        phrase_position = phrase_position.clamp(0, N_NOTES - 1)
        note_emb = note_emb + self.phrase_pos_embed(phrase_position)
        # Broadcast phrase token and position-in-phrase to every context position
        phrase_bias = self.phrase_embed(phrase_id).unsqueeze(1)     # (B, 1, D)
        pos_bias    = self.pos_embed(pos_in_phrase).unsqueeze(1)    # (B, 1, D)
        note_emb = note_emb + phrase_bias + pos_bias
        if tempo_bpm is not None:
            if tempo_bpm.dim() == 1:
                tempo_bpm = tempo_bpm.unsqueeze(-1)                  # (B, 1)
            tempo_bias = self.tempo_embed(tempo_bpm / 200.0).unsqueeze(1)  # (B, 1, D)
            note_emb = note_emb + tempo_bias
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
        n_notes:     int   = 16,
        window:      int   = 8,
        tempo_bpm:   float = 180.0,
        prefix_pitch: list = None,   # cross-chord context from previous section
        prefix_dur:   list = None,
        prefix_rest:  list = None,
        temperature:           float = 1.0,
        duration_temperature:  float = 1.0,
        rest_boost:            float = 0.0,
        final_note_chord_tone_boost: float = 3.0,
        chord_tone_bias:       bool = False,
        chord_tone_bias_strength: float = 2.0,
        non_chord_penalty:     float = 0.0,
        strong_beat_only:      bool = True,
        is_final_segment:      bool = False,
        active_chord_symbol:   Optional[str] = None,
    ) -> list:
        """Autoregressive sampling loop.

        `temperature` scales pitch sampling (multiplied by 0.95 inside the
        nucleus sampler). `duration_temperature` scales duration sampling
        independently — the prior code used `dur_logits / (0.8 * temperature)`
        (effective temperature 0.64 with the default temperature=0.8). The
        new code uses `dur_logits / duration_temperature` directly. With the
        executor-side default duration_temperature=1.0, callers that do not
        pass duration_temperature get effective temperature 1.0 instead of
        0.64 — a deliberate behavior change to address duration mode-collapse.

        `rest_boost` is added to the "rest" class logit of `is_rest_head`
        before argmax. With rest_boost=0 behavior is identical to before;
        positive values tip argmax toward rest in close-margin cases.

        `final_note_chord_tone_boost` is an inference-only cadence helper.
        When this is the final chord segment, it nudges the last sampled pitch
        toward chord-tone pitch classes of `active_chord_symbol` and keeps that
        final token sounding so the cadence is audible.

        `chord_tone_bias` is an inference-time harmonic steering control. When
        enabled, it adds `chord_tone_bias_strength` to pitch logits whose pitch
        class is a chord tone of `active_chord_symbol`. If `non_chord_penalty`
        is positive, it subtracts that amount from non-chord-tone pitch logits.
        With `strong_beat_only=True`, the steering applies only to phrase
        anchors 0, 4, 8, and 12, matching the training harmonic-loss target.
        """
        self.eval()
        device = chord_ids.device
        memory = self.encode(chord_ids, artist_id, src_key_padding_mask=None)

        cadence_tones = None
        if active_chord_symbol:
            parsed = parse_chord(active_chord_symbol)
            if parsed is not None:
                cadence_tones = set(chord_tones(*parsed))

        # Seed history: use cross-chord prefix if provided, else empty
        pitch_hist = list(prefix_pitch or [])
        dur_hist   = list(prefix_dur   or [])
        rest_hist  = list(prefix_rest  or [])

        # Pre-compute the tempo bias once — it doesn't change across the loop.
        tempo_t    = torch.tensor([[tempo_bpm / 200.0]], dtype=torch.float, device=device)
        tempo_bias = self.tempo_embed(tempo_t).unsqueeze(1)   # (1, 1, D)

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
            phrase_position = torch.arange(
                pos - window, pos, dtype=torch.long, device=device
            ).clamp(0, N_NOTES - 1).unsqueeze(0)

            note_emb = (
                self.pitch_embed(cp)
                + self.dur_embed(cd)
                + self.is_rest_embed(cr)
                + self.phrase_pos_embed(phrase_position)
            )
            phrase_bias = self.phrase_embed(phrase_id).unsqueeze(1)
            pos_bias    = self.pos_embed(pos_id).unsqueeze(1)
            note_emb = note_emb + phrase_bias + pos_bias + tempo_bias
            note_emb = self._sinusoidal_pe(note_emb)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(window, device=device)
            out  = self.decoder(note_emb, memory, tgt_mask=tgt_mask)
            last = out[:, -1, :]

            # Repetition penalty on raw logits before sampling
            pitch_logits = self.pitch_head(last).squeeze(0).clone()  # (pitch_vocab,)
            pitch_logits[:4] = float('-inf')  # mask PAD, BOS, EOS, REST
            recent_pcs = [p % 12 for p in pitch_hist[-3:] if p >= 4]
            for rp in pitch_hist[-3:]:
                if 4 <= rp < pitch_logits.size(0):
                    pitch_logits[rp] = pitch_logits[rp] * 0.4
            for idx in range(4, pitch_logits.size(0)):
                if (idx - 4) % 12 in recent_pcs:
                    pitch_logits[idx] = pitch_logits[idx] * 0.7
            if chord_tone_bias and cadence_tones and (not strong_beat_only or pos in {0, 4, 8, 12}):
                for idx in range(4, pitch_logits.size(0)):
                    if (idx - 4) % 12 in cadence_tones:
                        pitch_logits[idx] = pitch_logits[idx] + chord_tone_bias_strength
                    elif non_chord_penalty > 0.0:
                        pitch_logits[idx] = pitch_logits[idx] - non_chord_penalty

            if (
                is_final_segment
                and pos == n_notes - 1
                and final_note_chord_tone_boost > 0.0
                and cadence_tones
            ):
                for idx in range(4, pitch_logits.size(0)):
                    if (idx - 4) % 12 in cadence_tones:
                        pitch_logits[idx] = pitch_logits[idx] + final_note_chord_tone_boost

            # Nucleus sampling for pitch; temperature sampling for duration
            next_pitch = _nucleus_sample(pitch_logits, top_p=0.92, temperature=0.95 * temperature)
            dur_logits = self.dur_head(last).squeeze(0).clone()
            dur_logits[:4] = float('-inf')  # mask PAD, BOS, EOS, REST (DUR_OFFSET=4)
            dur_probs  = torch.softmax(dur_logits / duration_temperature, dim=-1)
            next_dur   = torch.multinomial(dur_probs, num_samples=1).item()

            is_rest_logits = self.is_rest_head(last).squeeze(0).clone()
            is_rest_logits[1] += rest_boost   # boost "rest" class
            if is_final_segment and pos == n_notes - 1 and final_note_chord_tone_boost > 0.0:
                is_rest_logits[0] = is_rest_logits[0] + final_note_chord_tone_boost
                is_rest_logits[1] = float('-inf')
            next_rest  = is_rest_logits.argmax(-1).item()

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
    ctx_dur       = torch.randint(3, 18,  (B, W))
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
    assert d_logits.shape == (B, 18)
    assert r_logits.shape == (B, 2)
    print("Shape assertions passed.")
