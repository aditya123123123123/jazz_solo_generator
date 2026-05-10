import math

import torch
import torch.nn as nn


class PhrasePlanner(nn.Module):
    def __init__(
        self,
        chord_vocab_size: int = 107,
        artist_vocab_size: int = 16,
        phrase_vocab_size: int = 67,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.phrase_vocab_size = phrase_vocab_size

        self.chord_embed = nn.Embedding(chord_vocab_size, d_model, padding_idx=0)
        self.artist_embed = nn.Embedding(artist_vocab_size, d_model)
        self.phrase_embed = nn.Embedding(phrase_vocab_size, d_model, padding_idx=0)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.phrase_head = nn.Linear(d_model, phrase_vocab_size)

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
        src[:, 0, :] = src[:, 0, :] + self.artist_embed(artist_id)
        src = self._sinusoidal_pe(src)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return memory

    def forward(
        self,
        chord_ids: torch.Tensor,
        artist_id: torch.Tensor,
        phrase_ids: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        memory = self.encode(chord_ids, artist_id, src_key_padding_mask)

        tgt = self.phrase_embed(phrase_ids)
        tgt = self._sinusoidal_pe(tgt)

        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                phrase_ids.size(1), device=phrase_ids.device
            )

        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.phrase_head(out)
        return logits

    @torch.no_grad()
    def generate(
        self,
        chord_ids: torch.Tensor,
        artist_id: torch.Tensor,
        max_len: int = 32,
        bos: int = 1,
        eos: int = 2,
    ) -> list:
        self.eval()
        device = chord_ids.device
        memory = self.encode(chord_ids, artist_id, src_key_padding_mask=None)

        generated = [bos]
        for _ in range(max_len):
            tgt = torch.tensor([generated], dtype=torch.long, device=device)
            tgt_emb = self.phrase_embed(tgt)
            tgt_emb = self._sinusoidal_pe(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), device=device
            )
            out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            next_token = self.phrase_head(out[:, -1, :]).argmax(-1).item()
            generated.append(next_token)
            if next_token == eos:
                break

        return [t for t in generated if 3 <= t < 67]


if __name__ == "__main__":
    torch.manual_seed(0)
    chord_ids = torch.randint(4, 107, (4, 100))
    phrase_ids = torch.randint(3, 67, (4, 30))
    artist_id = torch.randint(1, 3, (4,))

    model = PhrasePlanner()
    model.eval()
    with torch.no_grad():
        logits = model(chord_ids, artist_id, phrase_ids)

    print(f"\nForward pass OK. Output shape: {logits.shape}")
    assert logits.shape == torch.Size([4, 30, 67]), f"Unexpected shape: {logits.shape}"
    print("Shape assertion passed.")
