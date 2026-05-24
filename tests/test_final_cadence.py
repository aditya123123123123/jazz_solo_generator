import torch

import src.models.note_executor as note_executor_module
from src.models.note_executor import NoteExecutor


def test_final_cadence_boost_favors_c_major_chord_tone(monkeypatch):
    """Boosted final cadence should land on a Cj7 chord tone for ii-V-I."""

    def argmax_sample(logits, top_p=0.92, temperature=0.95):
        return int(torch.argmax(logits).item())

    monkeypatch.setattr(note_executor_module, "_nucleus_sample", argmax_sample)

    torch.manual_seed(42)
    model = NoteExecutor(
        chord_vocab_size=8,
        phrase_vocab_size=8,
        artist_vocab_size=4,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model.eval()

    # Make the unboosted argmax a non-chord tone and keep all other logits low.
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.pitch_head.bias.fill_(-1.0)
        model.pitch_head.bias[10] = 1.0  # token 10 -> F# pitch class

    chord_ids = torch.tensor([[1]], dtype=torch.long)
    phrase_id = torch.tensor([1], dtype=torch.long)
    artist_id = torch.tensor([1], dtype=torch.long)

    unboosted = model.generate(
        chord_ids,
        phrase_id,
        artist_id,
        n_notes=4,
        final_note_chord_tone_boost=0.0,
        is_final_segment=True,
        active_chord_symbol="Cj7",
    )
    boosted = model.generate(
        chord_ids,
        phrase_id,
        artist_id,
        n_notes=4,
        final_note_chord_tone_boost=5.0,
        is_final_segment=True,
        active_chord_symbol="Cj7",
    )

    assert (unboosted[-1][0] - 4) % 12 == 6
    assert (boosted[-1][0] - 4) % 12 in {0, 4, 7, 11}
    assert boosted[-1][2] == 0


def test_chord_tone_bias_can_target_strong_beats_only(monkeypatch):
    """General chord-tone bias should steer phrase anchors without touching weak positions."""

    sampled_pcs = []

    def record_argmax(logits, top_p=0.92, temperature=0.95):
        token = int(torch.argmax(logits).item())
        sampled_pcs.append((token - 4) % 12)
        return token

    monkeypatch.setattr(note_executor_module, "_nucleus_sample", record_argmax)

    torch.manual_seed(42)
    model = NoteExecutor(
        chord_vocab_size=8,
        phrase_vocab_size=8,
        artist_vocab_size=4,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.pitch_head.bias.fill_(-2.0)
        model.pitch_head.bias[10] = 1.0   # F# non-chord tone
        model.pitch_head.bias[4] = 0.0    # C chord tone, needs boost to win

    chord_ids = torch.tensor([[1]], dtype=torch.long)
    phrase_id = torch.tensor([1], dtype=torch.long)
    artist_id = torch.tensor([1], dtype=torch.long)

    generated = model.generate(
        chord_ids,
        phrase_id,
        artist_id,
        n_notes=6,
        chord_tone_bias=True,
        chord_tone_bias_strength=2.0,
        non_chord_penalty=0.0,
        strong_beat_only=True,
        final_note_chord_tone_boost=0.0,
        active_chord_symbol="Cj7",
    )

    pcs = [((pitch - 4) % 12) for pitch, _dur, _rest in generated]
    assert pcs[0] in {0, 4, 7, 11}
    assert pcs[4] in {0, 4, 7, 11}
    assert pcs[1] == 6
    assert pcs[2] == 6
    assert pcs[3] == 6
    assert pcs[5] == 6
