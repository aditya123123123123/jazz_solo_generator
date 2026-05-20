import torch

from src.training.train_v6 import build_chord_tone_tensor, harmonic_penalty


class DummyChordTokenizer:
    def __init__(self):
        self._inv = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>",
            4: "C7",
            5: "Db7",
        }
        self.vocab_size = len(self._inv)


def test_harmonic_penalty_rewards_chord_tone_probability():
    chord_tones = build_chord_tone_tensor(DummyChordTokenizer())
    pos = torch.tensor([0, 0])
    chord_ids = torch.tensor([[4], [4]])
    logits = torch.full((2, 132), -10.0)
    logits[0, 4] = 10.0   # token 4 maps to pitch class 0, a C7 chord tone
    logits[1, 5] = 10.0   # token 5 maps to pitch class 1, not a C7 chord tone

    mixed = harmonic_penalty(logits, pos, chord_ids, chord_tones)
    good = harmonic_penalty(logits[:1], pos[:1], chord_ids[:1], chord_tones)
    bad = harmonic_penalty(logits[1:], pos[1:], chord_ids[1:], chord_tones)

    assert good.item() < 1e-3
    assert bad.item() > 10.0
    assert good.item() < mixed.item() < bad.item()


def test_harmonic_penalty_uses_literal_first_chord_token():
    chord_tones = build_chord_tone_tensor(DummyChordTokenizer())
    pos = torch.tensor([0])
    chord_ids = torch.tensor([[4, 5]])
    logits = torch.full((1, 132), -10.0)
    logits[0, 4] = 10.0

    penalty = harmonic_penalty(logits, pos, chord_ids, chord_tones)

    assert penalty.item() < 1e-3


def test_harmonic_penalty_returns_zero_on_non_anchor_positions_with_grad():
    chord_tones = build_chord_tone_tensor(DummyChordTokenizer())
    pos = torch.tensor([1, 2, 3])
    chord_ids = torch.tensor([[4], [4], [4]])
    logits = torch.randn(3, 132, requires_grad=True)

    penalty = harmonic_penalty(logits, pos, chord_ids, chord_tones)
    penalty.backward()

    assert penalty.item() == 0.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))
