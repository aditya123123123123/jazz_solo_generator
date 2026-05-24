import pytest
import torch

from src.training.train_v6 import build_chord_tone_tensor, harmonic_penalty


class DummyChordTokenizer:
    def __init__(self):
        self._inv = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>",
            4: "",
            5: "C7",
            6: "Db7",
        }
        self.vocab_size = len(self._inv)


def test_harmonic_penalty_rewards_chord_tone_probability():
    chord_tones = build_chord_tone_tensor(DummyChordTokenizer())
    pos = torch.tensor([0, 0])
    target_chord_id = torch.tensor([5, 5])
    logits = torch.full((2, 132), -10.0)
    logits[0, 4] = 10.0   # token 4 maps to pitch class 0, a C7 chord tone
    logits[1, 5] = 10.0   # token 5 maps to pitch class 1, not a C7 chord tone

    mixed = harmonic_penalty(logits, pos, target_chord_id, chord_tones)
    good = harmonic_penalty(logits[:1], pos[:1], target_chord_id[:1], chord_tones)
    bad = harmonic_penalty(logits[1:], pos[1:], target_chord_id[1:], chord_tones)

    assert good.item() < 1e-3
    assert bad.item() > 10.0
    assert good.item() < mixed.item() < bad.item()


def test_harmonic_penalty_uses_per_sample_target_chord_id():
    chord_tones = build_chord_tone_tensor(DummyChordTokenizer())
    pos = torch.tensor([0, 0])
    target_chord_id = torch.tensor([5, 6])  # C7, Db7
    logits = torch.full((2, 132), -10.0)
    logits[0, 4] = 10.0  # C pitch class: good for C7
    logits[1, 5] = 10.0  # Db pitch class: good for Db7

    penalty = harmonic_penalty(logits, pos, target_chord_id, chord_tones)

    assert penalty.item() < 1e-3


def test_harmonic_penalty_rejects_sequence_chord_ids():
    chord_tones = build_chord_tone_tensor(DummyChordTokenizer())
    pos = torch.tensor([0])
    sequence_chord_ids = torch.tensor([[5, 6]])
    logits = torch.full((1, 132), -10.0)

    with pytest.raises(ValueError, match="per-sample 1D tensor"):
        harmonic_penalty(logits, pos, sequence_chord_ids, chord_tones)


def test_harmonic_penalty_returns_zero_on_non_anchor_positions_with_grad():
    chord_tones = build_chord_tone_tensor(DummyChordTokenizer())
    pos = torch.tensor([1, 2, 3])
    target_chord_id = torch.tensor([5, 5, 5])
    logits = torch.randn(3, 132, requires_grad=True)

    penalty = harmonic_penalty(logits, pos, target_chord_id, chord_tones)
    penalty.backward()

    assert penalty.item() == 0.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_harmonic_penalty_masks_special_chords_with_grad():
    chord_tones = build_chord_tone_tensor(DummyChordTokenizer())
    pos = torch.tensor([0, 4])
    target_chord_id = torch.tensor([1, 4])
    logits = torch.randn(2, 132, requires_grad=True)

    penalty = harmonic_penalty(logits, pos, target_chord_id, chord_tones)
    penalty.backward()

    assert penalty.item() == 0.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))
