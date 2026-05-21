"""Shared training loss helpers."""
from __future__ import annotations

import torch
import torch.nn.functional as F

PITCH_TOKEN_OFFSET = 4


def interval_penalty(
    pred_pitches: torch.Tensor,
    rest_mask: torch.Tensor,
    free_interval: int = 7,
) -> torch.Tensor:
    """Penalize adjacent non-rest intervals larger than `free_interval`.

    `pred_pitches` may contain decoded MIDI pitches or pitch token IDs. If all
    non-rest values are valid pitch-token IDs (>=4), they are decoded by
    subtracting the v6 pitch offset. Rest/special-token positions are ignored
    through `rest_mask`, so their numeric values do not matter.
    """
    if pred_pitches.ndim != 2 or rest_mask.ndim != 2:
        raise ValueError("pred_pitches and rest_mask must both have shape (B, T)")
    if pred_pitches.shape != rest_mask.shape:
        raise ValueError("pred_pitches and rest_mask must have identical shapes")
    if pred_pitches.size(1) < 2:
        return pred_pitches.sum() * 0.0

    pitches = pred_pitches.float()
    active = ~rest_mask.bool()
    non_rest = pitches[active]
    if non_rest.numel() and torch.all(non_rest >= PITCH_TOKEN_OFFSET):
        pitches = pitches - PITCH_TOKEN_OFFSET

    adjacent_active = active[:, 1:] & active[:, :-1]
    if not adjacent_active.any():
        return pitches.sum() * 0.0

    delta = (pitches[:, 1:] - pitches[:, :-1]).abs()
    excess = F.relu(delta - float(free_interval)).pow(2)
    return excess[adjacent_active].mean()


def expected_midi_pitch(pitch_logits: torch.Tensor) -> torch.Tensor:
    """Differentiable expected MIDI pitch from v6 pitch logits.

    Special pitch tokens are masked out before softmax. This is used instead
    of argmax in training so the interval term can shape pitch probabilities.
    """
    logits = pitch_logits.float().clone()
    logits[:, :PITCH_TOKEN_OFFSET] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    midi_values = torch.arange(
        logits.size(1),
        dtype=probs.dtype,
        device=probs.device,
    ) - PITCH_TOKEN_OFFSET
    midi_values[:PITCH_TOKEN_OFFSET] = 0
    return (probs * midi_values).sum(dim=-1)


def interval_penalty_from_logits(
    pitch_logits: torch.Tensor,
    ctx_pitch: torch.Tensor,
    ctx_rest: torch.Tensor,
    target_rest: torch.Tensor,
    free_interval: int = 7,
) -> torch.Tensor:
    """Differentiable interval penalty for windowed NoteExecutor training.

    The dataset exposes one target note per row. To approximate a phrase-local
    sequence, append the expected predicted MIDI pitch for that target to the
    previous context pitches. Context pitches/rest flags are fixed data; the
    appended expected pitch remains differentiable with respect to logits.
    """
    expected_pitch = expected_midi_pitch(pitch_logits).unsqueeze(1)
    ctx_midi = (ctx_pitch.float() - PITCH_TOKEN_OFFSET).clamp_min(0.0)
    sequence = torch.cat([ctx_midi, expected_pitch], dim=1)
    rest_sequence = torch.cat([ctx_rest.bool(), target_rest.bool().unsqueeze(1)], dim=1)
    return interval_penalty(sequence, rest_sequence, free_interval=free_interval)
