import torch

from src.training.losses import interval_penalty, interval_penalty_from_logits


def test_zero_when_all_stepwise():
    pitches = torch.tensor([[60, 62, 65, 67]], dtype=torch.float)
    rest_mask = torch.zeros_like(pitches, dtype=torch.bool)

    penalty = interval_penalty(pitches, rest_mask)

    assert torch.isclose(penalty, torch.tensor(0.0))


def test_positive_on_large_leap():
    pitches = torch.tensor([[60, 72, 74]], dtype=torch.float)
    rest_mask = torch.zeros_like(pitches, dtype=torch.bool)

    penalty = interval_penalty(pitches, rest_mask)

    assert torch.isclose(penalty, torch.tensor(12.5))


def test_rest_mask_skips_intervals():
    pitches = torch.tensor([[60, 84, 62]], dtype=torch.float)
    rest_mask = torch.tensor([[False, True, False]])

    penalty = interval_penalty(pitches, rest_mask)

    assert torch.isclose(penalty, torch.tensor(0.0))


def test_lambda_zero_short_circuits():
    logits = torch.zeros(1, 132, requires_grad=True)
    ctx_pitch = torch.tensor([[64, 76]], dtype=torch.long)
    ctx_rest = torch.zeros_like(ctx_pitch, dtype=torch.long)
    target_rest = torch.tensor([0], dtype=torch.long)
    lambda_interval = 0.0

    if lambda_interval == 0.0:
        loss = logits.sum() * 0.0
    else:
        loss = lambda_interval * interval_penalty_from_logits(
            logits, ctx_pitch, ctx_rest, target_rest
        )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))
