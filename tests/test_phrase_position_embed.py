from pathlib import Path

import torch
import torch.nn as nn

import src.models.note_executor as note_executor_module
from src.models.note_executor import N_NOTES, NoteExecutor
from src.tokenization import ArtistTokenizer, ChordTokenizer


def _tiny_model():
    return NoteExecutor(
        chord_vocab_size=12,
        phrase_vocab_size=10,
        artist_vocab_size=5,
        pitch_vocab_size=32,
        dur_vocab_size=12,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )


def _batch(window=8):
    pitch = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.long)[:, :window]
    dur = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.long)[:, :window]
    return {
        "chord_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "phrase_id": torch.tensor([2], dtype=torch.long),
        "artist_id": torch.tensor([1], dtype=torch.long),
        "ctx_pitch": pitch,
        "ctx_dur": dur,
        "ctx_rest": torch.zeros(1, window, dtype=torch.long),
        "pos_in_phrase": torch.tensor([5], dtype=torch.long),
        "phrase_position": torch.arange(window, dtype=torch.long).unsqueeze(0),
    }


def test_zero_phrase_position_embedding_is_output_neutral():
    torch.manual_seed(123)
    model = _tiny_model().eval()
    batch = _batch()

    with torch.no_grad():
        explicit = model(**batch)
        varied = dict(batch)
        varied["phrase_position"] = torch.full_like(batch["phrase_position"], N_NOTES - 1)
        changed_positions = model(**varied)
        no_positions = dict(batch)
        no_positions.pop("phrase_position")
        legacy_call = model(**no_positions)

    for left, right in zip(explicit, changed_positions):
        assert torch.allclose(left, right, atol=1e-7)
    for left, right in zip(explicit, legacy_call):
        assert torch.allclose(left, right, atol=1e-7)


def test_v610_checkpoint_loads_with_only_phrase_position_missing_key():
    ckpt = torch.load(Path("checkpoints/v6.1.0_best.pt"), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    chord_tok = ChordTokenizer.from_json()
    artist_tok = ArtistTokenizer()
    dur_vocab_size = state["dur_embed.weight"].shape[0]
    model = NoteExecutor(
        chord_vocab_size=chord_tok.vocab_size,
        artist_vocab_size=artist_tok.vocab_size,
        dur_vocab_size=dur_vocab_size,
        dropout=0.2,
    )

    result = model.load_state_dict(state, strict=False)

    assert set(result.missing_keys) == {"phrase_pos_embed.weight"}
    assert set(result.unexpected_keys) == set()


def test_nonzero_phrase_position_embedding_changes_forward_output():
    torch.manual_seed(123)
    model = _tiny_model().eval()
    batch = _batch()

    with torch.no_grad():
        before = model(**batch)
        model.phrase_pos_embed.weight[batch["phrase_position"][0, -1]].fill_(2.0)
        after = model(**batch)

    assert not torch.allclose(before[0], after[0], atol=1e-7)


class TrackingPhrasePosition(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.calls = []

    def forward(self, positions):
        self.calls.append(positions.detach().cpu().clone())
        return torch.zeros(*positions.shape, self.d_model, device=positions.device)


def test_generate_phrase_positions_are_clamped_and_reset(monkeypatch):
    def argmax_sample(logits, top_p=0.92, temperature=0.95):
        return int(torch.argmax(logits).item())

    monkeypatch.setattr(note_executor_module, "_nucleus_sample", argmax_sample)
    torch.manual_seed(123)
    model = _tiny_model().eval()
    tracker = TrackingPhrasePosition(model.d_model)
    model.phrase_pos_embed = tracker

    chord_ids = torch.tensor([[1]], dtype=torch.long)
    phrase_id = torch.tensor([2], dtype=torch.long)
    artist_id = torch.tensor([1], dtype=torch.long)

    model.generate(chord_ids, phrase_id, artist_id, n_notes=4, window=4)
    first_segment_calls = [call.clone() for call in tracker.calls]
    tracker.calls.clear()
    model.generate(chord_ids, phrase_id, artist_id, n_notes=4, window=4)

    assert len(first_segment_calls) == 4
    assert len(tracker.calls) == 4
    for call in first_segment_calls + tracker.calls:
        assert int(call.min()) >= 0
        assert int(call.max()) <= N_NOTES - 1
    assert torch.equal(first_segment_calls[0], torch.tensor([[0, 0, 0, 0]]))
    assert torch.equal(tracker.calls[0], torch.tensor([[0, 0, 0, 0]]))
