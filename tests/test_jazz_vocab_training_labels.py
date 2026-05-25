import torch

from src.data.jazz_vocab_labels import compute_jazz_vocab_labels
from src.data.note_dataset import collate_note_window
from src.training.train_v6 import vocabulary_weighted_ce


def test_compute_jazz_vocab_labels_marks_target_note_devices():
    notes = [
        {"pitch": 61, "onset": 0.5, "duration": 0.25, "bar": 0, "beat": 1, "chord": "C7"},
        {"pitch": 60, "onset": 1.0, "duration": 0.25, "bar": 0, "beat": 2, "chord": ""},
        {"pitch": 66, "onset": 1.5, "duration": 0.25, "bar": 0, "beat": 2, "chord": ""},
        {"pitch": 62, "onset": 1.75, "duration": 0.25, "bar": 0, "beat": 2, "chord": ""},
        {"pitch": 64, "onset": 2.0, "duration": 0.25, "bar": 0, "beat": 3, "chord": ""},
    ]

    labels = compute_jazz_vocab_labels(notes)

    assert labels[1]["is_chromatic_approach_target"] == 1
    assert labels[4]["is_enclosure_target"] == 1
    assert labels[4]["is_guide_tone"] == 1
    assert labels[2]["is_dominant_blues_color"] == 1


def test_collate_note_window_preserves_optional_jazz_vocab_labels():
    def sample(value):
        return {
            "chord_ids": torch.tensor([5, 6]),
            "chord_len": torch.tensor(2),
            "phrase_id": torch.tensor(1),
            "artist_id": torch.tensor(1),
            "ctx_pitch": torch.zeros(8, dtype=torch.long),
            "ctx_dur": torch.zeros(8, dtype=torch.long),
            "ctx_rest": torch.zeros(8, dtype=torch.long),
            "phrase_position": torch.zeros(8, dtype=torch.long),
            "target_pitch": torch.tensor(64),
            "target_dur": torch.tensor(5),
            "target_rest": torch.tensor(0),
            "target_chord_id": torch.tensor(5),
            "pos_in_phrase": torch.tensor(1),
            "jazz_vocab_label": torch.tensor(value),
            "chromatic_approach_label": torch.tensor(value & 1),
            "enclosure_label": torch.tensor((value >> 1) & 1),
            "guide_tone_label": torch.tensor((value >> 2) & 1),
            "blues_color_label": torch.tensor((value >> 3) & 1),
        }

    batch = collate_note_window([sample(1), sample(12)])

    assert batch["jazz_vocab_label"].tolist() == [1, 12]
    assert batch["chromatic_approach_label"].tolist() == [1, 0]
    assert batch["guide_tone_label"].tolist() == [0, 1]
    assert batch["blues_color_label"].tolist() == [0, 1]


def test_vocabulary_weighted_ce_upweights_labeled_examples():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    target = torch.tensor([0, 0])
    labels = torch.tensor([0, 1])

    unweighted = vocabulary_weighted_ce(logits, target, labels, weight=1.0)
    weighted = vocabulary_weighted_ce(logits, target, labels, weight=3.0)

    assert weighted > unweighted
