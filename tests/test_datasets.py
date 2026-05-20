import torch
import pytest

from src.tokenization import ArtistTokenizer, ChordTokenizer, NoteTokenizer, PhraseTokenizer
from src.data.datasets import PhrasePlannerDataset, NoteExecutorDataset
from src.data.collate import collate_phrase_planner, collate_note_executor


@pytest.fixture(scope="module")
def tokenizers():
    return (
        ChordTokenizer.from_json(),
        NoteTokenizer.from_json(),
        ArtistTokenizer(),
        PhraseTokenizer(),
    )


@pytest.fixture(scope="module")
def planner_dataset(tokenizers):
    chord_tok, note_tok, artist_tok, phrase_tok = tokenizers
    return PhrasePlannerDataset(chord_tok, note_tok, artist_tok, phrase_tok)


@pytest.fixture(scope="module")
def executor_dataset(tokenizers):
    chord_tok, note_tok, artist_tok, phrase_tok = tokenizers
    return NoteExecutorDataset(chord_tok, note_tok, artist_tok, phrase_tok)


# --- Length tests ---

def test_phrase_planner_len(planner_dataset):
    assert len(planner_dataset) == 411


def test_note_executor_len(executor_dataset):
    assert len(executor_dataset) == 10163


# --- Key and dtype tests ---

def test_phrase_planner_sample_keys(planner_dataset):
    sample = planner_dataset[0]
    expected_keys = {"chord_ids", "artist_id", "phrase_ids", "chord_len", "phrase_len"}
    assert set(sample.keys()) == expected_keys
    for k, v in sample.items():
        assert isinstance(v, torch.Tensor), f"{k} is not a tensor"
        assert v.dtype == torch.long, f"{k} dtype is {v.dtype}, expected torch.long"


def test_note_executor_sample_keys(executor_dataset):
    sample = executor_dataset[0]
    expected_keys = {"phrase_id", "artist_id", "chord_ids", "pitch_ids", "dur_ids", "chord_len", "note_len"}
    assert set(sample.keys()) == expected_keys
    for k, v in sample.items():
        assert isinstance(v, torch.Tensor), f"{k} is not a tensor"
        assert v.dtype == torch.long, f"{k} dtype is {v.dtype}, expected torch.long"


def test_note_executor_pitch_dur_same_len(executor_dataset):
    for i in range(min(10, len(executor_dataset))):
        sample = executor_dataset[i]
        assert sample["pitch_ids"].shape == sample["dur_ids"].shape, (
            f"Sample {i}: pitch_ids {sample['pitch_ids'].shape} != dur_ids {sample['dur_ids'].shape}"
        )


# --- Collate tests ---

def test_collate_phrase_planner(planner_dataset):
    batch = [planner_dataset[i] for i in range(4)]
    out = collate_phrase_planner(batch)
    assert out["chord_ids"].shape[0] == 4
    assert out["phrase_ids"].ndim == 2
    assert out["chord_mask"].dtype == torch.bool
    assert out["phrase_mask"].dtype == torch.bool
    assert out["artist_id"].shape == (4,)
    assert out["chord_ids"].dtype == torch.long
    assert out["phrase_ids"].dtype == torch.long


def test_collate_note_executor(executor_dataset):
    batch = [executor_dataset[i] for i in range(4)]
    out = collate_note_executor(batch)
    assert out["pitch_ids"].shape == out["dur_ids"].shape
    assert out["chord_mask"].dtype == torch.bool
    assert out["pitch_mask"].dtype == torch.bool
    assert out["chord_ids"].shape[0] == 4
    assert out["pitch_ids"].dtype == torch.long
    assert out["dur_ids"].dtype == torch.long
