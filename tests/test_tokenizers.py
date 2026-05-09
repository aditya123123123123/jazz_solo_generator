import json
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
PHRASES_PATH = REPO / "data" / "processed" / "phrases.json"
CHORD_VOCAB_PATH = REPO / "data" / "processed" / "chord_vocab.json"
DURATION_BINS_PATH = REPO / "data" / "processed" / "duration_bins.json"


@pytest.fixture(scope="module")
def phrases():
    with open(PHRASES_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def chord_tok():
    from src.tokenization.chord_tokenizer import ChordTokenizer
    return ChordTokenizer.from_json(CHORD_VOCAB_PATH)


@pytest.fixture(scope="module")
def note_tok():
    from src.tokenization.note_tokenizer import NoteTokenizer
    return NoteTokenizer.from_json(DURATION_BINS_PATH)


@pytest.fixture(scope="module")
def artist_tok():
    from src.tokenization.artist_tokenizer import ArtistTokenizer
    return ArtistTokenizer()


@pytest.fixture(scope="module")
def phrase_tok():
    from src.tokenization.phrase_tokenizer import PhraseTokenizer
    return PhraseTokenizer()


# --- Chord tokenizer ---

def test_chord_roundtrip(chord_tok, phrases):
    seen = []
    for p in phrases:
        for n in p["notes"]:
            if n["chord"] not in seen:
                seen.append(n["chord"])
            if len(seen) == 10:
                break
        if len(seen) == 10:
            break
    assert len(seen) == 10
    for chord in seen:
        token = chord_tok.encode(chord)
        assert chord_tok.decode(token) == chord, (
            f"Chord round-trip failed: {chord!r} -> {token} -> {chord_tok.decode(token)!r}"
        )


def test_chord_unk(chord_tok):
    unknown = "ZZZNOT_A_CHORD"
    assert chord_tok.encode(unknown) == chord_tok.UNK
    assert chord_tok.decode(chord_tok.UNK) == "<UNK>"


def test_chord_empty_string(chord_tok):
    token = chord_tok.encode("")
    assert token != chord_tok.UNK, "Empty string should have its own token, not UNK"
    assert chord_tok.decode(token) == ""


# --- Note tokenizer ---

def test_pitch_roundtrip(note_tok, phrases):
    pitches = []
    for p in phrases:
        for n in p["notes"]:
            if n["pitch"] not in pitches:
                pitches.append(n["pitch"])
            if len(pitches) == 10:
                break
        if len(pitches) == 10:
            break
    for pitch in pitches:
        token = note_tok.encode_pitch(pitch)
        assert note_tok.decode_pitch(token) == pitch, f"Pitch round-trip failed: {pitch}"


def test_duration_bin_roundtrip(note_tok, phrases):
    durations = []
    for p in phrases:
        for n in p["notes"]:
            durations.append(n["duration"])
            if len(durations) == 10:
                break
        if len(durations) == 10:
            break
    for dur in durations:
        token = note_tok.encode_duration(dur)
        decoded = note_tok.decode_duration(token)
        re_token = note_tok.encode_duration(decoded)
        assert re_token == token, (
            f"Duration bin round-trip failed: {dur:.6f}s -> token {token} "
            f"-> {decoded:.6f}s -> token {re_token}"
        )


# --- Artist tokenizer ---

def test_artist_roundtrip(artist_tok):
    for name in ["Charlie Parker", "Miles Davis"]:
        token = artist_tok.encode(name)
        assert artist_tok.decode(token) == name, f"Artist round-trip failed: {name!r}"


def test_artist_unknown(artist_tok):
    assert artist_tok.encode("Unknown Artist") == artist_tok.PAD


# --- Phrase tokenizer ---

def test_phrase_roundtrip(phrase_tok):
    for i in range(64):
        label = f"PHRASE_{i:02d}"
        token = phrase_tok.encode(label)
        decoded = phrase_tok.decode(token)
        assert decoded == label, (
            f"Phrase round-trip failed: {label!r} -> {token} -> {decoded!r}"
        )


def test_phrase_special_tokens_distinct(phrase_tok):
    phrase_tokens = {phrase_tok.encode(f"PHRASE_{i:02d}") for i in range(64)}
    assert phrase_tok.PAD not in phrase_tokens
    assert phrase_tok.BOS not in phrase_tokens
    assert phrase_tok.EOS not in phrase_tokens


def test_phrase_vocab_size(phrase_tok):
    assert phrase_tok.vocab_size == 67
