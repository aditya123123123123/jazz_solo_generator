import json
from pathlib import Path

PHRASES_PATH = Path(__file__).parents[3] / "data" / "processed" / "phrases.json"
VOCAB_PATH = Path(__file__).parents[3] / "data" / "processed" / "chord_vocab.json"

# Relative path anchors: this file is at src/tokenization/chord_tokenizer.py
# parents[0] = src/tokenization, parents[1] = src, parents[2] = jazz_solo_generator (repo root)
_REPO = Path(__file__).parents[2]
PHRASES_PATH = _REPO / "data" / "processed" / "phrases.json"
VOCAB_PATH = _REPO / "data" / "processed" / "chord_vocab.json"


class ChordTokenizer:
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    _OFFSET = 4

    def __init__(self, vocab: dict):
        self._vocab = vocab  # chord_str -> int
        self._inv = {v: k for k, v in vocab.items()}

    @classmethod
    def from_phrases(cls, phrases_path=PHRASES_PATH):
        with open(phrases_path) as f:
            phrases = json.load(f)
        chords = sorted({n["chord"] for p in phrases for n in p["notes"]})
        vocab = {chord: i + cls._OFFSET for i, chord in enumerate(chords)}
        vocab["<PAD>"] = cls.PAD
        vocab["<UNK>"] = cls.UNK
        vocab["<BOS>"] = cls.BOS
        vocab["<EOS>"] = cls.EOS
        return cls(vocab)

    @classmethod
    def from_json(cls, vocab_path=VOCAB_PATH):
        with open(vocab_path) as f:
            vocab = json.load(f)
        return cls(vocab)

    def save(self, path=VOCAB_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._vocab, f, indent=2)

    def encode(self, chord_str: str) -> int:
        return self._vocab.get(chord_str, self.UNK)

    def decode(self, token_id: int) -> str:
        return self._inv.get(token_id, "<UNK>")

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


if __name__ == "__main__":
    tok = ChordTokenizer.from_phrases()
    tok.save()
    print(f"Chord vocab size: {tok.vocab_size}")
    print(f"Saved to {VOCAB_PATH}")
