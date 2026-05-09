class ArtistTokenizer:
    PAD = 0
    _VOCAB = {
        "Charlie Parker": 1,
        "Miles Davis": 2,
    }
    _INV = {v: k for k, v in _VOCAB.items()}

    def encode(self, name: str) -> int:
        return self._VOCAB.get(name, self.PAD)

    def decode(self, token: int) -> str:
        return self._INV.get(token, "<PAD>")

    @property
    def vocab_size(self) -> int:
        return len(self._VOCAB) + 1  # 3 (PAD + 2 artists)
