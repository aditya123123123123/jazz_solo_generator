class PhraseTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    N_PHRASES = 64
    OFFSET = 3  # PHRASE_NN -> NN + 3

    def encode(self, phrase_label: str) -> int:
        idx = int(phrase_label.split("_")[1])
        return idx + self.OFFSET

    def decode(self, token: int) -> str:
        idx = token - self.OFFSET
        return f"PHRASE_{idx:02d}"

    @property
    def vocab_size(self) -> int:
        return self.N_PHRASES + self.OFFSET  # 67
