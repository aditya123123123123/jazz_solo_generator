class ArtistTokenizer:
    PAD = 0
    _VOCAB = {
        "Charlie Parker": 1,
        "Miles Davis": 2,
        "Dizzy Gillespie": 3,
        "Sonny Rollins": 4,
        "John Coltrane": 5,
        "Clifford Brown": 6,
        "Dexter Gordon": 7,
        "Hank Mobley": 8,
        "Lee Morgan": 9,
        "Kenny Dorham": 10,
        "Sonny Stitt": 11,
        "Cannonball Adderley": 12,
        "Fats Navarro": 13,
        "Johnny Hodges": 14,
        "Lester Young": 15,
    }
    _INV = {v: k for k, v in _VOCAB.items()}

    def encode(self, name: str) -> int:
        return self._VOCAB.get(name, self.PAD)

    def decode(self, token: int) -> str:
        return self._INV.get(token, "<PAD>")

    @property
    def vocab_size(self) -> int:
        return len(self._VOCAB) + 1  # 16 (PAD + 15 artists)
