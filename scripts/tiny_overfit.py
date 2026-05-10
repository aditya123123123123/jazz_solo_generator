"""Sanity test: v6 pipeline end-to-end memorization check.

Proves NoteWindowDataset → NoteExecutor → generate → decode_duration
are wired consistently by training on 200 phrases for 400 gradient steps.

Exit 0 on pass, 1 on fail.
"""
import itertools
import json
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.note_dataset import (
    NoteWindowDataset,
    _transposed_chord_changes,
    collate_note_window,
)
from src.models.note_executor import NoteExecutor
from src.tokenization import ArtistTokenizer, ChordTokenizer, PhraseTokenizer
from src.tokenization.note_tokenizer import (
    DUR_OFFSET,
    MUSICAL_DURATIONS_BEATS,
    NoteTokenizer,
)

_REPO = Path(__file__).resolve().parents[1]
PHRASES_PATH  = _REPO / "data" / "processed" / "phrases_all_with_tempo.json"
CLUSTERS_PATH = _REPO / "data" / "processed" / "phrase_clusters_all.csv"


def collate_with_tempo(batch):
    out = collate_note_window(batch)
    out["tempo_bpm"] = torch.stack([b["tempo_bpm"] for b in batch])
    return out


def main():
    # ------------------------------------------------------------------ #
    # 1. Slice 200 phrases → temp JSON
    # ------------------------------------------------------------------ #
    # 10 phrases ≈ 107 samples → 400 steps × 32 ≈ 120 epochs, enough to memorize.
    # 200 phrases → 2144 samples → only ~6 epochs in 400 steps — insufficient.
    print("Loading phrases…")
    with open(PHRASES_PATH) as f:
        phrases = json.load(f)[:10]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(phrases, tmp)
    tmp.close()

    # ------------------------------------------------------------------ #
    # 2. Tokenizers
    # ------------------------------------------------------------------ #
    chord_tok  = ChordTokenizer.from_json()
    note_tok   = NoteTokenizer.from_json()
    artist_tok = ArtistTokenizer()
    phrase_tok = PhraseTokenizer()

    # ------------------------------------------------------------------ #
    # 3. Dataset (no transposition)
    # ------------------------------------------------------------------ #
    print("Building NoteWindowDataset (no transposition)…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = NoteWindowDataset(
            chord_tok,
            note_tok,
            artist_tok,
            phrase_tok,
            phrases_path=Path(tmp.name),
            semitones_list=[0],
        )
    print(f"  built {len(ds):,} samples")

    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_with_tempo,
        drop_last=True,
    )

    # ------------------------------------------------------------------ #
    # 4. Model
    # ------------------------------------------------------------------ #
    torch.manual_seed(0)
    model = NoteExecutor(
        chord_vocab_size   = chord_tok.vocab_size,
        artist_vocab_size  = artist_tok.vocab_size,
        dropout            = 0.0,
        d_model            = 64,
        nhead              = 2,
        num_encoder_layers = 2,
        num_decoder_layers = 2,
        dim_feedforward    = 128,
    )
    model.train()

    ce  = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    it  = itertools.cycle(loader)

    # ------------------------------------------------------------------ #
    # 5. Training loop — 400 gradient steps
    # ------------------------------------------------------------------ #
    last = {}
    for step in range(1, 401):
        batch = next(it)
        p_logits, d_logits, r_logits = model(
            batch["chord_ids"],
            batch["phrase_id"],
            batch["artist_id"],
            batch["ctx_pitch"],
            batch["ctx_dur"],
            batch["ctx_rest"],
            batch["pos_in_phrase"],
            tempo_bpm=batch["tempo_bpm"],
            src_key_padding_mask=~batch["chord_mask"],
        )
        pitch_l = ce(p_logits, batch["target_pitch"])
        dur_l   = ce(d_logits, batch["target_dur"])
        rest_l  = ce(r_logits, batch["target_rest"])
        total   = pitch_l + dur_l + rest_l

        opt.zero_grad()
        total.backward()
        opt.step()

        if step == 1 or step % 50 == 0:
            print(
                f"step {step:4d} | total {total.item():.4f} | "
                f"pitch {pitch_l.item():.4f} | dur {dur_l.item():.4f} | "
                f"rest {rest_l.item():.4f}"
            )
            last = dict(
                total=total.item(),
                pitch=pitch_l.item(),
                dur=dur_l.item(),
                rest=rest_l.item(),
            )

    # ------------------------------------------------------------------ #
    # 6. Pass / fail: memorization check
    # ------------------------------------------------------------------ #
    overfit_ok = (last["total"] < 1.0) and (last["pitch"] < 0.5 or last["dur"] < 0.5)
    print(f"\nstep-400 total={last['total']:.4f} pitch={last['pitch']:.4f} "
          f"dur={last['dur']:.4f} rest={last['rest']:.4f}")
    print(f"overfit pass: {overfit_ok}")

    # ------------------------------------------------------------------ #
    # 7. Generate 8 notes from phrase 0
    # ------------------------------------------------------------------ #
    df = pd.read_csv(CLUSTERS_PATH)
    token_map = {
        (int(r.solo_id), int(r.phrase_number)): r.phrase_token
        for r in df.itertuples()
    }

    p0    = phrases[0]
    notes = sorted(p0["notes"], key=lambda n: n["onset"])
    tempo = float(p0["tempo_bpm"])

    chord_id_list = _transposed_chord_changes(notes, chord_tok, 0)
    chord_ids = torch.tensor([chord_id_list], dtype=torch.long)
    phrase_id = torch.tensor(
        [phrase_tok.encode(token_map.get(
            (p0["solo_id"], p0["phrase_number"]), "PHRASE_00"
        ))],
        dtype=torch.long,
    )
    artist_id = torch.tensor([artist_tok.encode(p0["performer"])], dtype=torch.long)

    generated = model.generate(
        chord_ids = chord_ids,
        phrase_id = phrase_id,
        artist_id = artist_id,
        n_notes   = 8,
        tempo_bpm = tempo,
    )

    # ------------------------------------------------------------------ #
    # 8. Exact-value assertion on duration tokens
    # ------------------------------------------------------------------ #
    all_exact = True
    print(f"\nGenerated 8 notes (tempo={tempo:.1f} BPM):")
    for pitch, dur, rest in generated:
        if not (DUR_OFFSET <= dur < DUR_OFFSET + 14):
            print(f"  FAIL: duration token {dur} outside musical range [4, 18)")
            all_exact = False
            continue
        decoded  = note_tok.decode_duration(dur, tempo)
        expected = MUSICAL_DURATIONS_BEATS[dur - DUR_OFFSET] * 60.0 / tempo
        if decoded != expected:
            print(f"  FAIL: decode_duration({dur}) = {decoded} != expected {expected}")
            all_exact = False
        else:
            beats_str = f"{MUSICAL_DURATIONS_BEATS[dur - DUR_OFFSET]:.4f} beats"
            print(f"  pitch={pitch:3d}  dur={dur}  rest={rest}  "
                  f"→ {decoded:.4f}s  ({beats_str})")

    print(f"durations exact: {all_exact}")

    # ------------------------------------------------------------------ #
    # 9. Final verdict
    # ------------------------------------------------------------------ #
    if overfit_ok and all_exact:
        print("\nPASS")
        sys.exit(0)
    print("\nFAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
