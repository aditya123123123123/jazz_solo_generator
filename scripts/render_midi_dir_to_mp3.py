#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
from scipy.io import wavfile


def render_one(midi_path: Path, mp3_path: Path, sample_rate: int = 22050) -> None:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.synthesize(fs=sample_rate)
    if audio.size == 0:
        raise RuntimeError(f"no audio synthesized from {midi_path}")
    peak = float(np.max(np.abs(audio))) or 1.0
    audio_i16 = np.int16(np.clip(audio / peak * 0.95, -1.0, 1.0) * 32767)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        wavfile.write(str(wav_path), sample_rate, audio_i16)
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav_path), '-codec:a', 'libmp3lame', '-q:a', '2', str(mp3_path)],
            check=True,
        )
    finally:
        wav_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description='Render a directory of MIDI files to MP3 previews using pretty_midi synthesis.')
    parser.add_argument('--midi-dir', type=Path, required=True)
    parser.add_argument('--mp3-dir', type=Path, required=True)
    args = parser.parse_args()
    midis = sorted(args.midi_dir.glob('*.mid')) + sorted(args.midi_dir.glob('*.midi'))
    if not midis:
        raise SystemExit(f'no MIDI files found in {args.midi_dir}')
    for midi in midis:
        mp3 = args.mp3_dir / f'{midi.stem}.mp3'
        render_one(midi, mp3)
        print(mp3)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
