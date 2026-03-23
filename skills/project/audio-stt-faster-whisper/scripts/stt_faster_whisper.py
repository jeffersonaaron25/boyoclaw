#!/usr/bin/env python3
"""Transcribe an audio file with faster-whisper (CPU). Prints text to stdout.

Models download into WHISPER_CACHE on first use (run setup_audio_models.sh on the host with network).

Example:
  python3 skills/project/audio-stt-faster-whisper/scripts/stt_faster_whisper.py telegram_uploads/2026-03-23/abc_voice.ogg --model base
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Transcribe audio using faster-whisper.")
    p.add_argument(
        "audio_path",
        help="Path to audio (wav, mp3, m4a, ogg, …) relative to cwd or absolute.",
    )
    p.add_argument(
        "--model",
        default="base",
        help="Model size: tiny, base, small, medium, large-v2, … (default: base).",
    )
    p.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g. en); default = auto-detect.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda (default: cpu).",
    )
    args = p.parse_args()

    home = Path(os.environ.get("BOYOCLAW_AGENT_HOME") or os.environ.get("HOME") or ".").resolve()
    cache = Path(
        os.environ.get("WHISPER_CACHE", home / ".cache" / "faster-whisper"),
    ).resolve()
    cache.mkdir(parents=True, exist_ok=True)

    audio = Path(args.audio_path)
    if not audio.is_file():
        print(f"Not a file: {audio}", file=sys.stderr)
        return 1

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        print(
            "faster-whisper is not installed. pip install -r requirements-audio.txt "
            f"or use the boyoclaw-audio Docker image. ({e})",
            file=sys.stderr,
        )
        return 1

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type="int8",
        download_root=str(cache),
    )
    segments, _info = model.transcribe(
        str(audio),
        language=args.language,
    )
    parts: list[str] = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    sys.stdout.write(" ".join(parts).strip())
    if parts:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
