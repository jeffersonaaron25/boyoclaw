---
name: audio-stt-faster-whisper
description: Transcribe audio files in the workspace with faster-whisper using scripts/stt_faster_whisper.py (e.g. voice notes under telegram_uploads/).
allowed-tools: execute, read_file, send_message
metadata:
  platform: any
  requires: faster-whisper, ffmpeg (for many formats)
---

# Speech-to-text (faster-whisper)

Script on disk: ``$HOME/skills/project/audio-stt-faster-whisper/scripts/stt_faster_whisper.py`` (after ``cd "$HOME"``, you can use the relative path ``skills/project/audio-stt-faster-whisper/scripts/stt_faster_whisper.py``). Do **not** use ``/skills/project/...`` in the shell — that prefix is only for ``read_file`` / ``ls``.

## Setup

**faster-whisper** and **ffmpeg** must be available where ``execute`` runs. If they are missing, a human should install them (e.g. from the repo’s ``requirements-audio.txt`` plus system ffmpeg). The first run may download model weights into ``.cache/faster-whisper`` under the workspace—use a network once, or prefetch using the optional step in ``skills/project/kokoro-tts-telegram/scripts/setup_audio_models.sh`` with ``PREFETCH_WHISPER=1``.

## Transcribe

Paths are relative to the agent workspace (``.agent-home``), for example Telegram uploads:

```bash
cd "$HOME"
python3 skills/project/audio-stt-faster-whisper/scripts/stt_faster_whisper.py telegram_uploads/2026-03-23/voice.ogg --model base
```

Stdout is plain text—use it in ``send_message`` or follow-up steps.

## Options

- ``--model`` — ``tiny``, ``base``, ``small``, etc. (default ``base``).
- ``--language`` — optional; omit for auto-detect.
- ``--device`` — ``cpu`` (default) or ``cuda`` if available.

## Formats

Common formats (ogg, mp3, m4a, wav) need **ffmpeg** on ``PATH``.
