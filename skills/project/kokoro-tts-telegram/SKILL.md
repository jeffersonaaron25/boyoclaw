---
name: kokoro-tts-telegram
description: Synthesize speech with Kokoro TTS using scripts in this skill folder, save WAV/MP3 in the workspace, and send via send_file_to_user on Telegram when the user is on Telegram.
allowed-tools: execute, send_message
metadata:
  platform: any
  requires: kokoro-tts CLI, voices-v1.0.bin + kokoro-v1.0.onnx in kokoro_models/
---

# Kokoro TTS + Telegram audio

## Shell paths (`execute` only)

**`read_file` / `ls` may show `/skills/project/...` — that path does not exist in the shell.** Bundled skills are on disk at **`$HOME/skills/project/...`** (``$HOME`` is `.agent-home`). Always use one of:

- ``bash "$HOME/skills/project/kokoro-tts-telegram/scripts/tts_kokoro.sh" ...``
- ``cd "$HOME" && bash skills/project/kokoro-tts-telegram/scripts/tts_kokoro.sh ...``

To confirm the script exists: ``ls "$HOME/skills/project/kokoro-tts-telegram/scripts"``.

## Scripts

- ``scripts/tts_kokoro.sh`` — run Kokoro (expects model files under ``kokoro_models/``)
- ``scripts/setup_audio_models.sh`` — download ONNX and voice data (run **once** from a normal shell with internet if models are missing)

## Before first use

Model files must exist under ``kokoro_models/`` (e.g. ``kokoro-v1.0.onnx`` and ``voices-v1.0.bin``). If they are missing, a human should run ``setup_audio_models.sh`` with network access, or place the files there manually. Install Python deps from the repo’s ``requirements-audio.txt`` in the **project audio / venv** the operator set up — do **not** rely on ad-hoc ``pip install kokoro-tts`` into Xcode / system Python (often fails dependency pins).

## Generate speech

Write output under something like ``outputs/audio/``. Use ``$HOME`` for paths under the agent workspace when writing files.

```bash
mkdir -p "$HOME/outputs/audio"
printf '%s\n' "Your short spoken reply here." | bash "$HOME/skills/project/kokoro-tts-telegram/scripts/tts_kokoro.sh" - "$HOME/outputs/audio/reply.wav" --lang en-us --voice af_sarah --format wav
```

List voices: ``bash "$HOME/skills/project/kokoro-tts-telegram/scripts/tts_kokoro.sh" --help-voices``

## macOS-only fallback (no Kokoro)

If ``kokoro-tts`` is unavailable, you may use **`say`** to produce speech, then **`send_file_to_user`**. Modern macOS **`say -o file.aiff`** can write **compressed** AIFF; stdlib ``aifc`` / ``afconvert`` to WAV may fail — that is OK: send the **``.aiff``** (or **``.caf``**) as produced, or use Telegram-acceptable formats without a full-disk **`find /`**. Do not spend many steps on conversion unless a simple command works.

## Send on Telegram

After the file exists, use ``send_file_to_user`` with a **workspace-relative** path (e.g. ``outputs/audio/reply.wav``). Add ``send_message`` if the user should also get text.

## If something fails

- **127 on script:** you used ``/skills/...`` in the shell — switch to ``"$HOME/skills/project/..."``.
- Missing models: run ``"$HOME/skills/project/kokoro-tts-telegram/scripts/setup_audio_models.sh"`` (or install files into ``kokoro_models/``).
- Missing ``kokoro-tts``: ask a human to install from ``requirements-audio.txt`` in the proper environment; avoid random ``pip install`` on system Python.
- Telegram send fails: check file size limits; give the user the path in ``send_message``.
