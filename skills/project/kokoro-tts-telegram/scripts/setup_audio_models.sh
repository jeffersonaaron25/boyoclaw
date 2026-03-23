#!/usr/bin/env bash
# Download Kokoro ONNX + voice pack into the agent workspace.
#
# Run on the HOST (or any machine with network)—NOT inside BoyoClaw's Docker sandbox:
# that environment uses --network none, so wget/curl to GitHub will fail there.
#
# Optional Whisper prefetch also needs network and is best done on the host, not in the sandbox.
#
# Usage:
#   export BOYOCLAW_AGENT_HOME="/path/to/.sandbox/workspace/.agent-home"
#   bash skills/project/kokoro-tts-telegram/scripts/setup_audio_models.sh
set -euo pipefail
# Repo root: .../skills/project/kokoro-tts-telegram/scripts/ -> four levels up
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
AGENT_HOME="${BOYOCLAW_AGENT_HOME:-$ROOT/.sandbox/workspace/.agent-home}"
MODEL_DIR="${KOKORO_MODEL_DIR:-$AGENT_HOME/kokoro_models}"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo "Downloading Kokoro models into $MODEL_DIR ..."
if command -v wget >/dev/null 2>&1; then
  wget -q --show-progress -O voices-v1.0.bin \
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
  wget -q --show-progress -O kokoro-v1.0.onnx \
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
elif command -v curl >/dev/null 2>&1; then
  curl -fsSL -o voices-v1.0.bin \
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
  curl -fsSL -o kokoro-v1.0.onnx \
    "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
else
  echo "Need wget or curl." >&2
  exit 1
fi

echo "Kokoro files OK."

# Optional: prefetch faster-whisper base model (requires pip install faster-whisper)
if [ "${PREFETCH_WHISPER:-0}" = "1" ]; then
  CACHE="${WHISPER_CACHE:-$AGENT_HOME/.cache/faster-whisper}"
  mkdir -p "$CACHE"
  python3 - <<PY
from pathlib import Path
import os
os.environ.setdefault("BOYOCLAW_AGENT_HOME", "${AGENT_HOME}")
from faster_whisper import WhisperModel
cache = Path("${CACHE}")
WhisperModel("base", download_root=str(cache), device="cpu", compute_type="int8")
print("Whisper base model cached under", cache)
PY
fi

echo "Done. For Docker TTS/STT, build: docker build -f docker/boyoclaw-audio.Dockerfile -t boyoclaw-audio:local $ROOT"
echo "Then: export BOYOCLAW_DOCKER_IMAGE=boyoclaw-audio:local"
