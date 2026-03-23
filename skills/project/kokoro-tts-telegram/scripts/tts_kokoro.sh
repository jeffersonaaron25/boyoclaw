#!/usr/bin/env sh
# Kokoro TTS wrapper: models must live in KOKORO_MODEL_DIR (default: $BOYOCLAW_AGENT_HOME/kokoro_models
# or $HOME/kokoro_models). The CLI expects voices-v1.0.bin and kokoro-v1.0.onnx in the cwd during runs.
#
# Usage (paths under workspace, e.g. HOME=/sandbox in Docker):
#   mkdir -p "$HOME/outputs/audio"
#   printf '%s\n' "Short reply text." | bash skills/project/kokoro-tts-telegram/scripts/tts_kokoro.sh - "$HOME/outputs/audio/reply.wav" --lang en-us --voice af_sarah --format wav
set -e
AGENT_HOME="${BOYOCLAW_AGENT_HOME:-$HOME}"
MODEL_DIR="${KOKORO_MODEL_DIR:-$AGENT_HOME/kokoro_models}"
if [ ! -f "$MODEL_DIR/kokoro-v1.0.onnx" ] || [ ! -f "$MODEL_DIR/voices-v1.0.bin" ]; then
  echo "Missing Kokoro models under $MODEL_DIR (need kokoro-v1.0.onnx + voices-v1.0.bin)." >&2
  echo "On the host (with network), run: bash skills/project/kokoro-tts-telegram/scripts/setup_audio_models.sh" >&2
  exit 1
fi
cd "$MODEL_DIR"
exec kokoro-tts "$@"
