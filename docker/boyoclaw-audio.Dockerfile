# BoyoClaw sandbox image with Kokoro TTS + faster-whisper (CPU). Build on the host with network.
# Runtime uses --network none; ONNX / Whisper weights must live on the bind-mounted workspace
# (download with skills/project/kokoro-tts-telegram/scripts/setup_audio_models.sh on the host).
#
#   docker build -f docker/boyoclaw-audio.Dockerfile -t boyoclaw-audio:local .
#   export BOYOCLAW_DOCKER_IMAGE=boyoclaw-audio:local
#
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-audio.txt /tmp/requirements-audio.txt
RUN pip install --no-cache-dir -r /tmp/requirements-audio.txt \
    && rm /tmp/requirements-audio.txt

WORKDIR /sandbox
