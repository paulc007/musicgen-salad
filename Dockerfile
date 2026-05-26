# MusicGen Salad — text-to-music on SaladCloud (RTX 3060 12GB, batch priority)
# Defaults to musicgen-small (~3.5GB VRAM), or set MUSICGEN_MODEL=facebook/musicgen-medium
#
# Build:  docker build -t ghcr.io/paulc007/musicgen-salad:latest .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV TZ=America/Edmonton
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Base image ships PyTorch 2.5.1 + torchaudio. Only add what's missing.
RUN pip install --no-cache-dir \
    transformers \
    flask \
    scipy

COPY server.py /app/server.py
RUN mkdir -p /output

EXPOSE 8000
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
