# MusicGen Salad — text-to-music on SaladCloud (RTX 3060 12GB, batch priority)
# Defaults to musicgen-small (~3.5GB VRAM), can run medium on this GPU.
#
# Build:  docker build -t ghcr.io/paulc007/musicgen-salad:latest .
# Push:   docker push ghcr.io/paulc007/musicgen-salad:latest
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV TZ=America/Edmonton
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torchaudio==2.5.1 \
    "transformers<4.50" \
    flask \
    scipy

RUN mkdir -p /output

COPY server.py /app/server.py

EXPOSE 8000
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
