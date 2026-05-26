"""MusicGen Salad Server — text-to-music via HuggingFace transformers.
Endpoints:
  GET  /ready      — health check (Salad expects this)
  POST /generate   — {prompt, duration} → returns WAV
  GET  /models     — list available models + what's loaded

MODEL env var: facebook/musicgen-small (default), facebook/musicgen-medium
Small ~3.5GB VRAM, medium ~10GB. RTX 3060 12GB can run medium.
"""
import argparse
import io
import os
import time

import torch
import torchaudio
from flask import Flask, jsonify, request, send_file
from transformers import AutoProcessor, MusicgenForConditionalGeneration

app = Flask(__name__)
model = None
processor = None
current_model_name = None

DEFAULT_MODEL = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
HF_TOKEN = ***"HF_TOKEN", "")

SUPPORTED = {
    "small":  "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large":  "facebook/musicgen-large",
}


def _resolve(name):
    name = name.strip().lower()
    return SUPPORTED.get(name, name)


def _free_memory():
    global model, processor, current_model_name
    model = None
    processor = None
    current_model_name = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model(name=None, force=False):
    global model, processor, current_model_name
    name = _resolve(name or DEFAULT_MODEL)
    if not force and current_model_name == name:
        return True
    print(f"Loading {name}...")
    _free_memory()
    load_kwargs = {"torch_dtype": torch.float16}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN
    processor = AutoProcessor.from_pretrained(name)
    model = MusicgenForConditionalGeneration.from_pretrained(
        name, **load_kwargs
    ).to("cuda")
    current_model_name = name
    alloc = torch.cuda.memory_allocated() / 1e9
    print(f"MusicGen ready. Model: {name}  VRAM: {alloc:.1f}GB")
    return True


@app.route("/ready")
def ready():
    return jsonify({
        "status": "ok",
        "model": current_model_name or "not loaded",
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 1) if current_model_name else 0,
    })


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "current": current_model_name,
        "default": DEFAULT_MODEL,
        "available": SUPPORTED,
    })


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "upbeat electronic music")
    duration_sec = min(data.get("duration", 8), 30)
    want_model = data.get("model", DEFAULT_MODEL)

    if not current_model_name:
        load_model(want_model)
    elif _resolve(want_model) != current_model_name:
        load_model(want_model, force=True)

    t0 = time.time()
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
    max_tokens = min(int(duration_sec * 50), 1500)

    try:
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                max_new_tokens=max_tokens,
            )
    except torch.cuda.OutOfMemoryError:
        _free_memory()
        if "small" not in current_model_name.lower():
            print("OOM — falling back to small")
            load_model("facebook/musicgen-small", force=True)
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=min(max_tokens, 800),
                )
        else:
            return jsonify({"error": "OOM on musicgen-small"}), 507

    audio = audio_values[0].cpu().float()
    sample_rate = model.config.audio_encoder.sampling_rate
    elapsed = time.time() - t0

    buf = io.BytesIO()
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    torchaudio.save(buf, audio, sample_rate, format="wav")
    buf.seek(0)

    actual_dur = audio.shape[-1] / sample_rate
    print(f"Generated {actual_dur:.1f}s in {elapsed:.1f}s | {current_model_name} | {prompt[:80]}")
    return send_file(buf, mimetype="audio/wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    load_model(DEFAULT_MODEL)
    app.run(host=args.host, port=args.port, debug=False)
