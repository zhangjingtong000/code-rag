"""ASR transcriber — subprocess wrapper for Qwen3-ASR.

Supports multiple backends, auto-detected in order:
  1. qwen3-asr-cli (cnjack/qwen3-asr-cli)
  2. antirez/qwen-asr  (pure C binary)
  3. qwen-asr Python package (official)

Configure via Settings: asr_backend, asr_model_dir, asr_binary_path.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from backend.config import settings


def transcribe(audio_path: str) -> str:
    """Transcribe audio file to text. Returns empty string on failure."""
    backend = settings.asr_backend

    if backend == "auto":
        backend = _detect_backend()

    if backend == "qwen3-asr-cli":
        return _transcribe_cli(audio_path)
    elif backend == "qwen-asr-py":
        return _transcribe_python(audio_path)
    elif backend == "antirez-c":
        return _transcribe_antirez(audio_path)
    else:
        raise RuntimeError(f"No ASR backend available. Tried: qwen3-asr-cli, qwen-asr-py, antirez-c")


def _detect_backend() -> str:
    """Auto-detect available ASR backend."""
    # Check qwen3-asr-cli
    if shutil.which("qwen3-asr-cli"):
        return "qwen3-asr-cli"
    # Check antirez C binary
    for candidate in ["./qwen_asr", settings.asr_binary_path]:
        if candidate and os.path.exists(candidate):
            return "antirez-c"
    # Check qwen-asr Python package
    try:
        import qwen_asr  # noqa
        return "qwen-asr-py"
    except ImportError:
        pass
    return ""


def _transcribe_cli(audio_path: str) -> str:
    """Use cnjack/qwen3-asr-cli tool."""
    cmd = ["qwen3-asr-cli", audio_path]
    model_dir = settings.asr_model_dir
    if model_dir and os.path.isdir(model_dir):
        cmd.extend(["--model", model_dir])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=getattr(settings, 'asr_timeout', 60),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            print(f"[ASR] qwen3-asr-cli stderr: {result.stderr[:200]}", flush=True)
        return output
    except FileNotFoundError:
        raise RuntimeError("qwen3-asr-cli not found. Install: uv tool install git+https://github.com/cnjack/qwen3-asr-cli.git")
    except subprocess.TimeoutExpired:
        print("[ASR] Transcription timed out", flush=True)
        return ""


def _transcribe_python(audio_path: str) -> str:
    """Use official qwen-asr Python package."""
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        raise RuntimeError("qwen-asr not installed. Install: pip install qwen-asr")

    # Lazy-load model singleton
    if not hasattr(_transcribe_python, "_model"):
        print("[ASR] Loading Qwen3-ASR model (first use)...", flush=True)
        import torch
        # Prefer local model dir, fall back to HF model name
        model_path = settings.asr_model_dir or settings.asr_model_name
        if settings.asr_model_dir and not os.path.isabs(model_path):
            model_path = os.path.join(settings.project_root, model_path)
        print(f"[ASR] Model path: {model_path}", flush=True)
        _transcribe_python._model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=settings.asr_device,
        )

    model = _transcribe_python._model
    results = model.transcribe(audio=audio_path)
    if results and len(results) > 0:
        return results[0].text
    return ""


def _transcribe_antirez(audio_path: str) -> str:
    """Use antirez/qwen-asr pure C binary."""
    binary = settings.asr_binary_path or "./qwen_asr"
    model_dir = settings.asr_model_dir or "qwen3-asr-1.7b"

    cmd = [binary, "-d", model_dir, "-i", audio_path]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=getattr(settings, 'asr_timeout', 60),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError(f"qwen_asr binary not found at {binary}")
    except subprocess.TimeoutExpired:
        print("[ASR] Transcription timed out", flush=True)
        return ""


def convert_to_wav(input_bytes: bytes, output_path: str):
    """Convert audio bytes (webm/wav/mp3) to 16kHz mono WAV using pydub."""
    try:
        from pydub import AudioSegment
        import io

        audio = AudioSegment.from_file(io.BytesIO(input_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(output_path, format="wav")
    except Exception as e:
        # If pydub fails, try writing raw bytes (assumes browser WAV)
        print(f"[ASR] pydub conversion failed ({e}), writing raw bytes", flush=True)
        with open(output_path, "wb") as f:
            f.write(input_bytes)
