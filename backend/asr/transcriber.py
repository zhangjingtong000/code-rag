"""ASR transcriber — subprocess wrapper for Qwen3-ASR.

Supports multiple backends, auto-detected in order:
  1. qwen3-asr-cli (cnjack/qwen3-asr-cli)
  2. antirez/qwen-asr  (pure C binary)
  3. qwen-asr Python package (official)

Configure via Settings: asr_backend, asr_model_dir, asr_binary_path.
"""

import os
import sys
import subprocess
import shutil
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
        raise RuntimeError("No ASR backend available. Tried: qwen3-asr-cli, qwen-asr-py, antirez-c")


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


def _resolve_model_path() -> str:
    """Local model dir when it exists, otherwise the HF repo id.

    The comment used to promise this fallback but the code never implemented it:
    with qwen-asr installed and no model downloaded, the project-rooted
    ``models/qwen3-asr-0.6b`` path was passed straight to ``from_pretrained``,
    which raises ``ValueError("... is not a valid model identifier")``. audio.py
    only caught RuntimeError, so the endpoint answered 500 instead of the
    documented 503 "ASR backend unavailable".
    """
    model_dir = settings.asr_model_dir
    if model_dir:
        candidate = (model_dir if os.path.isabs(model_dir)
                     else os.path.join(settings.project_root, model_dir))
        if os.path.isdir(candidate):
            return candidate
        print(f"[ASR] 本地模型目录不存在：{candidate}，退回 HF repo id "
              f"{settings.asr_model_name}", flush=True)
    return settings.asr_model_name


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
        model_path = _resolve_model_path()
        print(f"[ASR] Model path: {model_path}", flush=True)
        try:
            _transcribe_python._model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map=settings.asr_device,
            )
        except Exception as exc:
            # 模型加载失败 = 后端不可用。抛 RuntimeError，让 /audio/transcribe
            # 按注释承诺返回 503，而不是被 ValueError 落进 500 分支。
            raise RuntimeError(f"Qwen3-ASR model unavailable: {exc}") from exc

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
