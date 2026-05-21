"""Audio transcription endpoint."""

import os
import uuid
import time
from fastapi import APIRouter, HTTPException, UploadFile, File
from backend.config import settings
from backend.asr.transcriber import transcribe

router = APIRouter()


@router.post("/audio/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Receive audio blob (WAV from frontend), transcribe via Qwen3-ASR, return text."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Read audio bytes (frontend already sends 16kHz mono PCM WAV)
    audio_bytes = await file.read()
    if not audio_bytes or len(audio_bytes) < 500:
        raise HTTPException(status_code=400, detail="Audio too short or empty")

    # Write directly — frontend already converted to WAV
    os.makedirs(settings.asr_temp_dir, exist_ok=True)
    temp_id = uuid.uuid4().hex[:12]
    raw_path = os.path.join(settings.asr_temp_dir, f"{temp_id}_{int(time.time())}.wav")

    try:
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

        if os.path.getsize(raw_path) < 500:
            os.remove(raw_path)
            raise HTTPException(status_code=400, detail="Audio conversion failed — resulting file too small")

        # Transcribe
        text = transcribe(raw_path)
        if not text:
            raise HTTPException(status_code=422, detail="No speech detected")

        return {"text": text.strip()}

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception:
            pass
