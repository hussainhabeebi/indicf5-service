"""
IndicF5 TTS microservice for Aiingo
Exposes POST /tts -> WAV audio
Run behind Coolify; n8n calls it via HTTP Request node.
"""

import io
import os

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoModel

# HF_TOKEN env var required (model is gated - accept terms on HF first)
MODEL_ID = "ai4bharat/IndicF5"

# Default reference voice (mount your own Malayalam sample + transcript)
DEFAULT_REF_AUDIO = os.environ.get("REF_AUDIO", "/refs/ml_female.wav")
DEFAULT_REF_TEXT = os.environ.get(
    "REF_TEXT",
    "ഇവിടെ നിങ്ങളുടെ റഫറൻസ് ഓഡിയോയുടെ കൃത്യമായ ട്രാൻസ്ക്രിപ്റ്റ് നൽകുക",
)

app = FastAPI(title="Aiingo IndicF5 TTS")

print("Loading IndicF5 model (first run downloads ~1.5GB)...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN"),
    low_cpu_mem_usage=False,
)
print("Model loaded.")


class TTSRequest(BaseModel):
    text: str
    ref_audio: str | None = None   # path inside container, e.g. /refs/ml_male.wav
    ref_text: str | None = None    # exact transcript of the reference audio


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/tts")
def tts(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    ref_audio = req.ref_audio or DEFAULT_REF_AUDIO
    ref_text = req.ref_text or DEFAULT_REF_TEXT

    if not os.path.exists(ref_audio):
        raise HTTPException(status_code=400, detail=f"ref audio not found: {ref_audio}")

    try:
        audio = model(
            req.text,
            ref_audio_path=ref_audio,
            ref_text=ref_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

    audio = np.array(audio, dtype=np.float32)
    # IndicF5 outputs 24kHz; normalize if int-ranged
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0

    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="tts.wav"'},
    )
