FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image smaller; swap for CUDA wheel if you ever add a GPU
RUN pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "transformers>=4.44" \
    fastapi \
    "uvicorn[standard]" \
    soundfile \
    numpy \
    huggingface_hub \
    f5-tts

COPY app.py /app/app.py

RUN mkdir -p /refs
VOLUME ["/refs", "/root/.cache/huggingface"]

ENV HF_TOKEN=""
ENV REF_AUDIO="/refs/ml_female.wav"
ENV REF_TEXT=""

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
