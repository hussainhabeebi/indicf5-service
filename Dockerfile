FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers==4.49.0 \
    accelerate==0.33.0 \
    soundfile==0.12.1 \
    safetensors==0.4.3 \
    huggingface_hub==0.29.0 \
    numpy==1.26.4 \
    pydub==0.25.1 \
    fastapi \
    "uvicorn[standard]" \
    git+https://github.com/ai4bharat/IndicF5.git

COPY app.py /app/app.py

RUN mkdir -p /refs
VOLUME ["/refs", "/root/.cache/huggingface"]

ENV HF_TOKEN=""
ENV REF_AUDIO="/refs/ml_female.wav"
ENV REF_TEXT=""

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
