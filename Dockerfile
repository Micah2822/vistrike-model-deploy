FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ /app/scripts/
COPY configs/ /app/configs/
COPY handler.py /app/handler.py

# Models are NOT baked in (too large).
# Either: mount a volume, download at startup, or use RunPod network volume.
# To bake in (if you want): uncomment the next line.
# COPY models/ /app/models/

CMD ["python", "-u", "/app/handler.py"]
