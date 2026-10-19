# LookLab GPU worker — FASHN VTON v1.5 (official code, pinned commit).
# Build:  docker build -f infrastructure/gpu-worker/Dockerfile -t looklab-gpu-worker .
# Run:    docker run --gpus all -e LOOKLAB_WORKER_TOKEN=... -v looklab-weights:/weights -p 8080:8080 looklab-gpu-worker
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG FASHN_VTON_COMMIT=7c0f10af3f91ad4048fe9729c470a13ef905d25a
ARG WITH_HUMAN_PARSER=0
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 FASHN_WEIGHTS_DIR=/weights FASHN_PARSER_MODE=parser_free

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Official inference code. --no-deps keeps fashn-human-parser (NVIDIA non-commercial
# license) out of the image unless WITH_HUMAN_PARSER=1 is passed explicitly.
RUN git clone https://github.com/fashn-AI/fashn-vton-1.5 /opt/fashn-vton \
 && git -C /opt/fashn-vton checkout ${FASHN_VTON_COMMIT} \
 && pip install --no-deps -e /opt/fashn-vton \
 && pip install safetensors huggingface_hub opencv-python-headless einops onnxruntime-gpu matplotlib tqdm \
    fastapi "uvicorn[standard]" python-multipart httpx pillow numpy runpod \
 && if [ "$WITH_HUMAN_PARSER" = "1" ]; then pip install fashn-human-parser transformers; fi

COPY ml /opt/looklab/ml
RUN pip install --no-deps -e /opt/looklab/ml
COPY infrastructure/gpu-worker /opt/looklab/worker
WORKDIR /opt/looklab/worker

EXPOSE 8080
# Weights (~2 GB) are fetched once into the /weights volume on first start.
CMD ["sh", "-c", "[ -f /weights/model.safetensors ] || python fetch_weights.py /weights; exec uvicorn worker:app --host 0.0.0.0 --port 8080"]
