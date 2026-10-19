# LookLab API (CPU-only image: no CUDA needed to run the app).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    LOOKLAB_DATA_DIR=/data LOOKLAB_AI_CONFIG_PATH=/app/config/ai.yaml
WORKDIR /app
COPY ml ml
COPY services/api services/api
COPY config config
RUN pip install ./ml ./services/api
VOLUME /data
EXPOSE 8000
# 0.0.0.0 inside the container; compose publishes it on 127.0.0.1 only.
CMD ["uvicorn", "looklab_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
