# Imagen de inferencia: Python slim + ONNX Runtime (sin TensorFlow, ~10x más liviana)
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/onnx/fruit_resnet50_int8.onnx \
    MODEL_METADATA_PATH=/app/models/onnx/model_metadata.json \
    DYNAMIC_BATCHING=true \
    MAX_BATCH_SIZE=8 \
    MAX_WAIT_MS=10 \
    PORT=8000

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt

RUN useradd --create-home --uid 10001 appuser
COPY app/ app/
COPY models/onnx/fruit_resnet50_int8.onnx models/onnx/model_metadata.json models/onnx/
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,sys; p=os.getenv('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
