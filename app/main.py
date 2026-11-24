"""API REST de inferencia (FastAPI) para el clasificador de frutas ResNet50 INT8."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from app.batcher import DynamicBatcher
from app.config import settings
from app.inference import InferenceEngine
from app.preprocessing import InvalidImageError, load_image
from app.schemas import (
    BatchPredictResponse,
    ClassProbability,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fruit-api")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
STATIC_DIR = Path(__file__).parent / "static"


class Metrics:
    def __init__(self) -> None:
        self.requests = 0
        self.errors = 0
        self.images = 0
        self.low_confidence = 0
        self.latencies: list[float] = []
        self.by_label: dict[str, int] = {}

    def observe(self, ms: float) -> None:
        self.latencies.append(ms)
        self.latencies = self.latencies[-1000:]

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.latencies, q)) if self.latencies else 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.metrics = Metrics()
    app.state.engine = None
    app.state.batcher = None
    try:
        engine = InferenceEngine(settings.model_path, settings.metadata_path, settings.intra_op_threads)
        app.state.engine = engine
        log.info("Modelo %s cargado en %.2fs", engine.metadata["model_version"], engine.load_time_s)
        if settings.dynamic_batching:
            app.state.batcher = DynamicBatcher(engine.predict, settings.max_batch_size, settings.max_wait_ms)
            await app.state.batcher.start()
    except Exception:
        log.exception("No se pudo cargar el modelo; el servicio responderá 503")
    yield
    if app.state.batcher:
        await app.state.batcher.stop()


app = FastAPI(
    title="Fruit Classifier API — Evaluación N°4",
    description="Inferencia en tiempo real de 5 clases de frutas con ResNet50 cuantizada a INT8 "
    "(ONNX Runtime) y batching dinámico. Continuación de la Evaluación N°3.",
    version="1.0.0",
    lifespan=lifespan,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
               415: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    request.app.state.metrics.requests += 1
    response = await call_next(request)
    if response.status_code >= 400:
        request.app.state.metrics.errors += 1
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Error no controlado en %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor de inferencia."})


def get_engine(request: Request) -> InferenceEngine:
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(503, "Modelo no disponible.")
    return engine


async def read_image(file: UploadFile) -> np.ndarray:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Tipo de archivo no soportado: {file.content_type}. "
                                 f"Use uno de {sorted(ALLOWED_TYPES)}.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "El archivo está vacío.")
    if len(data) > settings.max_upload_mb * 2**20:
        raise HTTPException(413, f"El archivo supera el límite de {settings.max_upload_mb} MB.")
    try:
        return await asyncio.to_thread(load_image, data)
    except InvalidImageError as exc:
        raise HTTPException(400, str(exc)) from exc


def to_prediction(probs: np.ndarray, classes: list[str], filename: str | None, metrics: Metrics) -> Prediction:
    order = np.argsort(probs)[::-1]
    label = classes[int(order[0])]
    conf = float(probs[order[0]])
    low = conf < settings.low_confidence_threshold
    metrics.images += 1
    metrics.low_confidence += int(low)
    metrics.by_label[label] = metrics.by_label.get(label, 0) + 1
    return Prediction(
        filename=filename,
        label=label,
        confidence=round(conf, 6),
        low_confidence=low,
        top_k=[ClassProbability(label=classes[int(i)], probability=round(float(probs[i]), 6)) for i in order[:3]],
        probabilities={c: round(float(p), 6) for c, p in zip(classes, probs, strict=False)},
    )


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["operación"])
async def health(request: Request):
    return HealthResponse(status="ok", model_loaded=request.app.state.engine is not None)


@app.get("/ready", response_model=HealthResponse, tags=["operación"])
async def ready(request: Request):
    get_engine(request)
    return HealthResponse(status="ready", model_loaded=True)


@app.get("/model/info", response_model=ModelInfo, tags=["modelo"])
async def model_info(request: Request):
    engine = get_engine(request)
    md = engine.metadata
    return ModelInfo(
        model_name=md["model_name"],
        model_version=md["model_version"],
        runtime=f"onnxruntime {ort.__version__} (CPUExecutionProvider)",
        quantization=md["quantization"],
        classes=engine.classes,
        input_shape=md["input_shape"],
        model_size_mb=round(engine.model_path.stat().st_size / 2**20, 2),
        load_time_s=round(engine.load_time_s, 3),
        dynamic_batching=settings.dynamic_batching,
        max_batch_size=settings.max_batch_size,
        max_wait_ms=settings.max_wait_ms,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inferencia"])
async def predict(request: Request, file: UploadFile = File(..., description="Imagen JPG/PNG/WEBP/BMP")):
    t0 = time.perf_counter()
    engine = get_engine(request)
    img = await read_image(file)
    t_inf = time.perf_counter()
    batcher: DynamicBatcher | None = request.app.state.batcher
    if batcher is not None:
        probs = await batcher.submit(img)
    else:
        probs = (await asyncio.to_thread(engine.predict, img[None]))[0]
    inference_ms = (time.perf_counter() - t_inf) * 1000
    pred = to_prediction(probs, engine.classes, file.filename, request.app.state.metrics)
    total_ms = (time.perf_counter() - t0) * 1000
    request.app.state.metrics.observe(total_ms)
    log.info("predict file=%s label=%s conf=%.3f total=%.1fms", file.filename, pred.label, pred.confidence, total_ms)
    return PredictResponse(prediction=pred, model_version=engine.metadata["model_version"],
                           inference_ms=round(inference_ms, 2), total_ms=round(total_ms, 2))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inferencia"])
async def predict_batch(request: Request, files: list[UploadFile] = File(...)):
    t0 = time.perf_counter()
    engine = get_engine(request)
    if not files:
        raise HTTPException(400, "Debe enviar al menos una imagen.")
    if len(files) > settings.max_files_per_request:
        raise HTTPException(413, f"Máximo {settings.max_files_per_request} imágenes por petición.")
    images = [await read_image(f) for f in files]
    t_inf = time.perf_counter()
    probs = await asyncio.to_thread(engine.predict, np.stack(images))
    inference_ms = (time.perf_counter() - t_inf) * 1000
    metrics = request.app.state.metrics
    preds = [to_prediction(p, engine.classes, f.filename, metrics) for p, f in zip(probs, files, strict=False)]
    total_ms = (time.perf_counter() - t0) * 1000
    request.app.state.metrics.observe(total_ms)
    return BatchPredictResponse(predictions=preds, model_version=engine.metadata["model_version"],
                                inference_ms=round(inference_ms, 2), total_ms=round(total_ms, 2))


@app.get("/metrics", response_class=PlainTextResponse, tags=["operación"])
async def metrics(request: Request):
    m: Metrics = request.app.state.metrics
    b: DynamicBatcher | None = request.app.state.batcher
    lines = [
        f"fruit_api_requests_total {m.requests}",
        f"fruit_api_errors_total {m.errors}",
        f"fruit_api_images_classified_total {m.images}",
        f"fruit_api_low_confidence_total {m.low_confidence}",
        f"fruit_api_latency_ms_p50 {m.percentile(50):.2f}",
        f"fruit_api_latency_ms_p95 {m.percentile(95):.2f}",
        f"fruit_api_dynamic_batch_avg_size {b.avg_batch_size if b else 0:.2f}",
    ]
    lines += [f'fruit_api_predictions_total{{label="{k}"}} {v}' for k, v in sorted(m.by_label.items())]
    return "\n".join(lines) + "\n"
