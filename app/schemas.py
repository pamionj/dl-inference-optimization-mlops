"""Contratos de entrada/salida (Pydantic) de la API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    label: str
    probability: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    filename: str | None = None
    label: str = Field(description="Clase con mayor probabilidad")
    confidence: float = Field(ge=0, le=1)
    low_confidence: bool = Field(description="True si confidence < umbral configurado")
    top_k: list[ClassProbability]
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    prediction: Prediction
    model_version: str
    inference_ms: float
    total_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    inference_ms: float
    total_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    runtime: str
    quantization: str
    classes: list[str]
    input_shape: list
    model_size_mb: float
    load_time_s: float
    dynamic_batching: bool
    max_batch_size: int
    max_wait_ms: float


class ErrorResponse(BaseModel):
    detail: str
