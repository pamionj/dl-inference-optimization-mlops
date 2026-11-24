"""Configuración del servicio vía variables de entorno (12-factor)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "si"}


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", ROOT / "models/onnx/fruit_resnet50_int8.onnx"))
    metadata_path: Path = Path(os.getenv("MODEL_METADATA_PATH", ROOT / "models/onnx/model_metadata.json"))
    intra_op_threads: int = int(os.getenv("ORT_INTRA_OP_THREADS", "0"))
    dynamic_batching: bool = _bool("DYNAMIC_BATCHING", True)
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "8"))
    max_wait_ms: float = float(os.getenv("MAX_WAIT_MS", "10"))
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "5"))
    max_files_per_request: int = int(os.getenv("MAX_FILES_PER_REQUEST", "16"))
    low_confidence_threshold: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.5"))


settings = Settings()
