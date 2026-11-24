"""Motor de inferencia sobre ONNX Runtime (runtime optimizado, sin TensorFlow)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


class InferenceEngine:
    def __init__(self, model_path: Path, metadata_path: Path, intra_op_threads: int = 0):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        with open(metadata_path, encoding="utf-8") as fh:
            self.metadata = json.load(fh)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if intra_op_threads > 0:
            so.intra_op_num_threads = intra_op_threads
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.load_time_s = time.perf_counter() - t0
        self.input_name = self.session.get_inputs()[0].name
        self.classes: list[str] = self.metadata["classes"]
        self.model_path = Path(model_path)
        self.warmup()

    def warmup(self) -> None:
        self.predict(np.zeros((1, 224, 224, 3), dtype=np.float32))

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """batch: float32 (N, 224, 224, 3) en [0, 255] -> probabilidades (N, C)."""
        return self.session.run(None, {self.input_name: batch.astype(np.float32, copy=False)})[0]
