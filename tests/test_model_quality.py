"""Quality gates de MLOps ejecutados en CI: exactitud, latencia e integridad del artefacto."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest
import yaml

from app.config import settings
from app.inference import InferenceEngine
from app.preprocessing import load_image

from .conftest import DATASET, ROOT

PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def engine():
    return InferenceEngine(settings.model_path, settings.metadata_path)


def validation_set(classes):
    vs = PARAMS["data"]["validation_split"]
    x, y = [], []
    for idx, c in enumerate(classes):
        files = sorted(p for p in (DATASET / c).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for p in files[: int(round(len(files) * vs))]:
            x.append(load_image(str(p)))
            y.append(idx)
    return np.stack(x), np.array(y)


def test_accuracy_gate(engine):
    x, y = validation_set(engine.classes)
    acc = float(np.mean(engine.predict(x).argmax(1) == y))
    assert acc >= PARAMS["serving"]["accuracy_gate"], f"exactitud {acc:.2%} bajo el umbral"


def test_latency_gate(engine):
    x = np.random.default_rng(0).uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
    for _ in range(3):
        engine.predict(x)
    lat = []
    for _ in range(30):
        t = time.perf_counter()
        engine.predict(x)
        lat.append((time.perf_counter() - t) * 1000)
    p95 = float(np.percentile(lat, 95))
    assert p95 <= PARAMS["serving"]["latency_p95_gate_ms"], f"p95={p95:.1f} ms excede el umbral"


def test_artifact_matches_dvc_lock():
    """El modelo INT8 versionado en git debe ser exactamente el producido por `dvc repro`."""
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text(encoding="utf-8"))
    outs = {o["path"]: o["md5"] for o in lock["stages"]["quantize"]["outs"]}
    assert hashlib.md5(settings.model_path.read_bytes()).hexdigest() == outs["models/onnx/fruit_resnet50_int8.onnx"]


def test_probabilities_are_valid(engine):
    x = np.random.default_rng(1).uniform(0, 255, (4, 224, 224, 3)).astype(np.float32)
    p = engine.predict(x)
    assert p.shape == (4, len(engine.classes))
    assert np.allclose(p.sum(1), 1, atol=1e-3)
