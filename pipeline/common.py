"""Utilidades comunes del pipeline (carga de dataset, parámetros, rutas)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.preprocessing import load_image  # noqa: E402

MODELS_DIR = ROOT / "models"
ONNX_DIR = MODELS_DIR / "onnx"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures_eva4"
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def list_dataset(dataset_dir: str | Path) -> tuple[list[str], dict[str, list[Path]]]:
    dataset_dir = ROOT / dataset_dir
    classes = sorted(d.name for d in dataset_dir.iterdir() if d.is_dir())
    files = {
        c: sorted(p for p in (dataset_dir / c).iterdir() if p.suffix.lower() in VALID_EXT)
        for c in classes
    }
    return classes, files


def split_dataset(params: dict):
    """Partición estratificada idéntica a ``validation_split`` de Keras (Eva3):
    el primer 20 % (orden alfabético) de cada clase va a validación."""
    classes, files = list_dataset(params["data"]["dataset_dir"])
    vs = params["data"]["validation_split"]
    train, val = [], []
    for idx, c in enumerate(classes):
        n_val = int(round(len(files[c]) * vs))
        val += [(p, idx) for p in files[c][:n_val]]
        train += [(p, idx) for p in files[c][n_val:]]
    return classes, train, val


def load_arrays(items, img_size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([load_image(str(p), img_size) for p, _ in items]).astype(np.float32)
    y = np.array([label for _, label in items], dtype=np.int64)
    return x, y


def md5sum(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
