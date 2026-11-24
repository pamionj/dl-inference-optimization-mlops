"""Etapa 5 del pipeline: registro del experimento en MLflow.

Registra parámetros (params.yaml), métricas de entrenamiento, paridad ONNX y benchmark
de cada variante, junto con el modelo INT8 servido y su metadata. Backend SQLite local
(``mlflow.db``); para un servidor remoto basta definir ``MLFLOW_TRACKING_URI``.
"""

from __future__ import annotations

import json
import os

import mlflow

from pipeline.common import METRICS_DIR, ONNX_DIR, ROOT, load_params, md5sum


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def main() -> None:
    params = load_params()
    train = json.loads((METRICS_DIR / "train_metrics.json").read_text())
    history = json.loads((METRICS_DIR / "train_history.json").read_text())
    parity = json.loads((METRICS_DIR / "export_parity.json").read_text())
    bench = json.loads((METRICS_DIR / "benchmark.json").read_text())["results"]
    int8 = ONNX_DIR / "fruit_resnet50_int8.onnx"

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("fruit-classifier-eva4")
    with mlflow.start_run(run_name=f"resnet50-int8-{md5sum(int8)[:8]}") as run:
        mlflow.set_tags({"eva": "4", "base": "eva3-resnet50", "serving_format": "onnx-int8-qdq"})
        mlflow.log_params(flatten(params))
        mlflow.log_metrics({"val_accuracy": train["val_accuracy"], "train_time_s": train["train_time_s"]})
        for i, (a, va) in enumerate(zip(history["accuracy"], history["val_accuracy"], strict=False)):
            mlflow.log_metrics({"epoch_accuracy": a, "epoch_val_accuracy": va}, step=i)
        mlflow.log_metrics({f"onnx_parity.{k}": v for k, v in parity.items() if isinstance(v, (int, float))})
        for name, r in bench.items():
            mlflow.log_metrics({
                f"{name}.size_mb": r["size_mb"],
                f"{name}.rss_model_mb": r["rss_model_mb"],
                f"{name}.latency_p50_ms": r["latency_ms"]["p50"],
                f"{name}.latency_p95_ms": r["latency_ms"]["p95"],
                f"{name}.accuracy_val": r["accuracy_val"],
                f"{name}.top1_agreement_vs_fp32": r["top1_agreement_vs_fp32"],
            })
        for path in [int8, ONNX_DIR / "model_metadata.json", ROOT / "params.yaml", *METRICS_DIR.glob("*.json")]:
            mlflow.log_artifact(str(path))
        print(f"[OK] MLflow run {run.info.run_id}")


if __name__ == "__main__":
    main()
