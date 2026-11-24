"""Etapa 4 del pipeline: benchmark de optimización (latencia, throughput, RAM, tamaño, exactitud).

Cada variante se mide en un subproceso aislado para que el consumo de RAM (RSS)
sea comparable. Uso: ``python -m pipeline.benchmark``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import psutil

from pipeline.common import METRICS_DIR, MODELS_DIR, ONNX_DIR, ROOT, load_arrays, load_params, split_dataset, write_json

VARIANTS = {
    "keras_fp32": MODELS_DIR / "fruit_resnet50.keras",
    "onnx_fp32": ONNX_DIR / "fruit_resnet50_fp32.onnx",
    "onnx_int8_dynamic": ONNX_DIR / "fruit_resnet50_int8_dynamic.onnx",
    "onnx_int8_static": ONNX_DIR / "fruit_resnet50_int8.onnx",
}
LABELS = {
    "keras_fp32": "TensorFlow/Keras FP32 (Eva3)",
    "onnx_fp32": "ONNX Runtime FP32",
    "onnx_int8_dynamic": "ONNX Runtime INT8 dinámica",
    "onnx_int8_static": "ONNX Runtime INT8 estática (QDQ)",
}
BATCH_SIZES = [1, 2, 4, 8, 16]


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


def load_runner(variant: str):
    path = VARIANTS[variant]
    if variant == "keras_fp32":
        import tensorflow as tf

        from pipeline.train import ResNetPreprocess  # noqa: F401

        model = tf.keras.models.load_model(path)
        return lambda x: model(x, training=False).numpy()
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return lambda x: sess.run(None, {name: x})[0]


def worker(variant: str, runs: int) -> dict:
    params = load_params()
    _, _, val_items = split_dataset(params)
    x_val, y_val = load_arrays(val_items, params["data"]["img_size"])
    rss0 = rss_mb()
    t0 = time.perf_counter()
    run = load_runner(variant)
    load_s = time.perf_counter() - t0
    rss_loaded = rss_mb()

    probs = np.concatenate([run(x_val[i : i + 8]) for i in range(0, len(x_val), 8)])
    acc = float(np.mean(probs.argmax(1) == y_val))

    for _ in range(5):
        run(x_val[:1])
    lat = []
    for i in range(runs):
        x = x_val[i % len(x_val)][None]
        t = time.perf_counter()
        run(x)
        lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)

    throughput = {}
    for bs in BATCH_SIZES:
        xb = np.concatenate([x_val] * 2)[:bs]
        run(xb)
        reps = max(3, 32 // bs)
        t = time.perf_counter()
        for _ in range(reps):
            run(xb)
        dt = (time.perf_counter() - t) / reps
        throughput[str(bs)] = {"batch_latency_ms": dt * 1000, "img_per_s": bs / dt}

    return {
        "variant": variant,
        "label": LABELS[variant],
        "size_mb": VARIANTS[variant].stat().st_size / 2**20,
        "load_time_s": load_s,
        "rss_model_mb": rss_loaded - rss0,
        "rss_peak_mb": rss_mb(),
        "accuracy_val": acc,
        "latency_ms": {
            "mean": float(lat.mean()),
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
        },
        "throughput": throughput,
        "probs": probs.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=list(VARIANTS))
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.runs)))
        return

    results = {}
    for v, path in VARIANTS.items():
        if not path.exists():
            print(f"[AVISO] {path} no existe, se omite {v}")
            continue
        out = subprocess.run(
            [sys.executable, "-m", "pipeline.benchmark", "--worker", v, "--runs", str(args.runs)],
            cwd=ROOT, capture_output=True, text=True, check=True,
            env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
        )
        results[v] = json.loads(out.stdout.strip().splitlines()[-1])
        r = results[v]
        print(f"{v:20s} size={r['size_mb']:6.1f}MB rss={r['rss_model_mb']:7.1f}MB "
              f"p50={r['latency_ms']['p50']:6.1f}ms p95={r['latency_ms']['p95']:6.1f}ms acc={r['accuracy_val']:.2f}")

    ref = np.array(results["onnx_fp32"]["probs"])
    for r in results.values():
        p = np.array(r.pop("probs"))
        r["top1_agreement_vs_fp32"] = float(np.mean(p.argmax(1) == ref.argmax(1)))
        r["max_abs_prob_diff_vs_fp32"] = float(np.max(np.abs(p - ref)))
    summary = {
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "python": platform.python_version(),
        },
        "results": results,
    }
    write_json(METRICS_DIR / "benchmark.json", summary)
    print(f"[OK] {METRICS_DIR / 'benchmark.json'}")


if __name__ == "__main__":
    main()
