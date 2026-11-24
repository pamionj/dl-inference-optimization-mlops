"""Prueba de carga HTTP contra la API: compara batching dinámico activado/desactivado.

Uso: python -m pipeline.load_test --url http://127.0.0.1:8000 --mode con_batching
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import numpy as np

from pipeline.common import METRICS_DIR, ROOT


async def run(url: str, concurrency: int, total: int) -> dict:
    images = sorted((ROOT / "data" / "dataset_frutas").glob("*/*.jpg"))[:20]
    payloads = [(p.name, p.read_bytes()) for p in images]
    lat: list[float] = []
    errors = 0
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=60) as client:
        async def one(i: int) -> None:
            nonlocal errors
            name, data = payloads[i % len(payloads)]
            async with sem:
                t = time.perf_counter()
                r = await client.post(f"{url}/predict", files={"file": (name, data, "image/jpeg")})
                lat.append((time.perf_counter() - t) * 1000)
                errors += r.status_code != 200

        await asyncio.gather(*(one(i) for i in range(8)))  # calentamiento
        lat.clear()
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(total)))
        elapsed = time.perf_counter() - t0
        avg_batch = None
        for line in (await client.get(f"{url}/metrics")).text.splitlines():
            if line.startswith("fruit_api_dynamic_batch_avg_size"):
                avg_batch = float(line.split()[-1])
    arr = np.array(lat)
    return {
        "concurrency": concurrency,
        "requests": total,
        "errors": errors,
        "throughput_rps": total / elapsed,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "avg_dynamic_batch_size": avg_batch,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests", type=int, default=320)
    args = ap.parse_args()
    res = asyncio.run(run(args.url, args.concurrency, args.requests))
    path = METRICS_DIR / "load_test.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[args.mode] = res
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.mode, json.dumps(res))


if __name__ == "__main__":
    main()
