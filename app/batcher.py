"""Batching dinámico (micro-batching) asíncrono.

Las peticiones concurrentes se encolan; un único consumidor agrupa hasta
``max_batch_size`` imágenes o espera como máximo ``max_wait_ms`` y ejecuta una sola
llamada vectorizada al runtime en un hilo, liberando el event loop de FastAPI.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import numpy as np


class DynamicBatcher:
    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray], max_batch_size: int = 8,
                 max_wait_ms: float = 10.0):
        self.predict_fn = predict_fn
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.batches_run = 0
        self.items_processed = 0

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, image: np.ndarray) -> np.ndarray:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((image, fut))
        return await fut

    @property
    def avg_batch_size(self) -> float:
        return self.items_processed / self.batches_run if self.batches_run else 0.0

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            deadline = time.perf_counter() + self.max_wait_s
            while len(items) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            batch = np.stack([img for img, _ in items])
            try:
                probs = await loop.run_in_executor(None, self.predict_fn, batch)
                for (_, fut), p in zip(items, probs, strict=False):
                    if not fut.done():
                        fut.set_result(p)
            except Exception as exc:  # propaga el error a cada petición del lote
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(exc)
            self.batches_run += 1
            self.items_processed += len(items)
