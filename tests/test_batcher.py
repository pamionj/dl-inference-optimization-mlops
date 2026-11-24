"""El batching dinámico debe agrupar peticiones concurrentes en un único lote."""

from __future__ import annotations

import asyncio

import numpy as np

from app.batcher import DynamicBatcher


def test_dynamic_batcher_groups_concurrent_requests():
    calls = []

    def fake_predict(batch):
        calls.append(len(batch))
        return batch.reshape(len(batch), -1)[:, :2]

    async def scenario():
        b = DynamicBatcher(fake_predict, max_batch_size=8, max_wait_ms=50)
        await b.start()
        imgs = [np.full((2, 1, 1), i, dtype=np.float32) for i in range(8)]
        out = await asyncio.gather(*(b.submit(x) for x in imgs))
        await b.stop()
        return out

    out = asyncio.run(scenario())
    assert calls == [8]
    assert [int(o[0]) for o in out] == list(range(8))


def test_dynamic_batcher_propagates_errors():
    def boom(batch):
        raise RuntimeError("fallo")

    async def scenario():
        b = DynamicBatcher(boom, max_batch_size=4, max_wait_ms=5)
        await b.start()
        try:
            await b.submit(np.zeros((1,), dtype=np.float32))
        finally:
            await b.stop()

    try:
        asyncio.run(scenario())
    except RuntimeError as exc:
        assert "fallo" in str(exc)
    else:
        raise AssertionError("se esperaba RuntimeError")
