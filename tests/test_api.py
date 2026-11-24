"""Pruebas funcionales de la API: contratos de entrada/salida y manejo de errores."""

from __future__ import annotations


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["classes"] == ["frutilla", "manzana", "naranja", "platano", "uva"]
    assert "INT8" in info["quantization"]
    assert info["model_size_mb"] < 40


def test_predict_ok(client, sample_image_bytes):
    r = client.post("/predict", files={"file": ("platano.jpg", sample_image_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    pred = body["prediction"]
    assert pred["label"] in pred["probabilities"]
    assert abs(sum(pred["probabilities"].values()) - 1) < 1e-3
    assert len(pred["top_k"]) == 3
    assert pred["top_k"][0]["label"] == pred["label"]
    assert body["inference_ms"] > 0 and body["total_ms"] >= body["inference_ms"]
    assert body["model_version"].startswith("eva4-")


def test_predict_png(client, png_bytes):
    r = client.post("/predict", files={"file": ("rojo.png", png_bytes, "image/png")})
    assert r.status_code == 200


def test_predict_batch(client, sample_image_bytes, png_bytes):
    files = [
        ("files", ("a.jpg", sample_image_bytes, "image/jpeg")),
        ("files", ("b.png", png_bytes, "image/png")),
    ]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 200
    assert [p["filename"] for p in r.json()["predictions"]] == ["a.jpg", "b.png"]


def test_unsupported_media_type(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hola", "text/plain")})
    assert r.status_code == 415


def test_corrupt_image(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"no-es-una-imagen", "image/jpeg")})
    assert r.status_code == 400
    assert "imagen" in r.json()["detail"]


def test_empty_file(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_missing_file(client):
    assert client.post("/predict").status_code == 422


def test_too_many_files(client, png_bytes):
    files = [("files", (f"{i}.png", png_bytes, "image/png")) for i in range(17)]
    assert client.post("/predict/batch", files=files).status_code == 413


def test_metrics(client):
    text = client.get("/metrics").text
    assert "fruit_api_requests_total" in text
    assert "fruit_api_latency_ms_p95" in text
