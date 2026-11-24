from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = ROOT / "data" / "dataset_frutas"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    return (DATASET / "platano" / "platano_001.jpg").read_bytes()


@pytest.fixture()
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (220, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()
