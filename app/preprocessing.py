"""Preprocesamiento compartido entre entrenamiento, evaluación y la API.

El modelo exportado incluye internamente ``resnet50.preprocess_input`` (RGB->BGR y
resta de la media de ImageNet), por lo que aquí solo se decodifica la imagen, se
convierte a RGB y se redimensiona a 224x224 manteniendo valores en [0, 255].
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

IMG_SIZE = 224


class InvalidImageError(ValueError):
    """La entrada no es una imagen decodificable."""


def load_image(source: str | bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    """Devuelve un tensor float32 (img_size, img_size, 3) con valores en [0, 255]."""
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("El archivo no es una imagen válida o está corrupto.") from exc
    img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)
