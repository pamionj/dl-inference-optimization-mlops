"""Etapa 3 del pipeline: cuantización post-entrenamiento FP32 -> INT8 con ONNX Runtime.

* Estática (QDQ, por canal) calibrada con imágenes reales de entrenamiento: es la
  variante de producción porque cuantiza pesos *y* activaciones de las Conv.
* Dinámica (solo pesos) se genera para comparación en el benchmark.
"""

from __future__ import annotations

import json

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from pipeline.common import ONNX_DIR, load_arrays, load_params, md5sum, split_dataset
from pipeline.export_onnx import FP32_PATH

PREP_PATH = ONNX_DIR / "fruit_resnet50_fp32_prep.onnx"
INT8_PATH = ONNX_DIR / "fruit_resnet50_int8.onnx"
INT8_DYNAMIC_PATH = ONNX_DIR / "fruit_resnet50_int8_dynamic.onnx"
METADATA_PATH = ONNX_DIR / "model_metadata.json"


class FruitCalibrationReader(CalibrationDataReader):
    def __init__(self, images: np.ndarray, input_name: str = "input_image"):
        self._iter = iter([{input_name: img[None, ...]} for img in images])

    def get_next(self):
        return next(self._iter, None)


def main() -> None:
    params = load_params()
    qp = params["quantize"]
    img = params["data"]["img_size"]
    classes, train_items, _ = split_dataset(params)

    rng = np.random.default_rng(params["data"]["seed"])
    idx = rng.choice(len(train_items), size=min(qp["calibration_images"], len(train_items)), replace=False)
    x_cal, _ = load_arrays([train_items[i] for i in sorted(idx)], img)

    quant_pre_process(str(FP32_PATH), str(PREP_PATH), skip_symbolic_shape=False)
    # La Softmax final se mantiene en FP32: cuantizar su salida a uint8 rompe la suma = 1 de las probabilidades
    softmax_nodes = [n.name for n in onnx.load(str(PREP_PATH)).graph.node if n.op_type == "Softmax"]
    quantize_static(
        str(PREP_PATH),
        str(INT8_PATH),
        FruitCalibrationReader(x_cal),
        quant_format=QuantFormat.QDQ,
        per_channel=qp["per_channel"],
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=softmax_nodes,
    )
    # ConvInteger en CPU solo admite pesos uint8 en ONNX Runtime
    quantize_dynamic(str(PREP_PATH), str(INT8_DYNAMIC_PATH), weight_type=QuantType.QUInt8)

    metadata = {
        "model_name": "fruit-resnet50",
        "model_version": f"eva4-{md5sum(INT8_PATH)[:8]}",
        "classes": classes,
        "input_name": "input_image",
        "input_shape": [None, img, img, 3],
        "input_range": "RGB float32 [0, 255] (preprocess_input incluido en el grafo)",
        "quantization": "INT8 estática QDQ por canal (ONNX Runtime)",
        "calibration_images": int(len(x_cal)),
        "base_architecture": "ResNet50 (ImageNet) + GAP + Dropout + Dense(256) + Softmax(5)",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    PREP_PATH.unlink(missing_ok=True)
    for p in (FP32_PATH, INT8_PATH, INT8_DYNAMIC_PATH):
        print(f"[OK] {p.name}: {p.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
