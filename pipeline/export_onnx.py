"""Etapa 2 del pipeline: exportación Keras (.keras) -> ONNX FP32 y validación de paridad."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

from pipeline.common import (
    METRICS_DIR,
    ONNX_DIR,
    load_arrays,
    load_params,
    split_dataset,
    write_json,
)
from pipeline.train import KERAS_PATH, ResNetPreprocess  # noqa: F401  (registra la capa)

FP32_PATH = ONNX_DIR / "fruit_resnet50_fp32.onnx"
OPSET = 17


def main() -> None:
    params = load_params()
    img = params["data"]["img_size"]
    model = tf.keras.models.load_model(KERAS_PATH)
    spec = (tf.TensorSpec((None, img, img, 3), tf.float32, name="input_image"),)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=OPSET, output_path=str(FP32_PATH))

    classes, _, val_items = split_dataset(params)
    x_val, _ = load_arrays(val_items, img)
    ref = model.predict(x_val, verbose=0)
    sess = ort.InferenceSession(str(FP32_PATH), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x_val})[0]
    max_diff = float(np.max(np.abs(ref - out)))
    top1_agree = float(np.mean(ref.argmax(1) == out.argmax(1)))
    write_json(
        METRICS_DIR / "export_parity.json",
        {"opset": OPSET, "max_abs_diff_vs_keras": max_diff, "top1_agreement_vs_keras": top1_agree},
    )
    print(f"[OK] {FP32_PATH} | max|Δ|={max_diff:.2e} | top1 agreement={top1_agree:.2%}")
    assert top1_agree == 1.0, "La exportación ONNX alteró las predicciones"


if __name__ == "__main__":
    main()
