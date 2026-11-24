"""Etapa 4b del pipeline: evaluación por clase del modelo servido (ONNX INT8 estático).

Genera reporte de clasificación (precision/recall/F1) y matriz de confusión sobre la
partición de validación de la Eva3, comparables con las evidencias de esa evaluación.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from app.inference import InferenceEngine  # noqa: E402
from pipeline.common import (  # noqa: E402
    FIGURES_DIR,
    METRICS_DIR,
    ONNX_DIR,
    load_arrays,
    load_params,
    split_dataset,
    write_json,
)


def main() -> None:
    params = load_params()
    classes, _, val_items = split_dataset(params)
    x, y = load_arrays(val_items, params["data"]["img_size"])
    engine = InferenceEngine(ONNX_DIR / "fruit_resnet50_int8.onnx", ONNX_DIR / "model_metadata.json")
    pred = engine.predict(x).argmax(1)
    report = classification_report(y, pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    write_json(METRICS_DIR / "classification_int8.json", {"report": report, "confusion_matrix": cm.tolist()})
    (METRICS_DIR / "classification_int8.txt").write_text(
        classification_report(y, pred, target_names=classes, zero_division=0), encoding="utf-8"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30)
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de confusión ONNX INT8 (acc={np.mean(pred == y):.0%})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matriz_confusion_int8.png", dpi=150)
    print(f"[OK] accuracy={np.mean(pred == y):.4f} macro-F1={report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
