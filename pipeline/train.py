"""Etapa 1 del pipeline: reentrenamiento reproducible del modelo de la Evaluación N°3.

Arquitectura idéntica a la Eva3 (ResNet50 ImageNet + GAP + Dropout 0.4 + Dense 256
+ Dropout 0.3 + Softmax 5) con dos fases (cabezal congelado + fine-tuning de las
últimas 15 capas). Se corrige la doble normalización detectada en la Eva3
(``rescale=1/255`` seguido de ``preprocess_input``), que colapsaba las predicciones.
El registro en MLflow se realiza en la etapa ``log_mlflow``.
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from pipeline.common import (
    METRICS_DIR,
    MODELS_DIR,
    load_arrays,
    load_params,
    set_seeds,
    split_dataset,
    write_json,
)

KERAS_PATH = MODELS_DIR / "fruit_resnet50.keras"


@tf.keras.saving.register_keras_serializable(package="fruit")
class ResNetPreprocess(tf.keras.layers.Layer):
    """Equivalente serializable de ``resnet50.preprocess_input`` (modo caffe):
    RGB->BGR y resta de la media de ImageNet. Queda dentro del grafo exportado."""

    def call(self, x):
        bgr = tf.reverse(x, axis=[-1])
        return bgr - tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)


def build_model(num_classes: int, p: dict, img_size: int):
    base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_image")
    x = ResNetPreprocess(name="resnet_preprocess")(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(p["dropout_1"])(x)
    x = tf.keras.layers.Dense(p["dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(p["dropout_2"])(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="fruit_resnet50"), base


def make_datasets(x_tr, y_tr, x_val, y_val, num_classes, batch_size, seed, aug):
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(aug["rotation_deg"] / 360, seed=seed),
            tf.keras.layers.RandomZoom(aug["zoom"], seed=seed),
            tf.keras.layers.RandomTranslation(aug["shift"], aug["shift"], seed=seed),
        ],
        name="data_augmentation",
    )
    y_tr_oh = tf.one_hot(y_tr, num_classes)
    y_val_oh = tf.one_hot(y_val, num_classes)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr_oh))
        .shuffle(len(x_tr), seed=seed)
        .batch(batch_size)
        .map(lambda x, y: (augment(x, training=True), y))
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val_oh)).batch(batch_size)
    return train_ds, val_ds


def main() -> None:
    params = load_params()
    seed = params["data"]["seed"]
    set_seeds(seed)
    tf.keras.utils.set_random_seed(seed)
    img = params["data"]["img_size"]
    tp = params["train"]

    classes, train_items, val_items = split_dataset(params)
    x_tr, y_tr = load_arrays(train_items, img)
    x_val, y_val = load_arrays(val_items, img)
    print(f"[DATA] clases={classes} train={len(x_tr)} val={len(x_val)}")
    train_ds, val_ds = make_datasets(
        x_tr, y_tr, x_val, y_val, len(classes), tp["batch_size"], seed, params["augmentation"]
    )

    model, base = build_model(len(classes), tp, img)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ]

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=tp["label_smoothing"])
    t0 = time.time()
    model.compile(tf.keras.optimizers.Adam(tp["head_learning_rate"]), loss, ["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=tp["head_epochs"], callbacks=cbs, verbose=2)

    base.trainable = True
    for layer in base.layers[: len(base.layers) - tp["fine_tune_layers"]]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(tp["fine_tune_learning_rate"]), loss, ["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=tp["fine_tune_epochs"], callbacks=cbs, verbose=2)
    train_time = time.time() - t0

    probs = model.predict(x_val, verbose=0)
    val_acc = float(np.mean(np.argmax(probs, 1) == y_val))
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(KERAS_PATH)

    history = {
        k: [float(v) for v in h1.history.get(k, [])] + [float(v) for v in h2.history.get(k, [])]
        for k in ("accuracy", "val_accuracy", "loss", "val_loss")
    }
    history["phase1_epochs"] = len(h1.history["loss"])
    metrics = {
        "val_accuracy": val_acc,
        "train_samples": len(x_tr),
        "val_samples": len(x_val),
        "classes": classes,
        "train_time_s": round(train_time, 1),
        "params_total": int(model.count_params()),
    }
    write_json(METRICS_DIR / "train_metrics.json", metrics)
    write_json(METRICS_DIR / "train_history.json", history)
    print(f"[OK] val_accuracy={val_acc:.4f} -> {KERAS_PATH}")


if __name__ == "__main__":
    main()
