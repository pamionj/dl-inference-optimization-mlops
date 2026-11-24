"""Genera las figuras del informe Eva4 a partir de reports/metrics/*.json."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline.common import FIGURES_DIR, METRICS_DIR  # noqa: E402

COLORS = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#1B365D"]


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bench = json.loads((METRICS_DIR / "benchmark.json").read_text(encoding="utf-8"))["results"]
    names = list(bench)
    labels = [bench[n]["label"].replace(" (", "\n(").replace("Runtime ", "Runtime\n") for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [
        ("Latencia batch=1 (ms)", [bench[n]["latency_ms"]["p50"] for n in names],
         [bench[n]["latency_ms"]["p95"] for n in names]),
        ("Tamaño del artefacto (MB)", [bench[n]["size_mb"] for n in names], None),
        ("RAM adicional al cargar el modelo (MB)", [bench[n]["rss_model_mb"] for n in names], None),
    ]
    for ax, (title, vals, p95) in zip(axes, panels, strict=False):
        bars = ax.bar(range(len(names)), vals, color=COLORS[: len(names)])
        if p95:
            ax.scatter(range(len(names)), p95, color="red", zorder=3, label="p95")
            ax.legend()
        for b, v in zip(bars, vals, strict=False):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.suptitle("Impacto de la optimización post-entrenamiento (CPU)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "benchmark_optimizacion.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for n, c in zip(names, COLORS, strict=False):
        tp = bench[n]["throughput"]
        bs = [int(k) for k in tp]
        axes[0].plot(bs, [tp[k]["img_per_s"] for k in tp], marker="o", color=c, label=bench[n]["label"])
        axes[1].plot(bs, [tp[k]["batch_latency_ms"] for k in tp], marker="s", color=c, label=bench[n]["label"])
    axes[0].set_title("Throughput vs tamaño de lote")
    axes[0].set_ylabel("imágenes / s")
    axes[1].set_title("Latencia por lote vs tamaño de lote")
    axes[1].set_ylabel("ms por lote")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("tamaño de lote")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "batching_throughput.png", dpi=150)
    plt.close(fig)

    hist_path = METRICS_DIR / "train_history.json"
    if hist_path.exists():
        h = json.loads(hist_path.read_text(encoding="utf-8"))
        ep = range(1, len(h["loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, key, title in [(axes[0], "accuracy", "Exactitud"), (axes[1], "loss", "Pérdida")]:
            ax.plot(ep, h[key], marker="o", label="Entrenamiento")
            ax.plot(ep, h[f"val_{key}"], marker="s", label="Validación")
            ax.axvline(h["phase1_epochs"] + 0.5, color="red", linestyle="--", label="Inicio fine-tuning")
            ax.set_title(f"{title} por época (reentrenamiento Eva4)")
            ax.set_xlabel("Época")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "curvas_reentrenamiento.png", dpi=150)
        plt.close(fig)

    load_path = METRICS_DIR / "load_test.json"
    if load_path.exists():
        lt = json.loads(load_path.read_text(encoding="utf-8"))
        modes = list(lt)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        axes[0].bar(modes, [lt[m]["throughput_rps"] for m in modes], color=["#7f7f7f", "#1B365D"])
        axes[0].set_title(f"Throughput con {lt[modes[0]]['concurrency']} clientes concurrentes")
        axes[0].set_ylabel("peticiones / s")
        x = range(len(modes))
        axes[1].bar([i - 0.2 for i in x], [lt[m]["p50_ms"] for m in modes], width=0.4, label="p50")
        axes[1].bar([i + 0.2 for i in x], [lt[m]["p95_ms"] for m in modes], width=0.4, label="p95")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(modes)
        axes[1].set_title("Latencia extremo a extremo (HTTP)")
        axes[1].set_ylabel("ms")
        axes[1].legend()
        for ax in axes:
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "load_test_batching.png", dpi=150)
        plt.close(fig)
    print(f"[OK] figuras en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
