# Clasificador de Frutas con Deep Learning en Producción (ResNet50 · ONNX INT8 · FastAPI · Docker · MLOps)

[![CI/CD](https://github.com/pamionj/dl-inference-optimization-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/pamionj/dl-inference-optimization-mlops/actions/workflows/ci-cd.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-INT8%20QDQ-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![DVC](https://img.shields.io/badge/DVC-Reproducible%20Pipeline-945DD6?logo=dvc&logoColor=white)](dvc.yaml)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](pipeline/log_mlflow.py)
[![Ruff](https://img.shields.io/badge/Code%20Style-Ruff-D7FF64?logo=ruff&logoColor=black)](pyproject.toml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14iMpDRShODkPczxAgvGY8LvH-RCBW7S_?usp=sharing)

Sistema completo de visión por computador para clasificación de 5 clases de frutas (*frutilla, manzana, naranja, plátano y uva*), optimizado para inferencia en tiempo real sobre CPU y gobernado con prácticas de **MLOps** (**DVC**, **MLflow**, **GitHub Actions** y **Prometheus metrics**).

> **Informe Técnico (Evaluación Sumativa N°4 — TIEL26):**  
> 📄 [`reports/Informe_Tecnico_Evaluacion4_MachineLearning.pdf`](reports/Informe_Tecnico_Evaluacion4_MachineLearning.pdf) · [`.docx`](reports/Informe_Tecnico_Evaluacion4_MachineLearning.docx)

---

## Resultados de optimización post-entrenamiento (CPU, batch = 1)

| Variante | Tamaño (MB) | Carga (s) | RAM modelo (MB) | Latencia p50 (ms) | Latencia p95 (ms) | Throughput (b=8, img/s) | Exactitud val. | Acuerdo top-1 vs FP32 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TensorFlow / Keras FP32 (Eva3) | 138.6 | 46.70 | 642.3 | 107.8 | 126.2 | 36.5 | 70 % | 100 % |
| ONNX Runtime FP32 | 91.6 | 0.24 | 305.6 | 19.0 | 33.2 | 66.8 | 70 % | 100 % |
| ONNX Runtime INT8 dinámica | 23.1 | 0.10 | 65.1 | 21.6 | 36.7 | 36.8 | 70 % | 100 % |
| **ONNX Runtime INT8 estática (QDQ)** | **23.5** | **0.12** | **70.9** | **7.7** | **8.8** | **150.0** | **70 %** | **100 %** |

* **Modelo servido en producción:** `ONNX INT8 estático (QDQ)` — **5.9× más liviano**, **9.1× menos RAM** y **14× más rápido** en latencia p50 que Keras FP32, manteniendo **100 % de acuerdo top-1**.
* **Batching dinámico (16 clientes concurrentes, 320 peticiones, contenedor 2 CPU / 1 GB RAM):** `32.8` vs `31.3 req/s` (`+5 %`), latencia p95 `604` vs `789 ms` (`−23 %`), tamaño promedio de lote `7.63`.
* **Auditoría de calidad respecto a la fase previa:** el cuaderno de la Eva3 aplicaba `rescale=1/255` y luego `resnet50.preprocess_input` (doble normalización), colapsando las predicciones en una clase (`20 %`). En esta fase se encapsuló `ResNetPreprocess` dentro del grafo exportado (`RGB [0, 255]`) eliminando el *training/serving skew* y alcanzando `70 %` en validación (`macro-F1 = 0.72`).

---

## Arquitectura del sistema

```text
Cliente ─HTTP multipart─► FastAPI (Uvicorn)
   ├─ validación: MIME (JPEG/PNG/WEBP/BMP), tamaño ≤ 5 MB, decodificable, ≤ 16 archivos
   ├─ preprocesamiento: RGB + resize 224×224 en thread pool (sin bloquear el event loop)
   ├─ DynamicBatcher (asyncio.Queue: hasta 8 imágenes o 10 ms de espera)
   └─ InferenceEngine ─► ONNX Runtime CPU ─► models/onnx/fruit_resnet50_int8.onnx
◄─ JSON: clase, confianza, top-3, probabilidades, model_version, tiempos (ms)
```

### Endpoints principales

| Método | Ruta | Descripción |
| :--- | :--- | :--- |
| `GET` | `/` | Interfaz web ligera para subir una imagen y visualizar probabilidades |
| `GET` | `/health` | *Liveness probe* (estado del proceso y carga del modelo) |
| `GET` | `/ready` | *Readiness probe* (`200` si el modelo está listo, `503` si no) |
| `GET` | `/model/info` | Versión (`eva4-<md5>`), clases, `input_shape`, cuantización y batching |
| `POST` | `/predict` | Clasificación de una imagen (`multipart/form-data`, campo `file`) |
| `POST` | `/predict/batch` | Clasificación vectorizada de hasta 16 imágenes (`files`) |
| `GET` | `/metrics` | Métricas en formato Prometheus (`requests`, `errors`, `p50/p95`, `avg_batch_size`, por clase) |
| `GET` | `/docs` | Documentación interactiva OpenAPI / Swagger |

---

## Ejecución rápida

```bash
git clone https://github.com/pamionj/dl-inference-optimization-mlops.git
cd dl-inference-optimization-mlops
```

### Opción 1: Solo API en Docker (recomendada para despliegue)

```bash
docker compose up --build -d          # UI: http://localhost:8000 · Swagger: /docs · Métricas: /metrics
curl -F "file=@data/dataset_frutas/platano/platano_001.jpg;type=image/jpeg" http://localhost:8000/predict
```

### Opción 2: Pipeline MLOps completo (DVC + MLflow + Quality Gates)

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate   |   Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-pipeline.txt -r requirements-dev.txt

dvc repro                             # train → export_onnx → quantize → benchmark → evaluate → log_mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
ruff check app tests pipeline && pytest -v
python scripts/generate_report_eva4.py
```

---

## Estructura del repositorio

```text
├── .github/workflows/ci-cd.yml        # CI/CD: lint, pytest (18 tests + quality gates), Docker smoke test, GHCR y deploy
├── app/                               # Servicio FastAPI: main.py, inference.py, batcher.py, schemas.py, static/index.html
├── data/dataset_frutas/               # Dataset balanceado (100 imágenes reales, 20 por clase)
├── models/onnx/                       # Modelo cuantizado servido (fruit_resnet50_int8.onnx) + model_metadata.json
├── pipeline/                          # Etapas DVC: train, export_onnx, quantize, benchmark, evaluate, log_mlflow, plots, load_test
├── tests/                             # Pruebas funcionales API (12), batcher (2) y quality gates MLOps (4)
├── reports/                           # Informe técnico (.docx/.pdf), métricas JSON, figuras y capturas MLflow/Swagger
├── scripts/                           # Utilidades de dataset y generación automatizada del informe
├── dvc.yaml / dvc.lock / params.yaml  # DAG reproducible, hashes MD5 e hiperparámetros versionados
├── Dockerfile / docker-compose.yml    # Imagen slim sin TensorFlow, usuario no-root (appuser) y límites 2 CPU / 1 GB RAM
└── deep_learning_example.ipynb        # Cuaderno histórico de la fase de experimentación (Evaluación N°3)
```

---

## Trazabilidad académica y evolución del proyecto

Este proyecto constituye la **Fase 3 (Producción y MLOps)** de una progresión técnica de tres etapas desarrollada en **INACAP Sede Puente Alto** (Asignatura *Machine Learning — TIEL26*, Docente Guía: *Francisco Valdivieso Tapia*):

1. **Fase 1 (Evaluación N°2 — Línea base):** Clasificador binario de formas geométricas con Perceptrón Multicapa (`MLP 784-64-32-2`) en PyTorch.
2. **Fase 2 (Evaluación N°3 — Experimentación Deep Learning):** Migración a clasificación multiclase de frutas (`224×224×3`) con Transfer Learning y Fine-Tuning sobre **ResNet50** ([cuaderno en Google Colab](https://colab.research.google.com/drive/14iMpDRShODkPczxAgvGY8LvH-RCBW7S_?usp=sharing)).
3. **Fase 3 (Evaluación N°4 — Ingeniería ML en Producción):** Auditoría del modelo, corrección de preprocesamiento dentro del grafo, optimización **ONNX INT8**, API **FastAPI** con *dynamic batching*, empaquetado **Docker** y automatización **DVC + MLflow + GitHub Actions**.

| Criterio Eva4 | Implementación en el repositorio |
| :--- | :--- |
| **4.1.1** Optimización y recursos | `pipeline/export_onnx.py`, `pipeline/quantize.py`, `pipeline/benchmark.py`, `app/batcher.py`, `pipeline/load_test.py` |
| **4.1.2** Plan de acción | Sección 3 del informe técnico; fases ejecutables en `dvc.yaml` y `.github/workflows/ci-cd.yml` |
| **4.1.3** CI/CD y MLOps | `dvc.yaml`, `dvc.lock`, `params.yaml`, `pipeline/log_mlflow.py`, `tests/test_model_quality.py`, `.github/workflows/ci-cd.yml` |
| **4.1.4** Implementación práctica | `app/`, `Dockerfile`, `docker-compose.yml`, `tests/test_api.py`, `tests/test_batcher.py` |

---

## Autores

* **Pablo Amion** — [pamionj@gmail.com](mailto:pamionj@gmail.com)
* **Ismael Gómez**
* *Ingeniería en Informática — INACAP Sede Puente Alto*
