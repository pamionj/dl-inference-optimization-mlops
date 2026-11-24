"""Genera el Informe Técnico de la Evaluación N°4 (DOCX + PDF) a partir de las métricas reales
producidas por el pipeline DVC (reports/metrics/*.json) y las figuras de reports/figures_eva4.

Uso:  python scripts/generate_report_eva4.py
      (requiere python-docx; exporta PDF vía Microsoft Word COM en Windows o LibreOffice si existe)
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"
FIG = ROOT / "reports" / "figures_eva4"
SHOT = ROOT / "reports" / "screenshots_eva4"
OUT_DOCX = ROOT / "reports" / "Informe_Tecnico_Evaluacion4_MachineLearning.docx"
ROOT_DOCX = ROOT / "Informe_Tecnico_Evaluacion4_MachineLearning.docx"

PRIMARY = RGBColor(34, 34, 34)
ACCENT = RGBColor(197, 48, 48)
GREY = RGBColor(102, 102, 102)
DARK = RGBColor(34, 34, 34)
COLAB = "https://colab.research.google.com/drive/14iMpDRShODkPczxAgvGY8LvH-RCBW7S_?usp=sharing"


def load(name: str):
    return json.loads((M / name).read_text(encoding="utf-8"))


def load_params_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        # Fallback ligero para entornos sin PyYAML (solo lectura de params.yaml plano/anidado)
        data: dict = {}
        section: str | None = None
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" ") and line.endswith(":"):
                section = line[:-1].strip()
                data[section] = {}
            elif section and ":" in line:
                k, v = [p.strip() for p in line.split(":", 1)]
                if v.lower() in {"true", "false"}:
                    val: object = v.lower() == "true"
                else:
                    try:
                        val = int(v) if "." not in v and "e" not in v.lower() else float(v)
                    except ValueError:
                        val = v
                data[section][k] = val
        return data


bench = load("benchmark.json")
R = bench["results"]
train = load("train_metrics.json")
parity = load("export_parity.json")
load_test = load("load_test.json")
cls = load("classification_int8.json")
sample = load("sample_prediction.json")
params = load_params_yaml(ROOT / "params.yaml")
meta = json.loads((ROOT / "models" / "onnx" / "model_metadata.json").read_text(encoding="utf-8"))
ks, st, dy, of = R["keras_fp32"], R["onnx_int8_static"], R["onnx_int8_dynamic"], R["onnx_fp32"]
lb, ln = load_test["con_batching"], load_test["sin_batching"]

TEMPLATE = ROOT / "reports" / "templates" / "INACAP_Carta_Vertical.docx"
FONT = "Myriad Pro"
BULLET_NUM_ID = 4  # viñeta "-" definida en la plantilla INACAP

doc = Document(str(TEMPLATE))
body = doc.element.body
paras = doc.paragraphs
COVER = {p.text.strip(): p for p in paras if p.text.strip()}
# Se elimina el contenido de ejemplo posterior a la portada (se conservan portada, secciones y estilos)
start = next(i for i, p in enumerate(paras) if p.text.strip() == "Introducción")
for p in paras[start:]:
    body.remove(p._p)
for name in ["Normal", "Heading 1", "Heading 2", "Heading 3", "List Paragraph"]:
    sty = doc.styles[name]
    sty.font.name = FONT
    rf = sty.element.get_or_add_rPr().get_or_add_rFonts()
    for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(k), FONT)
    for k in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rf.attrib.pop(qn(k), None)
doc.styles["Normal"].font.size = Pt(11)
if "Table Grid" not in [x.name for x in doc.styles]:
    doc.styles.element.append(copy.deepcopy(Document().styles["Table Grid"].element))


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def set_row_repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))
    trPr.append(OxmlElement("w:cantSplit"))


def set_row_cant_split(row):
    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def add_field(run, field_name: str):
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = f" {field_name} "
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")
    fldChar3 = OxmlElement("w:fldChar")
    fldChar3.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)


TOC_HEADINGS: list[tuple[str, str]] = []


def h(text: str, level: int = 1, numbered: bool = True, toc_label: str | None = None):
    """Títulos con la numeración automática de la plantilla INACAP (se quita la numeración manual)."""
    clean = re.sub(r"^\d+(\.\d+)*\.?\s+", "", text)
    hd = doc.add_heading(clean, level=level)
    hd.paragraph_format.space_after = Pt(4)
    hd.paragraph_format.keep_with_next = True
    if not numbered:
        hd._p.get_or_add_pPr().get_or_add_numPr().get_or_add_numId().val = 0
    if level == 1 and toc_label is not None:
        TOC_HEADINGS.append((toc_label, clean))
    return hd


def para(text, bold_prefix=None, italic=False, size=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.color.rgb = PRIMARY
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    return p


def bullets(items):
    for it in items:
        p = doc.add_paragraph(style="List Paragraph")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        numPr = p._p.get_or_add_pPr().get_or_add_numPr()
        numPr.get_or_add_ilvl().val = 0
        numPr.get_or_add_numId().val = BULLET_NUM_ID
        p.paragraph_format.space_after = Pt(2)
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def table(header, rows, widths=None, font=8.5):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_row_repeat_header(t.rows[0])
    for i, txt in enumerate(header):
        c = t.rows[0].cells[i]
        shade(c, "E30613")
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = c.paragraphs[0].add_run(txt)
        r.bold, r.font.size, r.font.color.rgb = True, Pt(font), RGBColor(255, 255, 255)
    for k, row in enumerate(rows):
        tbl_row = t.add_row()
        set_row_cant_split(tbl_row)
        cells = tbl_row.cells
        for i, txt in enumerate(row):
            if k % 2:
                shade(cells[i], "FCE8E8")
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = cells[i].paragraphs[0].add_run(str(txt))
            r.font.size = Pt(font)
    if widths:
        t.autofit = False
        for i, w in enumerate(widths):
            t.columns[i].width = Inches(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path, caption, width=6.0):
    if not Path(path).exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    pic_p = doc.paragraphs[-1]
    pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic_p.paragraph_format.space_after = Pt(2)
    pic_p.paragraph_format.keep_with_next = True  # Evita pie de figura huérfano en la página siguiente
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(caption)
    r.italic, r.font.size, r.font.color.rgb = True, Pt(8.5), GREY


def code(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(6)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F3F4F6")
    pPr.append(shd)
    r = p.add_run(text)
    r.font.name, r.font.size = "Courier New", Pt(8)


def f1(x, d=1):
    return f"{x:.{d}f}"


def pct(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f} %"


# ---------------------------------------------------------------- Portada (plantilla INACAP)
def set_text(p, text):
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    p.runs[0].text = text


def cover_field(label, value):
    p = COVER[label]
    set_text(p, label + " ")
    p.runs[0].bold = True
    p.add_run(value).font.name = FONT


set_text(COVER["TITULO PORTADA TRABAJO"],
         "INFORME TÉCNICO EVALUACIÓN N°4: IMPLEMENTACIÓN DEL CLASIFICADOR DE FRUTAS RESNET50")
set_text(COVER["Ejemplo de subtítulo"],
         "Optimización post-entrenamiento (ONNX INT8), API FastAPI, Docker y MLOps (DVC, MLflow, GitHub Actions)")
cover_field("NOMBRE:", "Pablo Amion & Ismael Gómez")
cover_field("CARRERA:", "Ingeniería en Informática — INACAP Sede Puente Alto")
cover_field("ASIGNATURA:", "Machine Learning (TIEL26) — Evaluación Sumativa N°4")
cover_field("PROFESOR:", "Francisco Valdivieso Tapia")
cover_field("FECHA:", "Noviembre 2026")
# La portada debe caber en una página: se quitan párrafos vacíos sobrantes tras FECHA (salvo el salto de sección)
fecha = COVER["FECHA:"]._p
sib = fecha.getnext()
while sib is not None and sib.find(qn("w:pPr") + "/" + qn("w:sectPr")) is None:
    nxt = sib.getnext()
    if sib.tag == qn("w:p") and not "".join(sib.itertext()).strip():
        body.remove(sib)
    sib = nxt

# Actualiza el recuadro del Área Académica en la portada conforme a la pauta (Informática y Telecomunicaciones)
hdr_el = doc.sections[0].header._element
for ext in hdr_el.iter(qn("wp:extent")):
    if ext.get("cx") == "1217930":
        ext.set("cx", "1520000")
for aext in hdr_el.iter(qn("a:ext")):
    if aext.get("cx") == "1217930":
        aext.set("cx", "1520000")
for pos in hdr_el.iter(qn("wp:posOffset")):
    if pos.text == "242570":
        pos.text = "55000"
for r_el in hdr_el.iter(qn("w:r")):
    texts = list(r_el.iter(qn("w:t")))
    if any(t.text in {"ADMINISTRACIÓN", "Y NEGOCIOS"} for t in texts):
        rPr = r_el.find(qn("w:rPr"))
        if rPr is None:
            rPr = OxmlElement("w:rPr")
            r_el.insert(0, rPr)
        for tag, val in [("w:sz", "14"), ("w:szCs", "14")]:
            sz = rPr.find(qn(tag))
            if sz is None:
                sz = OxmlElement(tag)
                rPr.append(sz)
            sz.set(qn("w:val"), val)
        if rPr.find(qn("w:b")) is None:
            rPr.append(OxmlElement("w:b"))
        for t in texts:
            if t.text == "ADMINISTRACIÓN":
                t.text = "INFORMÁTICA Y"
            elif t.text == "Y NEGOCIOS":
                t.text = "TELECOMUNICACIONES"

# Pie de página con numeración dinámica en el cuerpo del documento (sección 1)
ftr = doc.sections[1].footer
for p_old in list(ftr.paragraphs)[1:]:
    p_old._p.getparent().remove(p_old._p)
fp = ftr.paragraphs[0]
fp.text = ""
fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
fp.paragraph_format.space_before = Pt(0)
fp.paragraph_format.space_after = Pt(0)
for txt, fld in [
    ("Página ", None),
    (None, "PAGE"),
    (" de ", None),
    (None, "NUMPAGES"),
    ("   |   INACAP Sede Puente Alto — Machine Learning (TIEL26)", None),
]:
    run = fp.add_run(txt or "")
    run.font.name = FONT
    run.font.size = Pt(8.5)
    run.font.color.rgb = GREY
    if fld:
        add_field(run, fld)

# ---------------------------------------------------------------- Resumen + Índice
h("Resumen Ejecutivo", numbered=False, toc_label="Resumen Ejecutivo")
para(
    "La Evaluación N°3 entregó un clasificador de cinco frutas (frutilla, manzana, naranja, plátano y uva) basado en "
    "Transfer Learning con ResNet50 y Fine-Tuning de las últimas 15 capas. Esta Evaluación N°4 toma ese modelo como "
    "punto de partida y lo lleva a producción. Primero se auditó el modelo de la Eva3 y se detectó que la matriz de "
    "confusión concentraba todas las predicciones en una sola clase (exactitud 20 %) debido a una doble normalización "
    "(rescale=1/255 seguido de preprocess_input de ResNet). Se corrigió integrando el preprocesamiento dentro del grafo "
    f"del modelo y se reentrenó de forma reproducible con DVC, alcanzando {pct(train['val_accuracy'])} de exactitud en la "
    "misma partición de validación de la Eva3."
)
para(
    "Luego se aplicó optimización post-entrenamiento: exportación a ONNX (opset 17) y cuantización INT8 estática "
    "QDQ por canal con calibración sobre imágenes reales. El modelo servido pasa de "
    f"{f1(ks['size_mb'])} MB a {f1(st['size_mb'])} MB (−{(1 - st['size_mb'] / ks['size_mb']) * 100:.0f} %), la RAM adicional al "
    f"cargarlo baja de {f1(ks['rss_model_mb'])} MB a {f1(st['rss_model_mb'])} MB y la latencia p50 por imagen cae de "
    f"{f1(ks['latency_ms']['p50'])} ms a {f1(st['latency_ms']['p50'])} ms "
    f"({ks['latency_ms']['p50'] / st['latency_ms']['p50']:.0f}× más rápido), manteniendo 100 % de acuerdo top-1 con el "
    "modelo FP32 y la misma exactitud."
)
para(
    "El modelo se expone con una API FastAPI (validación de entradas, manejo de excepciones, batching dinámico, "
    "métricas de operación), se empaqueta en una imagen Docker sin TensorFlow que corre como usuario no root, y se "
    "gobierna con un pipeline MLOps: DVC versiona datos, parámetros y artefactos; MLflow registra experimentos; "
    "GitHub Actions ejecuta lint, 18 pruebas automáticas (incluidos quality gates de exactitud y latencia), build y "
    "smoke test del contenedor, publicación en GHCR y despliegue continuo opcional."
)

h("Índice", 1, numbered=False)
TOC_PARAGRAPHS: list[tuple[str, object]] = []
TOC_ITEMS = [
    ("Resumen Ejecutivo", "Resumen Ejecutivo"),
    ("1 Contexto y nexo con la Evaluación N°3", "1 Contexto y nexo con la Evaluación N°3"),
    ("2 Criterio 4.1.1 — Optimización post-entrenamiento y uso de recursos",
     "2 Criterio 4.1.1 — Optimización post-entrenamiento y uso de recursos"),
    ("3 Criterio 4.1.2 — Plan de acción para la implementación",
     "3 Criterio 4.1.2 — Plan de acción para la implementación"),
    ("4 Criterio 4.1.3 — Integración continua, MLOps y despliegue",
     "4 Criterio 4.1.3 — Integración continua, MLOps y despliegue"),
    ("5 Criterio 4.1.4 — Implementación práctica: API FastAPI y Docker",
     "5 Criterio 4.1.4 — Implementación práctica: API FastAPI y Docker"),
    ("6 Resultados, discusión y limitaciones", "6 Resultados, discusión y limitaciones"),
    ("7 Conclusiones", "7 Conclusiones"),
    ("8 Referencias (APA 7)", "8 Referencias (APA 7)"),
    ("Anexo A — Reproducción paso a paso", "Anexo A — Reproducción paso a paso"),
]
for idx_i, (label, _) in enumerate(TOC_ITEMS):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    r_lbl = p.add_run(label)
    r_lbl.font.size = Pt(10.5)
    p.add_run("\t")
    r_pg = p.add_run(f"Pág. {{TOC_{idx_i}}}")
    r_pg.font.size = Pt(10.5)
    r_pg.font.color.rgb = GREY
    TOC_PARAGRAPHS.append((f"{{TOC_{idx_i}}}", p))
doc.add_page_break()

# ---------------------------------------------------------------- 1
h("1. Contexto y nexo con la Evaluación N°3", toc_label="1 Contexto y nexo con la Evaluación N°3")
para(
    "El proyecto se desarrolla en tres fases encadenadas: la Eva2 construyó un MLP para formas geométricas; la Eva3 "
    "migró a Deep Learning con una CNN ResNet50 para clasificar frutas; y la Eva4 (este informe) implementa ese "
    "modelo como un servicio de inferencia eficiente, reproducible y desplegable. Se conservan el dataset (100 "
    "imágenes, 20 por clase), la partición 80/20, la entrada 224×224×3 y la arquitectura documentada en el informe de "
    "la Eva3 (ResNet50 ImageNet + GlobalAveragePooling2D + Dropout 0.4 + Dense 256 ReLU + Dropout 0.3 + Softmax 5)."
)
h("1.1 Auditoría del modelo de la Eva3", 2)
para(
    "Antes de optimizar se verificó la calidad del punto de partida. La matriz de confusión de la Eva3 "
    "(outputs/matriz_confusion_final.png) mostraba las 20 imágenes de validación predichas en una única clase. Al "
    "inspeccionar el cuaderno deep_learning_example.ipynb (celdas 7 y 10) se confirmó la causa raíz: el "
    "ImageDataGenerator aplicaba rescale=1.0/255.0 llevando los píxeles a [0, 1] y luego el modelo invocaba "
    "preprocess_input de ResNet50, que espera valores en [0, 255]; las activaciones quedaban prácticamente constantes "
    "y el clasificador colapsaba (además de presentar una pequeña divergencia entre el cabezal con BatchNormalization "
    "del cuaderno y el doble Dropout documentado en el informe). Optimizar un modelo defectuoso no tiene sentido, por "
    "lo que la primera fase del plan fue corregirlo:"
)
bullets([
    ("Preprocesamiento dentro del grafo: ", "capa serializable ResNetPreprocess (RGB→BGR y resta de la media de "
     "ImageNet). La API y el modelo ONNX reciben píxeles RGB en [0, 255], eliminando el riesgo de desalinear "
     "entrenamiento e inferencia (training/serving skew)."),
    ("Entrenamiento reproducible: ", "pipeline/train.py con semillas fijas, hiperparámetros en params.yaml y "
     "las dos fases de la Eva3 (cabezal congelado y Fine-Tuning de las últimas 15 capas con LR 1e-5)."),
    ("Resultado: ", f"exactitud de validación {pct(train['val_accuracy'])} frente al 20 % efectivo de la Eva3, "
     f"macro-F1 {cls['report']['macro avg']['f1-score']:.2f}."),
])
table(
    ["Aspecto", "Evaluación N°3", "Evaluación N°4"],
    [
        ["Formato del modelo", "Keras (.keras), TensorFlow 2.x", "ONNX opset 17, INT8 QDQ por canal"],
        ["Preprocesamiento", "Fuera del modelo (doble normalización)", "Dentro del grafo (RGB 0-255)"],
        ["Exactitud validación", "20 % (colapso a una clase)", pct(train["val_accuracy"])],
        ["Runtime de inferencia", "TensorFlow (≈ 640 MB RAM)", f"ONNX Runtime ({f1(st['rss_model_mb'])} MB RAM)"],
        ["Ejecución", "Cuaderno Colab", "API REST FastAPI en contenedor Docker"],
        ["Reproducibilidad", "Manual", "DVC (dvc repro) + MLflow + CI/CD"],
    ],
    widths=[1.6, 2.4, 2.5],
)

# ---------------------------------------------------------------- 2
h("2. Criterio 4.1.1 — Optimización post-entrenamiento y uso de recursos",
  toc_label="2 Criterio 4.1.1 — Optimización post-entrenamiento y uso de recursos")
para(
    "Se evaluaron cuatro variantes del mismo modelo con un benchmark automatizado (pipeline/benchmark.py). Cada "
    "variante se ejecuta en un subproceso aislado para medir la memoria sin interferencias; se registran tamaño en "
    "disco, tiempo de carga, RSS (RAM residente y pico de memoria), latencia de 100 inferencias de una imagen (media, "
    "p50, p95 y p99) con calentamiento previo, throughput para batches de 1 a 16 imágenes, exactitud y concordancia "
    f"con FP32. Hardware objetivo: CPU x86_64 con {bench['hardware']['logical_cores']} núcleos lógicos y "
    f"{bench['hardware']['ram_gb']} GB RAM, sin GPU dedicada (0 MB de VRAM requerida; escenario de despliegue edge o "
    "servidor económico)."
)
h("2.1 Técnicas aplicadas", 2)
bullets([
    ("Exportación a ONNX (tf2onnx, opset 17): ", "formato abierto e independiente del framework; ONNX Runtime "
     "aplica fusiones de grafo (Conv+BatchNorm+ReLU) y optimizaciones ORT_ENABLE_ALL. Paridad verificada: "
     f"máx |Δp| = {parity['max_abs_diff_vs_keras']:.1e}, acuerdo top-1 = {pct(parity['top1_agreement_vs_keras'])}."),
    ("Cuantización INT8 estática (QDQ, por canal): ", f"pesos y activaciones FP32→INT8 calibrados con "
     f"{params['quantize']['calibration_images']} imágenes reales de entrenamiento (MinMax). La Softmax final se "
     "mantiene en FP32 para que las probabilidades sumen exactamente 1."),
    ("Cuantización INT8 dinámica (comparativa): ", "solo pesos en INT8, activaciones cuantizadas en tiempo de "
     "ejecución; no requiere calibración pero es más lenta en convoluciones."),
    ("Batching dinámico en el servidor: ", f"las peticiones concurrentes se agrupan hasta "
     f"{params['serving']['max_batch_size']} imágenes o {params['serving']['max_wait_ms']} ms y se ejecutan en una "
     "sola llamada al modelo."),
    ("TensorRT / pruning: ", "TensorRT requiere GPU NVIDIA (VRAM = N/A en el entorno objetivo CPU). El mismo "
     "artefacto ONNX puede compilarse con trtexec o ejecutarse con TensorrtExecutionProvider en un nodo GPU sin "
     "cambiar la API. Pruning se descartó porque la reducción de 5.9× ya se obtiene con INT8 sin reentrenar y la "
     "dispersión no estructurada no acelera la inferencia en CPU con ONNX Runtime."),
])
h("2.2 Resultados del benchmark", 2)
rows = []
for k in ["keras_fp32", "onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static"]:
    v = R[k]
    rows.append([
        v["label"], f1(v["size_mb"]), f1(v["load_time_s"], 2), f1(v["rss_model_mb"]),
        f1(v["latency_ms"]["p50"]), f1(v["latency_ms"]["p95"]), f1(v["throughput"]["8"]["img_per_s"]),
        pct(v["accuracy_val"]), pct(v["top1_agreement_vs_fp32"]),
    ])
table(
    ["Variante", "Tamaño (MB)", "Carga (s)", "RAM (MB)", "p50 (ms)", "p95 (ms)", "img/s (b=8)", "Exactitud",
     "Top-1 = FP32"],
    rows, widths=[1.35, 0.7, 0.55, 0.6, 0.55, 0.55, 0.7, 0.75, 0.75], font=7.5,
)
figure(FIG / "benchmark_optimizacion.png", "Figura 1. Latencia, tamaño y RAM de cada variante (menor es mejor).")
para(
    f"La variante INT8 estática es la elegida para producción: reduce el tamaño {ks['size_mb'] / st['size_mb']:.1f}×, la "
    f"RAM del modelo {ks['rss_model_mb'] / st['rss_model_mb']:.1f}× (y el pico total de proceso de "
    f"{f1(ks['rss_peak_mb'])} MB a {f1(st['rss_peak_mb'])} MB, con 0 MB de VRAM), el tiempo de carga de "
    f"{f1(ks['load_time_s'])} s a {f1(st['load_time_s'], 2)} s (arranques en frío casi instantáneos, clave para "
    f"autoescalado) y la latencia p95 de {f1(ks['latency_ms']['p95'])} ms a {f1(st['latency_ms']['p95'])} ms. La "
    f"exactitud se mantiene en {pct(st['accuracy_val'])} y todas las predicciones top-1 coinciden con FP32; la "
    f"diferencia máxima de probabilidad ({st['max_abs_prob_diff_vs_fp32']:.2f}) es el costo esperable de la "
    "cuantización y no altera la decisión. La INT8 dinámica reduce tamaño y RAM pero es más lenta que ONNX FP32, "
    "porque recalcula la escala de las activaciones en cada convolución; por eso se prefiere la estática."
)
figure(FIG / "batching_throughput.png", "Figura 2. Throughput (imágenes/s) según tamaño de batch para cada variante.")
h("2.3 Batching dinámico bajo carga concurrente", 2)
para(
    f"Se ejecutó una prueba de carga (pipeline/load_test.py) con {lb['concurrency']} clientes concurrentes y "
    f"{lb['requests']} peticiones contra dos contenedores idénticos limitados a 2 CPU y 1 GB de RAM, con y sin "
    "batching dinámico:"
)
table(
    ["Modo", "Throughput (req/s)", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Batch medio", "Errores"],
    [[m.replace("_", " "), f1(v["throughput_rps"]), f1(v["p50_ms"], 0), f1(v["p95_ms"], 0), f1(v["p99_ms"], 0),
      f1(v["avg_dynamic_batch_size"], 2), v["errors"]] for m, v in [("sin_batching", ln), ("con_batching", lb)]],
    widths=[1.2, 1.1, 0.7, 0.7, 0.7, 0.9, 0.7],
)
figure(FIG / "load_test_batching.png", "Figura 3. Prueba de carga con 16 clientes: throughput y latencia HTTP.")
para(
    f"El batching agrupa en promedio {f1(lb['avg_dynamic_batch_size'], 1)} imágenes por llamada, aumenta el throughput "
    f"{(lb['throughput_rps'] / ln['throughput_rps'] - 1) * 100:.0f} % y reduce la latencia p95 "
    f"{(1 - lb['p95_ms'] / ln['p95_ms']) * 100:.0f} %. La ganancia es moderada porque, con el modelo INT8 (≈ 8 ms), el "
    "cuello de botella pasa a ser la decodificación y el redimensionamiento de las fotografías en 2 CPU; en GPU el "
    "beneficio del batching es mucho mayor. La latencia extremo a extremo de una petición individual se mantiene en "
    "decenas de milisegundos (véase la sección 5.5 y la Figura 6)."
)

# ---------------------------------------------------------------- 3
h("3. Criterio 4.1.2 — Plan de acción para la implementación",
  toc_label="3 Criterio 4.1.2 — Plan de acción para la implementación")
para(
    "La implementación se organizó en fases secuenciales con entradas, actividades, salidas verificables y criterio de "
    "salida; cada fase solo comienza cuando la anterior cumple su criterio. Las fases 1 a 4 están automatizadas como "
    "etapas del DAG de DVC (dvc.yaml) y las fases 5 a 7 en GitHub Actions, de modo que el plan es ejecutable y no solo "
    "documental."
)
table(
    ["Fase", "Entradas", "Actividades y herramientas", "Salida / evidencia", "Criterio de salida"],
    [
        ["0. Auditoría Eva3", "Notebook Eva3, matriz de confusión",
         "Revisar métricas, detectar doble normalización, definir línea base",
         "Diagnóstico documentado (sección 1.1)", "Causa raíz identificada"],
        ["1. Reentrenamiento", "Dataset (100 img), params.yaml",
         "Preprocesamiento en grafo, semillas fijas (TensorFlow, DVC train)",
         "models/fruit_resnet50.keras, train_metrics.json", "Exactitud ≥ 0.65 (gate)"],
        ["2. Exportación", "Modelo .keras",
         "Conversión a ONNX opset 17 y paridad (tf2onnx, ONNX Runtime)",
         "fruit_resnet50_fp32.onnx, export_parity.json", "Top-1 = 100 %, |Δp| < 1e-3"],
        ["3. Optimización", "Modelo ONNX FP32, 40 img cal.",
         "Cuantización INT8 estática/dinámica (onnxruntime.quantization)",
         "fruit_resnet50_int8.onnx, model_metadata.json", "Tamaño −75 %, top-1 = FP32"],
        ["4. Validación", "Variantes FP32/INT8, dataset val.",
         "Benchmark RAM/latencia/throughput y reporte (psutil, sklearn, MLflow)",
         "benchmark.json, figuras, run MLflow", "p95 ≤ 400 ms, exactitud ≥ 0.65"],
        ["5. Servicio API", "Modelo INT8, metadata",
         "Endpoints, contratos Pydantic, batching, métricas (FastAPI, Uvicorn)",
         "app/, 18 pruebas pytest", "18 pruebas en verde"],
        ["6. Empaquetado", "app/, requirements-api.txt, INT8",
         "Imagen slim sin TF, usuario no root, healthcheck (Docker, Compose)",
         "Imagen fruit-classifier-api", "Smoke test HTTP OK"],
        ["7. CI/CD y deploy", "Repo Git, ci-cd.yml, secrets",
         "Lint, tests, build, GHCR, deploy y monitoreo (GitHub Actions, Render)",
         "Workflow ci-cd.yml, imagen con tag SHA", "Pipeline verde en main"],
    ],
    widths=[1.05, 1.15, 1.75, 1.45, 1.1], font=7.5,
)
h("3.1 Riesgos y mitigaciones", 2)
table(
    ["Riesgo", "Mitigación implementada"],
    [
        ["Pérdida de exactitud al cuantizar", "Calibración con imágenes reales, cuantización por canal, Softmax en "
         "FP32 y quality gate que compara contra FP32"],
        ["Training/serving skew", "Preprocesamiento dentro del grafo; la API solo decodifica y redimensiona"],
        ["Regresiones de latencia", "Gate de latencia p95 en pytest y smoke test (< 1 s) en el contenedor"],
        ["Artefacto no trazable", "Hash MD5 del modelo en dvc.lock verificado por una prueba; versión del modelo "
         "derivada del hash y devuelta en cada respuesta"],
        ["Saturación del servicio", "Límites de tamaño y cantidad de archivos, batching dinámico, límites de CPU/RAM "
         "en Compose, healthcheck y reinicio automático"],
        ["Dataset pequeño (100 imágenes)", "Plan de mejora continua: recolectar más datos y reentrenar con dvc repro"],
    ],
    widths=[2.0, 4.5],
)
h("3.2 Cronograma propuesto", 2)
table(
    ["Semana", "Fases", "Hito"],
    [["1", "0–1", "Modelo corregido y reproducible"], ["2", "2–4", "Modelo INT8 validado y registrado en MLflow"],
     ["3", "5–6", "API contenedorizada con pruebas en verde"], ["4", "7", "Pipeline CI/CD y despliegue en la nube"]],
    widths=[0.8, 1.0, 4.7],
)

# ---------------------------------------------------------------- 4
h("4. Criterio 4.1.3 — Integración continua, MLOps y despliegue",
  toc_label="4 Criterio 4.1.3 — Integración continua, MLOps y despliegue")
h("4.1 Versionado de datos, parámetros y modelos con DVC", 2)
para(
    "dvc.yaml define el pipeline como un DAG de seis etapas. DVC calcula hashes de cada dependencia y salida, los "
    "registra en dvc.lock y solo reejecuta las etapas afectadas por un cambio (dvc repro). Así cualquier versión del "
    "modelo queda ligada al commit exacto de código, parámetros y datos que la produjo."
)
code(
    "train ─► export_onnx ─► quantize ─► benchmark ─► log_mlflow\n"
    "                              └──► evaluate ─────┘\n\n"
    "params.yaml  → hiperparámetros (data, augmentation, train, quantize, serving)\n"
    "dvc.lock     → MD5 de datos, código, modelos y métricas de cada etapa"
)
h("4.2 Seguimiento de experimentos con MLflow", 2)
para(
    "La etapa log_mlflow registra en MLflow (backend SQLite local; configurable con MLFLOW_TRACKING_URI para un "
    "servidor remoto) los 24 parámetros de params.yaml, las curvas por época, la paridad ONNX, las métricas del "
    "benchmark de cada variante y, como artefactos, el modelo INT8, su metadata y los JSON de métricas. Cada run se "
    "nombra con el hash del modelo (p. ej. resnet50-int8-" + meta["model_version"].split("-")[1] + "), lo que permite "
    "comparar ejecuciones y volver a una versión anterior."
)
figure(SHOT / "mlflow_run.png", "Figura 4. Run de MLflow con parámetros, tags y métricas del modelo servido.", 5.8)
figure(SHOT / "mlflow_metrics.png", "Figura 5. Métricas por época y de paridad registradas en MLflow.", 5.8)
h("4.3 Pipeline CI/CD con GitHub Actions", 2)
para("El workflow .github/workflows/ci-cd.yml se ejecuta en cada pull request y en cada push a main:")
table(
    ["Job", "Disparador", "Pasos", "Falla si…"],
    [
        ["test", "PR y push", "Instala requirements-dev, ruff check, pytest -v (18 pruebas), sube reporte JUnit",
         "Hay errores de estilo, una prueba funcional falla o no se cumple un quality gate"],
        ["docker", "Tras test", "docker build, levanta el contenedor, espera /ready, prueba /model/info, /predict "
         "con una imagen real y verifica 415 para un archivo no imagen", "La imagen no construye, no responde o tarda "
         "> 1 s"],
        ["publish (en docker)", "Push a main", "Publica ghcr.io/<owner>/fruit-classifier-api con tags SHA y latest",
         "Falla la autenticación o el push"],
        ["deploy", "Tras docker en main", "Dispara el deploy hook de Render (secreto RENDER_DEPLOY_HOOK); se omite si "
         "no está configurado", "El hook responde con error"],
    ],
    widths=[0.9, 0.9, 2.8, 1.9], font=8,
)
h("4.4 Pruebas automáticas y quality gates", 2)
table(
    ["Grupo", "Pruebas", "Qué garantiza"],
    [
        ["API (12)", "health, ready, model/info, predict JPEG/PNG, batch, métricas, 415, 400, 422, exceso de archivos",
         "Contrato de entrada/salida y manejo de excepciones"],
        ["Batching (2)", "Agrupa 8 peticiones concurrentes en un batch; propaga errores a todas las peticiones",
         "Correctitud del batcher asíncrono"],
        ["Calidad del modelo (4)", f"Exactitud ≥ {params['serving']['accuracy_gate']}, latencia p95 ≤ "
         f"{params['serving']['latency_p95_gate_ms']} ms, probabilidades válidas, MD5 del modelo = dvc.lock",
         "Que un modelo peor, más lento o no trazable no llegue a producción"],
    ],
    widths=[1.2, 3.1, 2.2],
)
h("4.5 Estrategia de despliegue, monitoreo y rollback", 2)
bullets([
    ("Artefacto inmutable: ", "cada imagen se etiqueta con el SHA del commit; el rollback consiste en redesplegar el "
     "tag anterior."),
    ("Despliegue: ", "la misma imagen corre localmente (docker compose up), en un servidor con Docker o en un PaaS "
     "de contenedores (Render, Cloud Run, Azure Container Apps) mediante el job deploy."),
    ("Monitoreo: ", "/metrics expone en formato Prometheus las peticiones, errores, imágenes clasificadas, "
     "predicciones por clase, predicciones de baja confianza, latencia p50/p95 y tamaño medio de batch. Un aumento "
     "de predicciones de baja confianza o un cambio en la distribución por clase indica deriva de datos y dispara "
     "un reentrenamiento (dvc repro)."),
    ("Salud: ", "/health (liveness) y /ready (readiness) usados por el HEALTHCHECK de Docker y por el orquestador."),
])

# ---------------------------------------------------------------- 5
h("5. Criterio 4.1.4 — Implementación práctica: API FastAPI y Docker",
  toc_label="5 Criterio 4.1.4 — Implementación práctica: API FastAPI y Docker")
h("5.1 Arquitectura del servicio", 2)
code(
    "Cliente ─HTTP multipart─► FastAPI (Uvicorn)\n"
    "   ├─ validación: tipo MIME, tamaño ≤ 5 MB, imagen decodificable, ≤ 16 archivos\n"
    "   ├─ preprocesamiento: RGB + resize 224×224 (thread pool, no bloquea el event loop)\n"
    "   ├─ DynamicBatcher (asyncio.Queue: hasta 8 imágenes o 10 ms)\n"
    "   └─ InferenceEngine ─► ONNX Runtime CPU ─► fruit_resnet50_int8.onnx\n"
    "◄─ JSON: clase, confianza, top-3, probabilidades, versión, tiempos"
)
h("5.2 Endpoints", 2)
table(
    ["Método", "Ruta", "Descripción"],
    [
        ["GET", "/", "Interfaz web para subir una imagen y ver la predicción"],
        ["GET", "/health", "Liveness: proceso activo y modelo cargado"],
        ["GET", "/ready", "Readiness: 200 si el modelo está listo, 503 si no"],
        ["GET", "/model/info", "Versión, clases, entrada esperada, cuantización y configuración de batching"],
        ["POST", "/predict", "Clasifica una imagen (campo file)"],
        ["POST", "/predict/batch", "Clasifica hasta 16 imágenes en una sola llamada al modelo (campo files)"],
        ["GET", "/metrics", "Métricas de operación en formato Prometheus"],
        ["GET", "/docs", "Documentación OpenAPI/Swagger generada automáticamente"],
    ],
    widths=[0.7, 1.3, 4.5],
)
h("5.3 Contrato de entrada y salida", 2)
para("Entrada: multipart/form-data con una imagen JPEG, PNG, BMP o WEBP. Ejemplo de respuesta real de /predict:")
code(json.dumps(sample, indent=2, ensure_ascii=False))
h("5.4 Manejo de excepciones", 2)
table(
    ["Situación", "Código HTTP", "Respuesta"],
    [
        ["Tipo de archivo no soportado (p. ej. text/plain)", "415", "Mensaje con los tipos permitidos"],
        ["Archivo vacío, imagen corrupta o lista vacía", "400", "\"El archivo no es una imagen válida o está corrupto.\""],
        ["Archivo mayor a MAX_UPLOAD_MB", "413", "Límite configurado"],
        ["Falta el campo file / formato inválido", "422", "Detalle de validación de Pydantic"],
        ["Más de MAX_FILES_PER_REQUEST archivos", "413", "Límite configurado"],
        ["Modelo no cargado", "503", "Servicio no disponible (readiness falla)"],
        ["Error inesperado", "500", "Mensaje genérico; detalle solo en logs y contador de errores"],
        ["Confianza < 0.5", "200", "low_confidence = true para revisión humana"],
    ],
    widths=[2.6, 0.9, 3.0],
)
h("5.5 Empaquetado Docker", 2)
bullets([
    ("Imagen mínima: ", "python:3.10-slim con solo requirements-api.txt (FastAPI, Uvicorn, ONNX Runtime, NumPy, "
     "Pillow); sin TensorFlow. Contenido comprimido ≈ 129 MB."),
    ("Seguridad: ", "usuario no root (appuser, UID 10001); .dockerignore excluye dataset, notebooks e informes."),
    ("Operación: ", "HEALTHCHECK contra /health (dinámico según PORT); docker-compose.yml con límites de 2 CPU y 1 GB, "
     "batching activo, ORT_INTRA_OP_THREADS=2 y restart unless-stopped. Consumo medido en carga: ≈ 240 MB de RAM."),
])
code(
    "docker compose up --build -d\n"
    "curl -F \"file=@data/dataset_frutas/platano/platano_001.jpg;type=image/jpeg\" http://localhost:8000/predict"
)
figure(SHOT / "ui_prediccion.png", "Figura 6. Interfaz web servida por la API clasificando una imagen de plátano.", 5.8)
figure(SHOT / "swagger_docs.png", "Figura 7. Documentación OpenAPI interactiva (/docs).", 5.8)

# ---------------------------------------------------------------- 6
h("6. Resultados, discusión y limitaciones", toc_label="6 Resultados, discusión y limitaciones")
rep = cls["report"]
table(
    ["Clase", "Precision", "Recall", "F1", "Soporte"],
    [[c, f1(rep[c]["precision"], 2), f1(rep[c]["recall"], 2), f1(rep[c]["f1-score"], 2), int(rep[c]["support"])]
     for c in meta["classes"]] + [["macro avg", f1(rep["macro avg"]["precision"], 2), f1(rep["macro avg"]["recall"], 2),
                                   f1(rep["macro avg"]["f1-score"], 2), int(rep["macro avg"]["support"])]],
    widths=[1.5, 1.1, 1.1, 1.1, 1.1],
)
figure(FIG / "matriz_confusion_int8.png", "Figura 8. Matriz de confusión del modelo servido (ONNX INT8) en validación.", 3.8)
figure(FIG / "curvas_reentrenamiento.png", "Figura 9. Curvas de entrenamiento del reentrenamiento reproducible.", 6.0)
para(
    "El modelo servido distingue correctamente la mayoría de las clases; la principal confusión ocurre con manzana, "
    "cuyas fotografías presentan colores y formas parecidos a naranja y frutilla. Con solo 16 imágenes de "
    "entrenamiento por clase y 4 de validación, cada imagen equivale a 5 puntos porcentuales de exactitud, por lo que "
    "las métricas tienen alta varianza. Esta limitación proviene del dataset de la Eva3 y no de la optimización: las "
    "cuatro variantes obtienen la misma exactitud. El pipeline deja preparado el ciclo de mejora: al agregar imágenes "
    "y ejecutar dvc repro se reentrena, cuantiza, evalúa y registra automáticamente, y el CI impide desplegar un modelo "
    "que no cumpla los quality gates."
)

# ---------------------------------------------------------------- 7
h("7. Conclusiones", toc_label="7 Conclusiones")
bullets([
    f"La optimización post-entrenamiento (ONNX + INT8 estática) reduce el tamaño {ks['size_mb'] / st['size_mb']:.1f}×, la "
    f"RAM {ks['rss_model_mb'] / st['rss_model_mb']:.1f}× y la latencia p50 {ks['latency_ms']['p50'] / st['latency_ms']['p50']:.0f}× "
    "respecto del modelo Keras de la Eva3, sin pérdida de exactitud.",
    "El plan de acción por fases quedó materializado en un pipeline ejecutable (DVC + GitHub Actions), con "
    "criterios de salida medibles para cada fase.",
    "DVC y MLflow garantizan trazabilidad entre código, datos, parámetros, modelo y métricas; las pruebas de "
    "exactitud, latencia e integridad bloquean regresiones antes del despliegue.",
    "La API FastAPI contenedorizada ofrece contratos claros, manejo robusto de errores, batching dinámico y "
    "observabilidad, y puede desplegarse localmente o en cualquier plataforma de contenedores.",
    "La auditoría inicial corrigió un defecto de la Eva3 (doble normalización), lo que demuestra la importancia "
    "de validar el modelo antes de optimizarlo e implementarlo.",
])

# ---------------------------------------------------------------- 8
h("8. Referencias (APA 7)", toc_label="8 Referencias (APA 7)")
for ref in [
    "Chen, A., Chow, A., Davidson, A., DCunha, A., Ghodsi, A., Hong, S. A., Konwinski, A., Mewald, C., Murching, S., "
    "Nykodym, T., Ogilvie, P., Parkhe, M., Singh, A., Xie, F., Zaharia, M., Zang, R., Zheng, J., & Zumar, C. (2020). "
    "Developments in MLflow: A system to accelerate the machine learning lifecycle. Proceedings of the Fourth "
    "International Workshop on Data Management for End-to-End Machine Learning, 1–4. "
    "https://doi.org/10.1145/3399579.3399867",
    "Docker Inc. (2024). Dockerfile best practices. "
    "https://docs.docker.com/develop/develop-images/dockerfile_best-practices/",
    "DVC. (2024). Data Version Control documentation. Iterative. https://dvc.org/doc",
    "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the "
    "IEEE Conference on Computer Vision and Pattern Recognition, 770–778.",
    "Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization "
    "and training of neural networks for efficient integer-arithmetic-only inference. Proceedings of the IEEE "
    "Conference on Computer Vision and Pattern Recognition, 2704–2713.",
    "Microsoft. (2024). ONNX Runtime: Quantize ONNX models. "
    "https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html",
    "Ramírez, S. (2024). FastAPI documentation. https://fastapi.tiangolo.com",
    "Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., "
    "& Dennison, D. (2015). Hidden technical debt in machine learning systems. Advances in Neural Information "
    "Processing Systems, 28.",
]:
    p = para(ref, align=WD_ALIGN_PARAGRAPH.LEFT, size=9)
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.first_line_indent = Inches(-0.4)

# ---------------------------------------------------------------- Anexo
p = doc.add_paragraph()
p.add_run().add_break(WD_BREAK.PAGE)
h("Anexo A — Reproducción paso a paso", numbered=False, toc_label="Anexo A — Reproducción paso a paso")
para(
    "Para reproducir el entrenamiento, la cuantización ONNX INT8, las pruebas automáticas, el servicio Docker "
    "y la generación de este informe en un entorno local, ejecute los siguientes pasos en orden:"
)
for step_title, step_body in [
    ("1. Clonar el repositorio: ",
     "git clone https://github.com/pamionj/dl-inference-optimization-mlops.git && cd dl-inference-optimization-mlops"),
    ("2. Crear entorno virtual e instalar dependencias: ",
     "python -m venv .venv (active con source .venv/bin/activate en Linux/macOS o .\\.venv\\Scripts\\Activate.ps1 "
     "en Windows) y luego ejecute pip install -r requirements-pipeline.txt -r requirements-dev.txt."),
    ("3. Ejecutar el pipeline reproducible (DVC): ",
     "dvc repro (ejecuta en secuencia las etapas train, export_onnx, quantize, benchmark, evaluate y log_mlflow)."),
    ("4. Inspeccionar experimentos en MLflow: ",
     "mlflow ui --backend-store-uri sqlite:///mlflow.db (interfaz disponible en http://localhost:5000)."),
    ("5. Ejecutar linting y pruebas automáticas (quality gates): ",
     "ruff check app tests pipeline && pytest -v"),
    ("6. Construir y levantar la API en Docker: ",
     "docker compose up --build -d (interfaz web en http://localhost:8000 y documentación OpenAPI en /docs)."),
    ("7. Ejecutar prueba de carga HTTP: ",
     "python -m pipeline.load_test --url http://127.0.0.1:8000 --mode con_batching"),
    ("8. Regenerar el informe técnico (DOCX y PDF): ",
     "python scripts/generate_report_eva4.py"),
]:
    para(step_body, bold_prefix=step_title, align=WD_ALIGN_PARAGRAPH.LEFT)

doc.save(OUT_DOCX)


def finalize_with_word(docx_path: Path) -> bool:
    """Usa Microsoft Word COM para calcular los números reales de página en el Índice, actualizar campos y exportar PDF."""
    try:
        import win32com.client  # type: ignore
    except ImportError:
        return False

    wdActiveEndPageNumber = 3
    wdExportFormatPDF = 17
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        wdoc = word.Documents.Open(str(docx_path.resolve()))
        wdoc.Repaginate()
        page_by_clean: dict[str, int] = {}
        for wp in wdoc.Paragraphs:
            txt = wp.Range.Text.strip()
            if txt in {c for _, c in TOC_HEADINGS} and txt not in page_by_clean:
                page_by_clean[txt] = int(wp.Range.Information(wdActiveEndPageNumber))

        for wp in wdoc.Paragraphs:
            raw = wp.Range.Text
            m = re.search(r"\{TOC_(\d+)\}", raw)
            if m:
                idx_i = int(m.group(1))
                label, _ = TOC_ITEMS[idx_i]
                clean = next(c for t, c in TOC_HEADINGS if t == label)
                pg = page_by_clean.get(clean, 0)
                start_pos = wp.Range.Start + m.start()
                end_pos = wp.Range.Start + m.end()
                wdoc.Range(start_pos, end_pos).Text = str(pg)

        wdoc.Fields.Update()
        for sec in wdoc.Sections:
            for f in sec.Footers:
                f.Range.Fields.Update()
        wdoc.Save()
        wdoc.ExportAsFixedFormat(str(docx_path.with_suffix(".pdf").resolve()), wdExportFormatPDF)
        wdoc.Close(False)
        return True
    finally:
        word.Quit()


if finalize_with_word(OUT_DOCX):
    print(f"[OK] {OUT_DOCX} (Índice y paginación actualizados vía Word)")
    print(f"[OK] {OUT_DOCX.with_suffix('.pdf')}")
else:
    print(f"[OK] {OUT_DOCX}")
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(OUT_DOCX.parent), str(OUT_DOCX)],
                       check=True, capture_output=True)
        print(f"[OK] {OUT_DOCX.with_suffix('.pdf')}")

shutil.copy2(OUT_DOCX, ROOT_DOCX)
print(f"[OK] Sincronizado en {ROOT_DOCX}")
