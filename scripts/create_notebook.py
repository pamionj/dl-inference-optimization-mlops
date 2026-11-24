import json
import os

notebook = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.12'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

def add_cell(cell_type, source):
    notebook['cells'].append({
        'cell_type': cell_type,
        'metadata': {},
        'execution_count': None,
        'outputs': [] if cell_type == 'code' else None,
        'source': [line + '\n' for line in source.split('\n')]
    })

# Cell 0: Header Markdown
add_cell('markdown', """# Evaluación Sumativa N°3: Deep Learning con Redes Neuronales Convolucionales (CNN)
**Asignatura:** Machine Learning (TIEL26) - Unidad 3: Deep Learning  
**Carrera:** Ingeniería en Informática  
**Institución:** Instituto Profesional INACAP - Sede Puente Alto  
**Docente Guía:** Francisco Valdivieso Tapia  
**Desarrolladores:** Pablo Amion & Ismael Gómez  
**Repositorio GitHub:** [pamionj/deep-learning-fruit-classifier](https://github.com/pamionj/deep-learning-fruit-classifier)  
**Proyecto:** Upgrade hacia Deep Learning con CNN (ResNet50) para Clasificación Multiclase de Frutas (Fase 2)

---
### Objetivos y Criterios Evaluados (Continuación y Upgrade de Evaluación N°2):
1. **Criterio 3.1.1 (Elementos Necesarios):** Identificación y justificación de hardware, framework, hiperparámetros (learning rate, batch size, épocas), optimizador y función de pérdida. Justificación técnica de la transición desde el Perceptrón Multicapa (MLP) de Evaluación 2 hacia Deep Learning con CNN.
2. **Criterio 3.1.2 (Diseño del Modelo CNN):** Arquitectura basada en Transfer Learning con ResNet50 pre-entrenada sobre ImageNet, GlobalAveragePooling2D, regularización por Dropout y clasificador Softmax.
3. **Criterio 3.1.3 (Dataset y Data Augmentation):** Estructura del conjunto de imágenes de frutas, división estratificada (80% entrenamiento / 20% validación) y técnicas de aumento de datos con justificación técnica.
4. **Criterio 3.1.4 (Entrenamiento y Solución):** Entrenamiento reproducible, control de sobreajuste mediante callbacks (EarlyStopping, ReduceLROnPlateau), curvas de Loss/Accuracy, Matriz de Confusión y reporte de métricas (Precision, Recall, F1-Score).""")

# Cell 1: Environment & Imports
add_cell('code', """# ==============================================================================
# 1. CONFIGURACIÓN DEL ENTORNO, IMPORTACIONES Y HARDWARE
# ==============================================================================
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Detección del entorno (Google Colab vs Local)
IS_COLAB = 'google.colab' in sys.modules
if IS_COLAB:
    print("[ENTORNO] Ejecutándose en Google Colab.")
else:
    print("[ENTORNO] Ejecutándose en entorno local (VS Code / Jupyter Notebook).")

# Importaciones de Deep Learning (TensorFlow y Keras)
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Métricas de evaluación
from sklearn.metrics import confusion_matrix, classification_report

# Diagnóstico de GPU
print(f"Versión de TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Aceleración por GPU ACTIVA: {len(gpus)} dispositivo(s) detectado(s) ({gpus[0].name})")
else:
    print("Aceleración GPU no detectada; el entrenamiento se ejecutará en CPU.")

# Semilla de reproducibilidad
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
print("Semillas de reproducibilidad fijadas (SEED = 42).")""")

# Cell 2: Hyperparameters Markdown
add_cell('markdown', """## 2. Criterio 3.1.1: Elementos Necesarios e Hiperparámetros
Para abordar la problemática de clasificación de imágenes frutícolas se configuran los siguientes elementos esenciales:
- **Hardware:** CPU multicore o GPU dedicada (CUDA / Google Colab T4).
- **Framework:** TensorFlow 2.x con Keras por su robustez, soporte de modelos pre-entrenados y despliegue estándar.
- **Resolución de Entrada:** $224 \\times 224 \\times 3$, dimensión canónica de ResNet50 para preservar la geometría y la textura de la fruta.
- **Tamaño de Lote (Batch Size):** $16$ (o $32$), equilibrando el cálculo del gradiente estocástico y el uso de memoria VRAM.
- **Tasa de Aprendizaje (Learning Rate):** $1 \\times 10^{-4}$ con el optimizador **Adam** para una convergencia suave sin desestabilizar los pesos de la red pre-entrenada.
- **Función de Pérdida (Loss):** `categorical_crossentropy`, idónea para clasificación multiclase mutuamente excluyente.
- **Épocas:** $15$, supervisadas mediante parada temprana (*Early Stopping*).""")

# Cell 3: Hyperparameters Code
add_cell('code', """# ==============================================================================
# 2. DEFINICIÓN DE HIPERPARÁMETROS Y RUTAS DEL PROYECTO
# ==============================================================================
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4

# Rutas de búsqueda dinámica para soportar Colab y Local sin modificar código
POSSIBLE_PATHS = [
    'data/dataset_frutas',
    '../data/dataset_frutas',
    '/content/dataset_frutas',
    'dataset_frutas'
]

DATASET_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p) and os.path.isdir(p):
        DATASET_PATH = p
        break

if DATASET_PATH is None:
    DATASET_PATH = 'data/dataset_frutas'
    os.makedirs(DATASET_PATH, exist_ok=True)
    print(f"[AVISO] Dataset no localizado. Creando directorio base: {DATASET_PATH}")
else:
    print(f"Directorio de imágenes seleccionado: {os.path.abspath(DATASET_PATH)}")

# Crear directorios para resultados y modelos
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)""")

# Cell 4: Data Augmentation Markdown
add_cell('markdown', """## 3. Criterio 3.1.3: Dataset y Data Augmentation (Aumento de Datos)
El conjunto de datos debe representar la variabilidad que enfrentaría un sistema en un entorno real (cintas clasificadoras, packing agrícola, anaqueles).
Se aplica aumento de datos en tiempo real mediante `ImageDataGenerator`:
1. **Reescalado (`rescale=1./255`):** Normaliza el tensor de entrada al rango $[0, 1]$.
2. **Rotación (`rotation_range=20`):** Simula frutas capturadas en orientaciones angulares diversas.
3. **Zoom (`zoom_range=0.2`):** Modela diferencias de distancia respecto a la cámara.
4. **Desplazamiento horizontal/vertical (`width_shift_range=0.2`, `height_shift_range=0.2`):** Permite reconocer frutos no centrados en el cuadro.
5. **Volteo horizontal (`horizontal_flip=True`):** Invarianza simétrica lateral.
6. **Partición de Validación (`validation_split=0.2`):** 80% para entrenamiento y 20% para validación independiente.""")

# Cell 5: Generators Code
add_cell('code', """# ==============================================================================
# 3. GENERADORES DE DATOS CON DATA AUGMENTATION Y VALIDACIÓN
# ==============================================================================
# Generador de entrenamiento con aumentación
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Generador de validación (estrictamente sin aumentaciones geométricas)
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# Flujo de datos para entrenamiento
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

# Flujo de datos para validación (shuffle=False para permitir matriz de confusión)
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

NUM_CLASSES = len(train_generator.class_indices)
CLASS_NAMES = list(train_generator.class_indices.keys())

print(f"\\nResumen del Dataset:")
print(f" - Clases detectadas ({NUM_CLASSES}): {CLASS_NAMES}")
print(f" - Imágenes de entrenamiento: {train_generator.samples}")
print(f" - Imágenes de validación:    {val_generator.samples}")""")

# Cell 6: Visualization Code
add_cell('code', """# ==============================================================================
# 4. VISUALIZACIÓN DE IMÁGENES AUMENTADAS (EVIDENCIA CRITERIO 3.1.3)
# ==============================================================================
if train_generator.samples > 0:
    batch_images, batch_labels = next(train_generator)
    
    plt.figure(figsize=(12, 6))
    num_to_show = min(8, len(batch_images))
    for i in range(num_to_show):
        plt.subplot(2, 4, i + 1)
        plt.imshow(batch_images[i])
        cls_idx = np.argmax(batch_labels[i])
        plt.title(f"{CLASS_NAMES[cls_idx]}", fontsize=11)
        plt.axis('off')
    plt.suptitle("Muestras de Frutas con Data Augmentation (Train)", fontsize=13)
    plt.tight_layout()
    plt.savefig('outputs/muestras_data_augmentation.png', dpi=150)
    plt.show()
    print("Figura guardada en: outputs/muestras_data_augmentation.png")
else:
    print("[AVISO] No se encontraron imágenes para previsualizar.")""")

# Cell 7: Architecture Markdown
add_cell('markdown', """## 4. Criterio 3.1.2: Diseño del Modelo CNN (Transfer Learning con ResNet50)
### Justificación Técnica frente a Métodos Tradicionales:
1. **Extracción Automática de Rasgos vs Feature Engineering manual:** En ML clásico (SVM, Random Forest), se requiere extraer descriptores manuales como SIFT, HOG o histogramas de color. Las CNN aprenden jerarquías de características visuales directamente desde los datos crudos (desde bordes y texturas en las primeras capas hasta formas compuestas y patrones de fruta en las capas profundas).
2. **Transfer Learning con ResNet50:** ResNet50 cuenta con conexiones residuales (*shortcut connections*) que mitigan el desvanecimiento del gradiente en redes de 50 capas. Al reutilizar pesos entrenados sobre 1.4 millones de imágenes de *ImageNet*, se transfiere el conocimiento visual básico, acelerando drásticamente el entrenamiento y requiriendo un menor volumen de datos.
3. **Head Clasificador Personalizado:**
   - `GlobalAveragePooling2D`: Transforma los mapas de activación de $7 \\times 7 \\times 2048$ en un vector plano de 2048 elementos calculando la media espacial, reduciendo notablemente el número de parámetros frente a un `Flatten` convencional y actuando como regularizador estructural.
   - `Dropout(0.4) y Dropout(0.3)`: Desactivan aleatoriamente neuronas durante el entrenamiento para evitar sobreajuste y co-adaptaciones complejas.
   - `Dense(256, activation='relu')`: Capa densa intermedia que combina las características visuales aprendidas.
   - `Dense(NUM_CLASSES, activation='softmax')`: Entrega la distribución de probabilidad para cada fruta clasificada.""")

# Cell 8: Architecture Code
add_cell('code', """# ==============================================================================
# 5. CONSTRUCCIÓN DE LA ARQUITECTURA DEL MODELO
# ==============================================================================
# Cargar base convolucional ResNet50 sin el clasificador original de 1000 clases
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Congelar capas base para preservar características pre-entrenadas
base_model.trainable = False

# Ensamblado del clasificador personalizado
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='input_image')
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D(name='global_avg_pool')(x)
x = Dropout(0.4, name='dropout_regularizer_1')(x)
x = Dense(256, activation='relu', name='dense_representation')(x)
x = Dropout(0.3, name='dropout_regularizer_2')(x)
outputs = Dense(NUM_CLASSES, activation='softmax', name='class_probabilities')(x)

model = Model(inputs=inputs, outputs=outputs, name='ResNet50_Clasificador_Frutas')

# Compilación del modelo
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Resumen de capas y parámetros
model.summary()""")

# Cell 9: Training Markdown
add_cell('markdown', """## 5. Criterio 3.1.4: Entrenamiento y Control de Sobreajuste (Overfitting)
Para asegurar que el modelo alcance su máximo rendimiento sin sobreajuste, se integran tres callbacks clave:
- **`EarlyStopping`:** Detiene automáticamente el proceso si la pérdida de validación (`val_loss`) no disminuye tras 4 épocas consecutivas, recuperando el mejor punto de pesos (`restore_best_weights=True`).
- **`ReduceLROnPlateau`:** Reduce dinámicamente la tasa de aprendizaje a la quinta parte (`factor=0.2`) cuando la pérdida en validación se estanca durante 2 épocas consecutivas.
- **`ModelCheckpoint`:** Guarda en disco el mejor estado del modelo evaluado en validación.""")

# Cell 10: Training Code
add_cell('code', """# ==============================================================================
# 6. CONFIGURACIÓN DE CALLBACKS Y ENTRENAMIENTO
# ==============================================================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='models/mejor_modelo_frutas.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Pasos por época
steps_per_epoch = max(1, math.ceil(train_generator.samples / BATCH_SIZE))
val_steps = max(1, math.ceil(val_generator.samples / BATCH_SIZE)) if val_generator.samples > 0 else None

print(f"Pasos de entrenamiento por época: {steps_per_epoch}")
print(f"Pasos de validación por época:    {val_steps}")

if train_generator.samples > 0 and val_generator.samples > 0:
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print("\\n[EXITO] Entrenamiento concluido correctamente.")
else:
    print("[ERROR] Se requieren imágenes en 'train' y 'val' para entrenar.")
    print("Ejecuta 'python scripts/download_sample_dataset.py' o coloca imágenes en 'data/dataset_frutas'.")
    history = None""")

# Cell 11: Plots Markdown
add_cell('markdown', """## 6. Resultados: Curvas de Pérdida (Loss) y Precisión (Accuracy)
El seguimiento de las curvas de entrenamiento y validación permite verificar:
1. **Convergencia:** Disminución progresiva del valor de pérdida (`loss` y `val_loss`).
2. **Generalización:** La cercanía entre las curvas de entrenamiento y validación evidencia que la regularización por *Data Augmentation* y *Dropout* evitó el sobreajuste.""")

# Cell 12: Plots Code
add_cell('code', """# ==============================================================================
# 7. GENERACIÓN DE CURVAS DE RENDIMIENTO (LOSS Y ACCURACY)
# ==============================================================================
if history is not None:
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Gráfico de Precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, marker='o', label='Entrenamiento (Train Acc)', color='#1f77b4', linewidth=2)
    if val_acc:
        plt.plot(epochs_range, val_acc, marker='s', label='Validación (Val Acc)', color='#ff7f0e', linewidth=2)
    plt.title('Precisión por Época (Accuracy)', fontsize=13)
    plt.xlabel('Época', fontsize=11)
    plt.ylabel('Exactitud', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    # Gráfico de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, marker='o', label='Entrenamiento (Train Loss)', color='#d62728', linewidth=2)
    if val_loss:
        plt.plot(epochs_range, val_loss, marker='s', label='Validación (Val Loss)', color='#9467bd', linewidth=2)
    plt.title('Pérdida por Época (Loss)', fontsize=13)
    plt.xlabel('Época', fontsize=11)
    plt.ylabel('Pérdida (Categorical Crossentropy)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('outputs/curvas_rendimiento.png', dpi=150)
    plt.show()
    print("Gráfica guardada en: outputs/curvas_rendimiento.png")""")

# Cell 13: Confusion Matrix Markdown
add_cell('markdown', """## 7. Evaluación Detallada: Matriz de Confusión y Reporte de Clasificación
Para cumplir rigurosamente con los indicadores de la **Actividad N°3 y Criterio 3.1.4**, se evalúa el modelo sobre el conjunto de validación:
- **Matriz de Confusión:** Diagnostica las predicciones correctas en la diagonal principal e identifica si existen confusiones entre frutas específicas (por ejemplo, entre naranja y manzana).
- **Métricas:** *Precision* (certeza de predicción positiva), *Recall* (capacidad de detección de la clase real) y *F1-Score* (balance armónico).""")

# Cell 14: Confusion Matrix Code
add_cell('code', """# ==============================================================================
# 8. MATRIZ DE CONFUSIÓN Y REPORTE DE CLASIFICACIÓN (SKLEARN)
# ==============================================================================
if val_generator.samples > 0:
    val_generator.reset()
    
    # Inferencia sobre todas las muestras de validación
    y_pred_probs = model.predict(val_generator, steps=val_steps, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes[:len(y_pred)]

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES,
        cbar=True
    )
    plt.title('Matriz de Confusión - Clasificación de Frutas', fontsize=13)
    plt.xlabel('Predicción del Modelo', fontsize=11)
    plt.ylabel('Etiqueta Real', fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/matriz_confusion.png', dpi=150)
    plt.show()
    print("Matriz de confusión guardada en: outputs/matriz_confusion.png\\n")

    # 2. Reporte de Clasificación Detallado
    print("=" * 65)
    print("REPORTE DETALLADO DE MÉTRICAS (PRECISION, RECALL, F1-SCORE)")
    print("=" * 65)
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report_str)
    
    with open('outputs/reporte_clasificacion.txt', 'w', encoding='utf-8') as f:
        f.write(report_str)
    print("Reporte guardado en: outputs/reporte_clasificacion.txt")""")

# Cell 15: Save Model Markdown
add_cell('markdown', """## 8. Almacenamiento del Modelo y Exportación
El modelo final y sus pesos óptimos se guardan en formato `.keras`, asegurando compatibilidad para futuras tareas de inferencia o despliegue en APIs y aplicaciones móviles/web.""")

# Cell 16: Save Model Code
add_cell('code', """# ==============================================================================
# 9. GUARDADO DEL MODELO FINAL Y COMPATIBILIDAD CON COLAB
# ==============================================================================
final_model_path = 'models/modelo_frutas_resnet50.keras'
model.save(final_model_path)
print(f"Modelo final guardado en: {os.path.abspath(final_model_path)}")

# En Google Colab, habilitar descarga automática
if IS_COLAB:
    try:
        from google.colab import files
        print("Iniciando descarga local desde Google Colab...")
        files.download(final_model_path)
    except Exception as e:
        print(f"Descarga automática no iniciada: {e}")
else:
    print("Ejecución en local: El modelo queda almacenado en la carpeta 'models/'.")""")

output_file = 'deep-learning-example.ipynb'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Archivo '{output_file}' generado exitosamente con {len(notebook['cells'])} celdas.")

