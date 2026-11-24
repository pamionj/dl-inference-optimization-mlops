# Guía y Estructura Formal del Informe Técnico (Evaluación N°3)
**Asignatura:** Machine Learning (Código: TIEL26) — Unidad 3: Deep Learning  
**Institución:** Instituto Profesional INACAP — Sede Puente Alto  
**Docente Guía:** Francisco Valdivieso Tapia  
**Desarrolladores:** Pablo Amion & Ismael Gómez  
**Repositorio GitHub:** [pamionj/deep-learning-fruit-classifier](https://github.com/pamionj/deep-learning-fruit-classifier)  
**Cuaderno Google Colab (Ejecución Directa):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14iMpDRShODkPczxAgvGY8LvH-RCBW7S_?usp=sharing)  
**Proyecto:** Upgrade hacia Deep Learning con CNN (ResNet50) para Clasificación Multiclase de Frutas (Fase 2)  

---

## 📑 Tabla de Contenidos del Informe Técnico

* **Portada Formal Institucional** ................................................................ **Pág. 1**
* **Resumen Ejecutivo y Nexo Metodológico con la Evaluación N°2** .................... **Pág. 2**
* **Tabla de Contenidos (Índice)** ................................................................ **Pág. 2**
* **Tabla 1: Comparativa de Evolución Metodológica y Arquitectónica (Fase 1 vs. 2)** .... **Pág. 3**
* **1. Actividad N°1: Definición del Problema y Elementos Esenciales (Criterio 3.1.1)** ..... **Pág. 3**
  * 1.1. Contexto Problemático Agroindustrial y Selección de Deep Learning ............... **Pág. 3**
  * 1.2. Imposibilidad Matemática y Paramétrica del MLP en Imágenes de Alta Definición ... **Pág. 3**
  * 1.3. Especificación Técnica de Elementos e Hiperparámetros (Tabla 2) .................. **Pág. 3**
* **2. Actividad N°2: Conjunto de Datos y Aumento de Datos (Criterio 3.1.3)** .......... **Pág. 4**
  * 2.1. Estructura, Curaduría y Balance Simétrico del Dataset de Frutas ................... **Pág. 4**
  * 2.2. Técnicas de Aumento de Datos (Data Augmentation) Justificadas (Tabla 3) ......... **Pág. 4**
  * 2.3. Partición Estratificada 80/20 y Protocolo Anti-Data Leakage ...................... **Pág. 4**
* **3. Actividad N°3: Diseño de la Arquitectura Convolucional CNN (Criterio 3.1.2)** .... **Pág. 5**
  * 3.1. Justificación de Transfer Learning con ResNet50 y Conexiones Residuales .......... **Pág. 5**
  * 3.2. Topología del Cabezal Clasificador (GlobalAveragePooling2D, Dropout, Softmax) .... **Pág. 5**
* **4. Actividad N°4: Entrenamiento, Solución y Evaluación del Modelo (Criterio 3.1.4)** ... **Pág. 6**
  * 4.1. Mecanismos de Control de Sobreajuste (Callbacks: EarlyStopping y ReduceLROnPlateau) **Pág. 6**
  * 4.2. Análisis Crítico: Fase Inicial de Transfer Learning y Detección de Mínimo Local . **Pág. 6**
  * 4.3. Fase Correctiva Avanzada: Fine-Tuning Controlado en Capas Convolucionales ....... **Pág. 6**
  * 4.4. Diagnóstico Diagnóstico Multiclase: Matriz de Confusión y Reporte de Métricas ... **Pág. 6**
* **5. Conclusiones y Cumplimiento del Upgrade Tecnológico** ............................ **Pág. 7**
* **6. Referencias Bibliográficas (Norma APA 7ma Edición)** ............................. **Pág. 7**

---

## 🔗 El Nexo Metodológico: De la Evaluación N°2 a la Evaluación N°3

### ¿Cómo se articula este informe como la continuación / upgrade de la Evaluación 2?
En la **Evaluación N°2** (obteniendo nota 7.0), desarrollamos el pipeline de ingeniería de datos y modelado para un clasificador binario de formas geométricas (*Círculos vs. Cuadrados*) utilizando una red básica **Perceptrón Multicapa (MLP)** en PyTorch sobre imágenes aplanadas de $28 \times 28$ píxeles.

En la sección de **Recomendaciones Técnicas para Escalabilidad en Producción** de aquel informe (pág. 18 de `INFORME_EVALUACION_2.pdf`), advertimos explícitamente:
> *"Aunque el MLP resolvió con éxito las imágenes de 28x28 píxeles, un Perceptrón Multicapa aplana la imagen a un vector unidimensional, destruyendo la estructura de vecindad bidimensional y requiriendo un crecimiento cuadrático de pesos si la resolución aumenta. Para desplegar este sistema con imágenes de alta definición... se recomienda migrar hacia arquitecturas convolucionales ligeras o residuales (ResNet) que explotan la invariancia traslacional mediante kernels deslizantes."*

### La Transformación en la Evaluación N°3 (Deep Learning):
Esta Evaluación N°3 es la **materialización directa** de esa recomendación técnica:
* **De problema canónico sintético a problema agroindustrial real:** En lugar de 2 figuras geométricas básicas en escala de grises, abordamos la clasificación multiclase de 5 categorías de frutas (*Manzana, Plátano, Naranja, Frutilla, Uva*) con imágenes fotográficas RGB en alta resolución ($224 \times 224 \times 3$).
* **De MLP a CNN Profunda con Transfer Learning & Fine-Tuning:** Reemplazamos la red densa (que requeriría más de 77 millones de pesos y colapsaría) por **ResNet50**, aprovechando sus conexiones residuales (*skip-connections*), sus filtros pre-entrenados sobre 1.4 millones de imágenes de *ImageNet* y una fase de ajuste fino (*Fine-Tuning*) de las últimas 15 capas convolucionales con learning rate atenuado ($1 \times 10^{-5}$).
* **De vector 1D a preservación espacial 2D:** Mantenemos la coherencia topológica mediante convoluciones y usamos `GlobalAveragePooling2D` para evitar la sobreparametrización.
* **De inyección de ruido simple a Data Augmentation en tiempo real:** Rotaciones de $20^\circ$, zoom, desplazamientos y volteo horizontal para simular frutas en cualquier posición de una línea de selección.
* **De métricas binarias a diagnóstico multiclase completo:** Evaluación mediante **Matriz de Confusión** (mapa de calor) y métricas desagregadas por fruta (*Precision, Recall, F1-Score*).
