# Proceso de Clasificación de Audio SMAW

Sistema de clasificación automática de audio de soldadura SMAW (Shielded Metal Arc Welding) usando aprendizaje profundo.

---

## 1. Objetivo

A partir de un audio de soldadura, el sistema predice automáticamente:

| Característica    | Valores posibles           |
| ----------------- | -------------------------- |
| Espesor de placa  | 3mm, 6mm, 12mm             |
| Tipo de electrodo | E6010, E6011, E6013, E7018 |
| Tipo de corriente | AC, DC                     |

---

## 2. Preparación del Entorno

### 2.1 Requisitos del Sistema

- Python 3.8+
- FFmpeg (para extracción de audio)
- CUDA (opcional, para GPU)

### 2.2 Dependencias Python

```bash
pip install torch torchaudio librosa pandas numpy scikit-learn tensorflow tensorflow-hub
```

---

## 3. Extracción de Audio desde Videos

### 3.1 Ejecutar la Extracción

```bash
# Vista previa (sin ejecutar)
python scripts/extract_and_organize_audio.py --dry-run --videos-dir videos-soldadura

# Extracción real
python scripts/extract_and_organize_audio.py \
    --videos-dir videos-soldadura \
    --output-dir audio \
    --samplerate 16000
```

### 3.2 Parámetros de Extracción

| Parámetro   | Valor          | Descripción            |
| ----------- | -------------- | ---------------------- |
| Sample rate | 16000 Hz       | Requerido por VGGish   |
| Canales     | Mono           | Un solo canal de audio |
| Formato     | WAV PCM 16-bit | Sin pérdida de calidad |

### 3.3 Estructura de Audio Resultante

```
audio/
+-- Placa_Xmm/                         <-- Espesor (3mm, 6mm, 12mm)
    +-- EXXXX/                         <-- Electrodo (E6010, E6011, etc.)
        +-- {AC,DC}/                   <-- Tipo de corriente
            +-- YYMMDD-HHMMSS_Audio/   <-- Sesión de grabación
                +-- *.wav              <-- Archivos de audio
```

**Sesión:** Una grabación continua de soldadura identificada por su carpeta con timestamp.

---

## 4. Segmentación de Audio

### 4.1 Parámetros de Segmentación

| Carpeta | Duración segmento | Hop (salto) | Solapamiento |
| ------- | ----------------- | ----------- | ------------ |
| 1seg/   | 1 segundo         | 0.5 seg     | 50%          |
| 2seg/   | 2 segundos        | 1 seg       | 50%          |
| 5seg/   | 5 segundos        | 2.5 seg     | 50%          |
| 10seg/  | 10 segundos       | 5 seg       | 50%          |
| 30seg/  | 30 segundos       | 15 seg      | 50%          |

### 4.2 Segmentación On-the-fly

Los segmentos NO se guardan como archivos separados. El sistema los calcula dinámicamente durante el entrenamiento:

```
segmentos = floor((duracion_audio - duracion_segmento) / hop) + 1
```

---

## 5. División de Datos

### 5.1 Ejecutar Generación de Splits

```bash
cd 5seg/
python generar_splits.py
```

### 5.2 Conjuntos Generados

| Archivo      | Porcentaje | Propósito                              |
| ------------ | ---------- | -------------------------------------- |
| train.csv    | 72%        | Entrenamiento (K-Fold CV)              |
| test.csv     | 18%        | Validación durante desarrollo          |
| holdout.csv  | 10%        | Evaluación final (nunca en desarrollo) |
| completo.csv | 100%       | Referencia con columna Split           |

### 5.3 Prevención de Data Leakage

El sistema utiliza `StratifiedGroupKFold` para garantizar que todos los segmentos de una misma sesión permanezcan en el mismo conjunto, evitando que el modelo memorice características de grabaciones específicas.

---

## 6. Extracción de Características con VGGish

VGGish es una red neuronal pre-entrenada que convierte audio en embeddings:

1. Carga el audio a 16kHz mono
2. Divide en ventanas de 1 segundo con solapamiento de 0.5 segundos
3. Cada ventana se convierte en un vector de 128 dimensiones
4. Resultado: secuencia de vectores `[T, 128]`

---

## 7. Arquitectura del Modelo

El modelo SMAWXVectorModel procesa los embeddings de VGGish:

```
Entrada: Embeddings VGGish [T, 128]
            |
            v
+-------------------------------------+
| BatchNorm1d                         |
| (normalizacion por lotes)           |
+-------------------------------------+
            |
            v
+-------------------------------------+
| XVector1D                           |
| - Conv1D: 128 --> 256 canales       |
| - Conv1D: 256 --> 256 canales       |
| - Conv1D: 256 --> 512 canales       |
| Cada capa: BatchNorm + ReLU         |
+-------------------------------------+
            |
            v
+-------------------------------------+
| StatsPooling                        |
| Calcula media y desviacion estandar |
| Salida: 512 x 2 = 1024 valores      |
+-------------------------------------+
            |
            v
+-------------------------------------+
| MultiHeadClassifier                 |
| - FC: 1024 --> 256 + ReLU           |
| - FC: 256 --> 3 (Espesor)           |
| - FC: 256 --> 4 (Electrodo)         |
| - FC: 256 --> 2 (Corriente)         |
+-------------------------------------+
```

---

## 8. Entrenamiento

### 8.1 Ejecutar Entrenamiento

```bash
cd 5seg/
python entrenar.py
```

### 8.2 Hiperparámetros

| Parámetro       | Valor  | Descripción                         |
| --------------- | ------ | ----------------------------------- |
| Épocas          | 100    | Número máximo de iteraciones        |
| Batch size      | 32     | Ejemplos por actualización de pesos |
| Learning rate   | 0.001  | Tasa de aprendizaje                 |
| Weight decay    | 0.0001 | Regularización L2                   |
| Label smoothing | 0.1    | Suavizado de etiquetas              |
| Early stopping  | 15     | Épocas sin mejora antes de parar    |
| K-Folds         | 5      | Particiones de validación cruzada   |

### 8.3 Validación Cruzada K-Fold

Se entrenan 5 modelos, cada uno validando con un fold diferente:

```
train.csv (241 sesiones) --> 5 Folds
              |
              v
Fold 1: Train=193 sesiones, Val=48 sesiones --> model_fold_0.pth
Fold 2: Train=193 sesiones, Val=48 sesiones --> model_fold_1.pth
Fold 3: Train=193 sesiones, Val=48 sesiones --> model_fold_2.pth
Fold 4: Train=193 sesiones, Val=48 sesiones --> model_fold_3.pth
Fold 5: Train=193 sesiones, Val=48 sesiones --> model_fold_4.pth
```

### 8.4 Técnicas de Entrenamiento

- **AdamW**: Optimizador con weight decay desacoplado
- **CrossEntropyLoss**: Función de pérdida con label smoothing
- **Class Weighting**: Pesos inversamente proporcionales a la frecuencia de clase
- **Early Stopping**: Detiene si no mejora en 15 épocas
- **SWA**: Promedia pesos a partir de época 5 para mejor generalización

---

## 9. Evaluación en Holdout

### 9.1 Ejecutar Inferencia

```bash
cd 5seg/
python infer.py --evaluar
```

### 9.2 Ensemble con Soft Voting

Los 5 modelos se combinan promediando sus logits antes de aplicar argmax:

```
Audio de entrada
        |
        v
    VGGish Embeddings
        |
        v
+-----------------------------------+
| model_0 --> logits_0              |
| model_1 --> logits_1              |
| model_2 --> logits_2              |
| model_3 --> logits_3              |
| model_4 --> logits_4              |
+-----------------------------------+
        |
        v
    mean(logits)
        |
        v
    argmax --> Prediccion final
```

### 9.3 Métricas

| Métrica  | Descripción                          |
| -------- | ------------------------------------ |
| Accuracy | Porcentaje de predicciones correctas |
| F1-score | Media armónica de precision y recall |

---

## 10. Archivos Generados

```
5seg/
|-- completo.csv          # Todos los datos con split asignado
|-- train.csv             # Datos de entrenamiento
|-- test.csv              # Datos de validación
|-- holdout.csv           # Datos de evaluación final
|-- results.json          # Métricas del entrenamiento
|-- infer.json            # Métricas de holdout
+-- models/
    |-- model_fold_0.pth
    |-- model_fold_1.pth
    |-- model_fold_2.pth
    |-- model_fold_3.pth
    +-- model_fold_4.pth
```

---

## 11. Diagrama de Flujo Completo

```
+-------------------------------------------------------------------------+
|                         FASE 1: PREPARACION                             |
+-------------------------------------------------------------------------+
|                                                                         |
|  Videos de Soldadura                                                    |
|  (videos-soldadura/Placa_*/E####*/*.mp4)                                |
|                     |                                                   |
|                     v                                                   |
|  +------------------------------------------+                           |
|  | extract_and_organize_audio.py            |                           |
|  | (FFmpeg: -vn -ar 16000 -ac 1 pcm_s16le)  |                           |
|  +------------------------------------------+                           |
|                     |                                                   |
|                     v                                                   |
|  Audio Organizado                                                       |
|  (audio/Placa_*/E####/{AC,DC}/TIMESTAMP_Audio/*.wav)                    |
|                                                                         |
+-------------------------------------------------------------------------+
                      |
                      v
+-------------------------------------------------------------------------+
|                    FASE 2: DIVISION DE DATOS                            |
+-------------------------------------------------------------------------+
|                                                                         |
|  +------------------------------------------+                           |
|  | generar_splits.py                        |                           |
|  | - Descubre sesiones                      |                           |
|  | - Calcula segmentos por sesion           |                           |
|  | - Estratifica por (Placa+Elect+Corr)     |                           |
|  | - Divide manteniendo sesiones intactas   |                           |
|  +------------------------------------------+                           |
|                     |                                                   |
|        +------------+------------+------------+                         |
|        v            v            v            v                         |
|   train.csv    test.csv    holdout.csv   completo.csv                   |
|     (72%)        (18%)        (10%)        (100%)                       |
|                                                                         |
+-------------------------------------------------------------------------+
                      |
                      v
+-------------------------------------------------------------------------+
|                    FASE 3: ENTRENAMIENTO                                |
+-------------------------------------------------------------------------+
|                                                                         |
|  train.csv ---------------------------------------------------+         |
|                                                                |         |
|  Para cada fold k in {0,1,2,3,4}:                              |         |
|  +-------------------------------------------------------------+---+    |
|  |                                                                 |    |
|  |  +------------------+    +------------------+                   |    |
|  |  | Train Sessions   |    |  Val Sessions    |                   |    |
|  |  |     (80%)        |    |     (20%)        |                   |    |
|  |  +--------+---------+    +--------+---------+                   |    |
|  |           |                       |                             |    |
|  |           v                       v                             |    |
|  |  +------------------------------------------+                   |    |
|  |  | Segmentacion On-the-fly                  |                   |    |
|  |  | (hop_ratio=0.5, overlap=50%)             |                   |    |
|  |  +--------------------+---------------------+                   |    |
|  |                       |                                         |    |
|  |                       v                                         |    |
|  |  +------------------------------------------+                   |    |
|  |  | VGGish Embedding (TensorFlow Hub)        |                   |    |
|  |  | Audio --> [T, 128] embeddings            |                   |    |
|  |  +--------------------+---------------------+                   |    |
|  |                       |                                         |    |
|  |                       v                                         |    |
|  |  +------------------------------------------+                   |    |
|  |  | SMAWXVectorModel                         |                   |    |
|  |  | - BatchNorm1d                            |                   |    |
|  |  | - XVector1D (Conv1D x3)                  |                   |    |
|  |  | - StatsPooling (mean + std)              |                   |    |
|  |  | - MultiHeadClassifier                    |                   |    |
|  |  +--------------------+---------------------+                   |    |
|  |                       |                                         |    |
|  |                       v                                         |    |
|  |  +------------------------------------------+                   |    |
|  |  | CrossEntropyLoss (label_smoothing=0.1)   |                   |    |
|  |  | + Class Weighting                        |                   |    |
|  |  | Loss = loss_plate + loss_elec + loss_curr|                   |    |
|  |  +--------------------+---------------------+                   |    |
|  |                       |                                         |    |
|  |                       v                                         |    |
|  |  +------------------------------------------+                   |    |
|  |  | AdamW Optimizer + SWA                    |                   |    |
|  |  | Early Stopping (patience=15)             |                   |    |
|  |  +--------------------+---------------------+                   |    |
|  |                       |                                         |    |
|  |                       v                                         |    |
|  |              model_fold_k.pth                                   |    |
|  |                                                                 |    |
|  +-----------------------------------------------------------------+    |
|                                                                         |
|  Resultado: 5 modelos + results.json                                    |
|                                                                         |
+-------------------------------------------------------------------------+
                      |
                      v
+-------------------------------------------------------------------------+
|                    FASE 4: EVALUACION                                   |
+-------------------------------------------------------------------------+
|                                                                         |
|  holdout.csv -------------------------------------------------+         |
|                                                                |         |
|  +-------------------------------------------------------------+---+    |
|  |                                                                 |    |
|  |  Para cada segmento en holdout:                                 |    |
|  |                                                                 |    |
|  |  +------------------------------------------+                   |    |
|  |  | VGGish Embedding                         |                   |    |
|  |  +--------------------+---------------------+                   |    |
|  |                       |                                         |    |
|  |         +-------------+-------------+                           |    |
|  |         v             v             v                           |    |
|  |     model_0       model_1  ...  model_4                         |    |
|  |         |             |             |                           |    |
|  |         v             v             v                           |    |
|  |     logits_0      logits_1 ... logits_4                         |    |
|  |         |             |             |                           |    |
|  |         +-------------+-------------+                           |    |
|  |                       v                                         |    |
|  |  +------------------------------------------+                   |    |
|  |  | Soft Voting: mean(logits)                |                   |    |
|  |  | Prediccion: argmax(mean_logits)          |                   |    |
|  |  +--------------------+---------------------+                   |    |
|  |                       |                                         |    |
|  |                       v                                         |    |
|  |  +------------------------------------------+                   |    |
|  |  | Metricas: Accuracy, F1-score             |                   |    |
|  |  | Por tarea: Placa, Electrodo, Corriente   |                   |    |
|  |  +------------------------------------------+                   |    |
|  |                                                                 |    |
|  +-----------------------------------------------------------------+    |
|                                                                         |
|  Resultado: infer.json + METRICAS.md                                    |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## 12. Resumen

El sistema de clasificación de audio SMAW transforma grabaciones de soldadura en predicciones automáticas de tres parámetros: espesor de placa, tipo de electrodo y tipo de corriente.

**Pipeline:**

1. **Extracción**: FFmpeg extrae audio WAV 16kHz mono de los videos
2. **División**: Sesiones se dividen en train/test/holdout sin mezclar segmentos
3. **Segmentación**: Audios se segmentan on-the-fly con 50% de solapamiento
4. **Características**: VGGish genera embeddings de 128 dimensiones
5. **Clasificación**: SMAWXVectorModel predice las tres etiquetas simultáneamente
6. **Entrenamiento**: 5-Fold CV con AdamW, SWA, early stopping y balanceo de clases
7. **Inferencia**: Ensemble de 5 modelos con soft voting

**Rendimiento típico** (segmentos de 5-10 segundos):

- Placa: ~75% accuracy
- Electrodo: ~85% accuracy
- Corriente: ~95% accuracy
