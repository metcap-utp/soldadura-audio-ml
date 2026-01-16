# Proceso de Clasificación de Audio SMAW

Sistema de clasificación automática de audio de soldadura SMAW (Shielded Metal Arc Welding) usando aprendizaje profundo.

---

## 1. Estructura del Proyecto

El proyecto tiene tres variantes según la duración de los segmentos de audio:

| Carpeta  | Duración    |
| -------- | ----------- |
| `5seg/`  | 5 segundos  |
| `10seg/` | 10 segundos |
| `30seg/` | 30 segundos |

Cada carpeta contiene el mismo proceso y estructura interna.

---

## 2. Objetivo

A partir de un audio de soldadura, el sistema predice automáticamente:

| Característica    | Valores posibles           |
| ----------------- | -------------------------- |
| Espesor de placa  | 3mm, 6mm, 12mm             |
| Tipo de electrodo | E6010, E6011, E6013, E7018 |
| Tipo de corriente | AC, DC                     |

---

## 3. Organización de los Datos de Audio

Los archivos de audio están organizados en carpetas que codifican las etiquetas:

```
audio/
└── Placa_Xmm/                    <- Espesor (3mm, 6mm, 12mm)
    └── EXXXX/                    <- Electrodo (E6010, E6011, etc.)
        └── {AC,DC}/              <- Tipo de corriente
            └── YYMMDD-HHMMSS_Audio/   <- Sesión de grabación
                └── *.wav              <- Archivos de audio
```

**Sesión:** Una grabación continua de soldadura. Se identifica por la carpeta con fecha y hora (ejemplo: `240912-143741_Audio`). De cada grabación original se extraen múltiples segmentos cortos.

---

## 4. Proceso Completo Paso a Paso

### Paso 1: División de Datos

**Archivo:** `generar_splits.py`

Este script prepara los datos antes del entrenamiento:

1. **Escanea** todos los archivos `.wav` en `audio/`
2. **Extrae etiquetas** automáticamente desde los nombres de carpetas
3. **Agrupa por sesión** para evitar contaminación de datos (data leakage: cuando información del conjunto de prueba se filtra al entrenamiento, causando resultados falsamente optimistas)
4. **Divide estratificadamente** (mantiene proporciones de cada clase en todos los conjuntos):

| Conjunto       | Porcentaje | Propósito                                           |
| -------------- | ---------- | --------------------------------------------------- |
| `train.csv`    | 72%        | Entrenamiento del modelo                            |
| `test.csv`     | 18%        | Validación durante desarrollo                       |
| `holdout.csv`  | 10%        | Evaluación final (nunca se toca durante desarrollo) |
| `completo.csv` | 100%       | Referencia con columna Split                        |

**Regla importante:** Todos los segmentos de una misma sesión van al mismo conjunto. Esto evita que el modelo "memorice" características específicas de una grabación.

---

### Paso 2: Extracción de Características

**Modelo usado:** VGGish

**Ubicación:** `vggish/`

VGGish es una red neuronal pre-entrenada (ya aprendió de millones de audios) que convierte audio en vectores numéricos llamados embeddings (representaciones numéricas compactas que capturan las características importantes del audio).

**Proceso de extracción:**

1. Carga el audio a 16kHz en formato mono
2. Divide el audio en ventanas de 1 segundo, con solapamiento de 0.5 segundos
3. Cada ventana se convierte en un vector de 128 números
4. Resultado: una secuencia de vectores `[T, 128]` donde T es el número de ventanas

Ejemplo: un audio de 5 segundos produce aproximadamente 9 vectores de 128 dimensiones.

---

### Paso 3: Arquitectura del Modelo

El modelo (SMAWXVectorModel) toma los embeddings de VGGish y los procesa para hacer las predicciones:

```
Entrada: Embeddings VGGish [T ventanas, 128 valores]
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │  BatchNorm1d                        │  Normaliza los datos para
    │  (normalización por lotes)          │  estabilizar el entrenamiento
    └─────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │  XVector1D                          │  3 capas convolucionales que
    │  - Conv1D: 128 → 256 canales        │  detectan patrones temporales
    │  - Conv1D: 256 → 256 canales        │  en el audio (como filtros que
    │  - Conv1D: 256 → 512 canales        │  buscan características específicas)
    │  Cada capa incluye BatchNorm + ReLU │
    └─────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │  StatsPooling                       │  Resume toda la secuencia en
    │  (agrupación estadística)           │  un solo vector calculando
    │  Calcula media y desviación         │  media y desviación estándar
    │  Salida: 512×2 = 1024 valores       │  (512 medias + 512 desviaciones)
    └─────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐
    │  MultiHeadClassifier                │  Capa densa que reduce a 256
    │  (clasificador multi-tarea)         │  valores y luego predice las
    │  - FC: 1024 → 256 + ReLU            │  3 tareas simultáneamente:
    │  - FC: 256 → 3 (Espesor)            │
    │  - FC: 256 → 4 (Electrodo)          │
    │  - FC: 256 → 2 (Corriente)          │
    └─────────────────────────────────────┘
```

**Términos explicados:**

- **Conv1D:** Capa convolucional 1D. Aplica filtros que se deslizan sobre la secuencia temporal para detectar patrones.
- **BatchNorm:** Normaliza los datos en cada lote para que el entrenamiento sea más estable y rápido.
- **ReLU:** Función de activación que introduce no-linealidad (permite al modelo aprender relaciones complejas). Convierte valores negativos en cero.
- **FC (Fully Connected):** Capa densa donde cada neurona se conecta con todas las del nivel anterior.
- **Logits:** Valores numéricos crudos (sin normalizar) que salen de la última capa del modelo antes de aplicar softmax. Representan la "confianza" del modelo en cada clase. Valores más altos indican mayor confianza. Por ejemplo, logits [2.1, 0.5, -1.3] significan que el modelo confía más en la primera clase.

---

### Paso 4: Estrategia de Entrenamiento

Los modelos fueron entrenados usando K-Fold Cross-Validation con PyTorch.

#### 4.1 Validación Cruzada K-Fold

En lugar de entrenar un solo modelo, se entrenan **5 modelos diferentes** usando K-Fold Cross-Validation:

```
Datos de entrenamiento (train.csv)
┌─────┬─────┬─────┬─────┬─────┐
│ F1  │ F2  │ F3  │ F4  │ F5  │   <- 5 partes (folds)
└─────┴─────┴─────┴─────┴─────┘

Modelo 1: Entrena con F2,F3,F4,F5 | Valida con F1
Modelo 2: Entrena con F1,F3,F4,F5 | Valida con F2
Modelo 3: Entrena con F1,F2,F4,F5 | Valida con F3
Modelo 4: Entrena con F1,F2,F3,F5 | Valida con F4
Modelo 5: Entrena con F1,F2,F3,F4 | Valida con F5
```

**Ventaja:** Cada modelo ve diferentes datos de validación, lo que aumenta la diversidad del conjunto (ensemble) y reduce el sobreajuste (cuando el modelo memoriza los datos de entrenamiento en lugar de aprender patrones generales).

#### 4.2 Hiperparámetros de Entrenamiento

| Parámetro       | Valor      | Explicación                                                 |
| --------------- | ---------- | ----------------------------------------------------------- |
| Épocas          | 100 máximo | Número de veces que el modelo ve todos los datos            |
| Batch size      | 32         | Cantidad de ejemplos procesados antes de actualizar pesos   |
| Learning rate   | 0.001      | Qué tan grandes son los pasos de aprendizaje                |
| Weight decay    | 0.0001     | Penalización para evitar pesos muy grandes (regularización) |
| Label smoothing | 0.1        | Suaviza las etiquetas para evitar sobreconfianza            |

#### 4.3 Optimizador: AdamW

AdamW es un algoritmo que ajusta los pesos del modelo durante el entrenamiento. Combina:

- **Momentum:** Acumula velocidad en direcciones consistentes
- **Adaptativo:** Ajusta el learning rate para cada parámetro
- **Weight decay:** Penaliza pesos grandes para regularizar

#### 4.4 Función de Pérdida: CrossEntropyLoss

Mide qué tan lejos están las predicciones de las etiquetas correctas. El modelo intenta minimizar este valor.

**Label smoothing (suavizado de etiquetas):** En lugar de decir "esta es 100% clase A", dice "esta es 90% clase A y 10% distribuido en otras". Esto evita que el modelo sea demasiado confiado y mejora la generalización.

#### 4.5 Balanceo de Clases (Class Weighting)

Si hay más ejemplos de una clase que de otra, el modelo tendería a predecir siempre la clase mayoritaria. Para evitarlo, se asignan pesos inversamente proporcionales a la frecuencia de cada clase.

Ejemplo: Si hay 1000 ejemplos de DC y 200 de AC, los ejemplos de AC recibirán más peso en la función de pérdida.

#### 4.6 Early Stopping (Parada Temprana)

El entrenamiento se detiene automáticamente si el rendimiento en validación no mejora durante 15 épocas consecutivas. Esto previene el sobreajuste.

**Métrica monitoreada:** F1-score macro (promedio del F1 de todas las clases, dando igual importancia a clases minoritarias).

#### 4.7 Stochastic Weight Averaging (SWA)

A partir de la época 5, el sistema guarda una versión promediada de los pesos del modelo. Al final del entrenamiento, usa estos pesos promediados que suelen generalizar mejor que los últimos pesos.

**Analogía:** Es como tomar fotos del modelo en diferentes momentos y combinarlas para obtener una imagen más estable.

#### 4.8 Proceso de una Época

```
Para cada época:
│
├─ FASE DE ENTRENAMIENTO
│   Para cada lote (batch) de 32 audios:
│   1. Cargar audios y extraer embeddings VGGish
│   2. Pasar embeddings por el modelo → obtener predicciones
│   3. Calcular pérdida (error) para las 3 tareas
│   4. Retropropagar el error (calcular cómo ajustar cada peso)
│   5. Actualizar pesos del modelo
│
├─ FASE DE VALIDACIÓN
│   Para cada lote del conjunto de validación:
│   1. Pasar datos por el modelo (sin actualizar pesos)
│   2. Calcular métricas (accuracy, F1-score)
│
├─ VERIFICAR EARLY STOPPING
│   ¿Mejoró el F1-score respecto a la mejor época anterior?
│   - Sí: Guardar modelo, reiniciar contador de paciencia
│   - No: Incrementar contador. Si llega a 15, detener.
│
└─ ACTUALIZAR SWA (si época >= 5)
    Promediar pesos actuales con pesos acumulados
```

**Salida:** 5 modelos guardados en `models/model_fold_{0-4}.pth`

---

### Paso 5: Predicción con Ensemble

**Archivo:** `predecir.py`

Para predecir, se combinan los 5 modelos usando **Soft Voting**:

```
Audio de entrada
      │
      ▼
┌─────────────────────────────┐
│ Extraer embeddings VGGish   │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Modelo 1 → logits₁   (valores antes de softmax)│
│  Modelo 2 → logits₂                             │
│  Modelo 3 → logits₃                             │
│  Modelo 4 → logits₄                             │
│  Modelo 5 → logits₅                             │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Promediar logits: (l₁ + l₂ + l₃ + l₄ + l₅) / 5 │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│  argmax: seleccionar la clase con mayor valor   │
└─────────────────────────────────────────────────┘
      │
      ▼
Predicción final: Espesor, Electrodo, Corriente
```

**Logits:** Valores numéricos que el modelo produce antes de convertirlos en probabilidades. Representan la "confianza" del modelo en cada clase.

**Soft Voting vs Hard Voting:**

- Hard Voting: Cada modelo vota por una clase, gana la mayoría
- Soft Voting: Se promedian los valores de confianza antes de decidir

Soft Voting es más robusto porque considera la confianza de cada modelo.

**Modos de uso:**

```bash
python 5seg/predecir.py                      # Muestra 10 predicciones aleatorias
python 5seg/predecir.py --audio ruta.wav     # Predice un archivo específico
python 5seg/predecir.py --evaluar            # Evalúa en conjunto holdout
```

---

## 5. Métricas de Evaluación

| Métrica       | Descripción                                             |
| ------------- | ------------------------------------------------------- |
| **Accuracy**  | Porcentaje de predicciones correctas sobre el total     |
| **Precision** | De las predicciones positivas, cuántas fueron correctas |
| **Recall**    | De los casos reales positivos, cuántos se detectaron    |

---

## 6. Reproducibilidad

Para garantizar resultados consistentes:

- **Semilla fija** (`RANDOM_SEED = 42`) en todos los scripts
- **Misma división** de datos si la estructura de carpetas no cambia
- **Arquitectura e hiperparámetros** documentados

---

## 7. Archivos Generados

```
5seg/
├── completo.csv          # Todos los datos con su split asignado
├── train.csv             # Datos de entrenamiento
├── test.csv              # Datos de validación durante desarrollo
├── holdout.csv           # Datos de evaluación final
├── results.json          # Métricas del entrenamiento
├── infer.json            # Resultados de predicción
└── models/
    ├── model_fold_0.pth  # Modelo entrenado fold 0
    ├── model_fold_1.pth  # Modelo entrenado fold 1
    ├── model_fold_2.pth  # Modelo entrenado fold 2
    ├── model_fold_3.pth  # Modelo entrenado fold 3
    └── model_fold_4.pth  # Modelo entrenado fold 4
```
