# Clasificación de Audio SMAW

Sistema de clasificación de audio de soldadura SMAW (Shielded Metal Arc Welding) usando deep learning con arquitectura X-Vector y ensemble de modelos con K-Fold Cross-Validation.

## Objetivos

Clasificar audio de soldadura en tres tareas:

- **Plate Thickness**: Espesor de placa (3mm, 6mm, 12mm)
- **Electrode Type**: Tipo de electrodo (E6010, E6011, E6013, E7018)
- **Current Type**: Tipo de corriente (AC, DC)

## Estructura del Proyecto

```
soldadura/
├── audio/                    # Audios originales completos
│   ├── Placa_3mm/
│   ├── Placa_6mm/
│   └── Placa_12mm/
├── 1seg/                     # Experimentos con segmentos de 1 segundo
├── 2seg/                     # Experimentos con segmentos de 2 segundos
├── 5seg/                     # Experimentos con segmentos de 5 segundos
├── 10seg/                    # Experimentos con segmentos de 10 segundos
├── 20seg/                    # Experimentos con segmentos de 20 segundos
├── 30seg/                    # Experimentos con segmentos de 30 segundos
├── 50seg/                    # Experimentos con segmentos de 50 segundos
├── modelo.py                 # Arquitectura X-Vector
├── utils/                    # Utilidades de audio
└── scripts/                  # Scripts de análisis y visualización
    ├── graficar_folds.py     # Métricas vs K-folds
    └── graficar_duraciones.py # Métricas vs duración de clips
```

### Estructura de modelos

```
5seg/
├── models/
│   ├── 3-fold/              # Modelos con k=3
│   │   ├── model_fold_0.pth
│   │   ├── model_fold_1.pth
│   │   └── model_fold_2.pth
│   ├── 5-fold/              # Modelos con k=5
│   │   └── ...
│   └── 10-fold/             # Modelos con k=10
│       └── ...
├── results.json             # Métricas de entrenamiento
└── infer.json               # Métricas de evaluación
```

## Inicio Rápido

### 1. Requisitos

```bash
pip install torch numpy pandas librosa tensorflow tensorflow-hub scikit-learn
```

### 2. Generar splits de datos

```bash
cd 5seg  # o cualquier duración (1seg, 2seg, 10seg, 30seg)
python generar_splits.py
```

Esto genera:

- `train.csv` - Datos de entrenamiento
- `test.csv` - Datos de prueba
- `blind.csv` - Datos de evaluación final (nunca vistos)

### 3. Entrenar modelos

```bash
# Entrenar con 5 folds (por defecto)
python entrenar.py

# Entrenar con diferente número de folds
python entrenar.py --k-folds 3
python entrenar.py --k-folds 10
```

Los modelos se guardan en carpetas según k-folds:

- `5seg/5-fold/` - Modelos entrenados con 5 folds
- `5seg/3-fold/` - Modelos entrenados con 3 folds
- etc.

### 4. Evaluar modelos

```bash
# Evaluar ensemble de 5-fold (por defecto)
python infer.py --evaluar

# Evaluar con diferente k-folds
python infer.py --evaluar --k-folds 3
python infer.py --evaluar --k-folds 10
```

### 5. Predecir un audio específico

```bash
python infer.py --audio ruta/al/archivo.wav
python infer.py --audio ruta/al/archivo.wav --k-folds 3
```

## Resultados

Los resultados se guardan automáticamente en:

- `results.json` - Métricas de entrenamiento (acumulativo)
- `infer.json` - Métricas de evaluación (acumulativo)
- `METRICAS.md` - Documento Markdown con matrices de confusión

Cada entrada incluye información para identificar el experimento:

- Número de folds
- Duración de segmento
- Timestamp
- Métricas por tarea y por fold

### Comparación de K (10 segundos)

Se añadió una tabla comparativa de K para 10seg en [RESULTADOS.md](RESULTADOS.md).

## Comparar diferentes valores de K

```bash
# Entrenar con diferentes k
python entrenar.py --k-folds 3
python entrenar.py --k-folds 5
python entrenar.py --k-folds 10

# Evaluar cada configuración
python infer.py --evaluar --k-folds 3
python infer.py --evaluar --k-folds 5
python infer.py --evaluar --k-folds 10
```

Luego revisa `results.json` para comparar las métricas de cada configuración.

## Visualización de Resultados

### Métricas vs Número de Folds

```bash
cd scripts

# Graficar métricas vs k-folds para una duración específica
python graficar_folds.py 5seg
python graficar_folds.py 10seg --save              # Guarda imagen
python graficar_folds.py 5seg --metric accuracy    # Solo accuracy
```

### Métricas vs Duración de Clips

```bash
cd scripts

# Comparar rendimiento entre diferentes duraciones
python graficar_duraciones.py
python graficar_duraciones.py --k-folds 5          # Solo resultados de 5-fold
python graficar_duraciones.py --save               # Guarda imagen
python graficar_duraciones.py --no-plot            # Solo tabla resumen
```

## Parámetros de Entrenamiento

| Parámetro       | Valor     |
| --------------- | --------- |
| Batch Size      | 32        |
| Epochs          | 100       |
| Learning Rate   | 1e-3      |
| Early Stopping  | 15 epochs |
| Optimizer       | AdamW     |
| Label Smoothing | 0.1       |

## Referencias

- Snyder et al. (2018) - X-Vectors
- Dietterich (2000) - Ensemble Methods
