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
├── entrenar.py               # Script de entrenamiento (consolidado)
├── generar_splits.py          # Generación de splits (consolidado)
├── infer.py                   # Inferencia y evaluación (consolidado)
├── modelo.py                  # Arquitectura X-Vector
├── audio/                     # Audios originales completos
│   ├── Placa_3mm/
│   ├── Placa_6mm/
│   └── Placa_12mm/
├── {N}seg/                    # Datos y modelos por duración (1,2,5,10,20,30,50)
│   ├── train.csv / test.csv / blind.csv
│   ├── results.json / infer.json / data_stats.json
│   ├── models/
│   │   └── k{K}_overlap_{ratio}/    # Modelos organizados por K y overlap
│   ├── metricas/
│   └── confusion_matrices/
├── utils/                     # Utilidades de audio y timing
└── scripts/                   # Scripts de análisis y visualización
    ├── graficar_folds.py      # Métricas vs K-folds
    ├── graficar_duraciones.py # Métricas vs duración de clips
    └── graficar_overlap.py    # Comparación de overlap (heatmaps, curvas)
```

## Inicio Rápido

### 1. Requisitos

```bash
pip install torch numpy pandas librosa tensorflow tensorflow-hub scikit-learn
```

### 2. Generar splits de datos

```bash
# Desde la raíz del proyecto
python generar_splits.py --duration 5 --overlap 0.5
python generar_splits.py --duration 10 --overlap 0.0
```

Esto genera en `{N}seg/`:

- `train.csv` - Datos de entrenamiento
- `test.csv` - Datos de prueba
- `blind.csv` - Datos de evaluación final (nunca vistos)

### 3. Entrenar modelos

```bash
# Entrenar con 5 folds y overlap 50%
python entrenar.py --duration 5 --overlap 0.5 --k-folds 5

# Entrenar con 10 folds sin overlap
python entrenar.py --duration 10 --overlap 0.0 --k-folds 10
```

Los modelos se guardan en:

- `05seg/models/k05_overlap_0.5/` - 5-fold con overlap 0.5
- `10seg/models/k10_overlap_0.0/` - 10-fold sin overlap

### 4. Evaluar modelos

```bash
# Evaluar ensemble en conjunto blind
python infer.py --duration 5 --overlap 0.5 --evaluar
python infer.py --duration 5 --overlap 0.5 --evaluar --k-folds 10
```

### 5. Predecir un audio específico

```bash
python infer.py --duration 5 --overlap 0.5 --audio ruta/al/archivo.wav
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

## Comparar diferentes valores de K y overlap

```bash
# Entrenar con diferentes k y overlap
python entrenar.py --duration 5 --overlap 0.5 --k-folds 5
python entrenar.py --duration 5 --overlap 0.5 --k-folds 10
python entrenar.py --duration 5 --overlap 0.0 --k-folds 5
python entrenar.py --duration 5 --overlap 0.75 --k-folds 5

# Evaluar cada configuración
python infer.py --duration 5 --overlap 0.5 --evaluar --k-folds 5
python infer.py --duration 5 --overlap 0.5 --evaluar --k-folds 10
```

Luego revisa `{N}seg/results.json` y `{N}seg/infer.json` para comparar métricas.

## Visualización de Resultados

### Métricas vs Número de Folds

```bash
# Graficar métricas vs k-folds para una duración específica
python scripts/graficar_folds.py 05seg
python scripts/graficar_folds.py 10seg --save
```

### Métricas vs Duración de Clips

```bash
# Comparar rendimiento entre diferentes duraciones
python scripts/graficar_duraciones.py
python scripts/graficar_duraciones.py --k-folds 5 --save
```

### Comparación de Overlap

```bash
# Gráficas por duración y heatmaps duración×overlap
python scripts/graficar_overlap.py --save
python scripts/graficar_overlap.py --heatmap --save
python scripts/graficar_overlap.py --duration 5 --k-folds 10 --save
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
