# Clasificación de Audio SMAW con Ensemble K-Fold

## Objetivo

Alcanzar accuracy ≥90% en las tres tareas de clasificación:

- Espesor de placa (3mm, 6mm, 12mm)
- Tipo de electrodo (E6010, E6011, E6013, E7018)
- Tipo de corriente (AC, DC)

---

## Resultados por Duración de Audio

### Comparativa de Ensemble (Soft Voting, 5 Folds)

| Duración | Placa  | Electrodo | Corriente |
| -------- | ------ | --------- | --------- |
| 5 seg    | 92.09% | 92.87%    | 98.38%    |
| 10 seg   | 98.99% | 98.12%    | 99.78%    |
| 30 seg   | 99.88% | 98.85%    | 99.77%    |

### Métricas Detalladas

#### 5 segundos

| Tarea     | Accuracy | F1-Score | Precision | Recall |
| --------- | -------- | -------- | --------- | ------ |
| Placa     | 92.09%   | 92.11%   | 92.14%    | 92.09% |
| Electrodo | 92.87%   | 92.87%   | 92.93%    | 92.87% |
| Corriente | 98.38%   | 98.39%   | 98.41%    | 98.38% |

#### 10 segundos

| Tarea     | Accuracy | F1-Score | Precision | Recall |
| --------- | -------- | -------- | --------- | ------ |
| Placa     | 98.99%   | 98.99%   | 98.99%    | 98.99% |
| Electrodo | 98.12%   | 98.12%   | 98.12%    | 98.12% |
| Corriente | 99.78%   | 99.78%   | 99.78%    | 99.78% |

#### 30 segundos

| Tarea     | Accuracy | F1-Score | Precision | Recall |
| --------- | -------- | -------- | --------- | ------ |
| Placa     | 99.88%   | 99.88%   | 99.89%    | 99.88% |
| Electrodo | 98.85%   | 98.85%   | 98.87%    | 98.85% |
| Corriente | 99.77%   | 99.77%   | 99.77%    | 99.77% |

---

## Metodología

### Arquitectura del Modelo

- **Extractor de características**: VGGish (embeddings de 128 dimensiones)
- **Clasificadores disponibles**:
  - X-Vector con StatsPooling (por defecto)
  - ECAPA-TDNN con Attentive Pooling (mayor precisión)
  - FeedForward con capas densas (baseline rápido)
- **Salidas**: Clasificador multi-cabeza (Placa, Electrodo, Corriente)

### Estrategia de Entrenamiento

- **Validación cruzada**: 5-Fold estratificado
- **Ensemble**: Votación suave (soft voting) sobre los 5 modelos
- **Épocas**: 100 (con early stopping, patience=15)
- **Batch size**: 32
- **Optimizador**: AdamW con weight decay=1e-4
- **Scheduler**: Cosine Annealing con warm restarts

### Técnicas de Regularización

- Label smoothing (0.1)
- Gradient clipping (max_norm=1.0)
- Class weights para balanceo de clases
- Multi-task learning con incertidumbre aprendible

---

## Mejora del Ensemble vs Modelos Individuales

| Duración | Placa   | Electrodo | Corriente |
| -------- | ------- | --------- | --------- |
| 5 seg    | +16.86% | +13.13%   | +3.44%    |
| 10 seg   | +19.49% | +13.58%   | +3.55%    |
| 30 seg   | +15.68% | +8.65%    | +2.42%    |

---

## Conclusiones

1. **Todas las duraciones superan el objetivo** de ≥90% en las tres tareas
2. **Mayor duración mejora resultados**: 30 seg alcanza ~99% en todas las métricas
3. **El ensemble aporta mejora significativa**: entre +8% y +19% sobre modelos individuales
4. **Corriente es la tarea más fácil**: >98% en todas las duraciones
5. **5 segundos es viable**: 92%+ es suficiente para aplicaciones en tiempo real
