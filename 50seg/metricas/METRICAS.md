# Métricas de Clasificación SMAW - 50seg

**Fecha de evaluación:** 2026-02-07 01:21:30

**Configuración:**
- Duración de segmento: 50.0s
- Número de muestras (blind): 89
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.5730 | 0.5851 |
| Electrode Type | 0.8202 | 0.8485 |
| Current Type | 0.9551 | 0.9505 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.5730
- **Macro F1-Score:** 0.5851

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 17 | 0 | 4 |
| **Placa_3mm** | 1 | 17 | 2 |
| **Placa_6mm** | 11 | 20 | 17 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.5862 | 0.8095 | 0.6800 | 21 |
| Placa_3mm | 0.4595 | 0.8500 | 0.5965 | 20 |
| Placa_6mm | 0.7391 | 0.3542 | 0.4789 | 48 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8202
- **Macro F1-Score:** 0.8485

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 10 | 0 | 0 | 0 |
| **E6011** | 0 | 25 | 0 | 0 |
| **E6013** | 0 | 0 | 17 | 1 |
| **E7018** | 0 | 2 | 13 | 21 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 1.0000 | 1.0000 | 1.0000 | 10 |
| E6011 | 0.9259 | 1.0000 | 0.9615 | 25 |
| E6013 | 0.5667 | 0.9444 | 0.7083 | 18 |
| E7018 | 0.9545 | 0.5833 | 0.7241 | 36 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9551
- **Macro F1-Score:** 0.9505

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 29 | 0 |
| **DC** | 4 | 56 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.8788 | 1.0000 | 0.9355 | 29 |
| DC | 1.0000 | 0.9333 | 0.9655 | 60 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
