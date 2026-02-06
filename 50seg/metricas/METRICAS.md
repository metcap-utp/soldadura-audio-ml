# Métricas de Clasificación SMAW - 50seg

**Fecha de evaluación:** 2026-02-05 13:27:29

**Configuración:**
- Duración de segmento: 50.0s
- Número de muestras (blind): 59
- Número de modelos (ensemble): 5
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 66.10% | 0.6668 |
| Electrode Type | 84.75% | 0.8585 |
| Current Type | 96.61% | 0.9612 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 66.10%
- **Macro F1-Score:** 0.6668

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 15 | 0 | 1 |
| **Placa_3mm** | 1 | 12 | 2 |
| **Placa_6mm** | 5 | 11 | 12 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.7143 | 0.9375 | 0.8108 | 16 |
| Placa_3mm | 0.5217 | 0.8000 | 0.6316 | 15 |
| Placa_6mm | 0.8000 | 0.4286 | 0.5581 | 28 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 84.75%
- **Macro F1-Score:** 0.8585

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 7 | 0 | 0 | 0 |
| **E6011** | 0 | 17 | 0 | 0 |
| **E6013** | 0 | 0 | 13 | 1 |
| **E7018** | 1 | 3 | 4 | 13 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.8750 | 1.0000 | 0.9333 | 7 |
| E6011 | 0.8500 | 1.0000 | 0.9189 | 17 |
| E6013 | 0.7647 | 0.9286 | 0.8387 | 14 |
| E7018 | 0.9286 | 0.6190 | 0.7429 | 21 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 96.61%
- **Macro F1-Score:** 0.9612

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 18 | 0 |
| **DC** | 2 | 39 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9000 | 1.0000 | 0.9474 | 18 |
| DC | 1.0000 | 0.9512 | 0.9750 | 41 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
