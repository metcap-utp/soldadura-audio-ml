# Métricas de Clasificación SMAW - 30seg

**Fecha de evaluación:** 2026-01-23 22:40:27

**Configuración:**
- Duración de segmento: 30.0s
- Número de muestras (blind): 113
- Número de modelos (ensemble): 5
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 69.03% | 0.6946 |
| Electrode Type | 88.50% | 0.8712 |
| Current Type | 95.58% | 0.9524 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 69.03%
- **Macro F1-Score:** 0.6946

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 26 | 1 | 3 |
| **Placa_3mm** | 2 | 26 | 1 |
| **Placa_6mm** | 13 | 15 | 26 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6341 | 0.8667 | 0.7324 | 30 |
| Placa_3mm | 0.6190 | 0.8966 | 0.7324 | 29 |
| Placa_6mm | 0.8667 | 0.4815 | 0.6190 | 54 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 88.50%
- **Macro F1-Score:** 0.8712

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 14 | 0 | 0 | 0 |
| **E6011** | 0 | 37 | 0 | 0 |
| **E6013** | 1 | 0 | 22 | 1 |
| **E7018** | 6 | 1 | 4 | 27 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6667 | 1.0000 | 0.8000 | 14 |
| E6011 | 0.9737 | 1.0000 | 0.9867 | 37 |
| E6013 | 0.8462 | 0.9167 | 0.8800 | 24 |
| E7018 | 0.9643 | 0.7105 | 0.8182 | 38 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 95.58%
- **Macro F1-Score:** 0.9524

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 39 | 1 |
| **DC** | 4 | 69 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9070 | 0.9750 | 0.9398 | 40 |
| DC | 0.9857 | 0.9452 | 0.9650 | 73 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
