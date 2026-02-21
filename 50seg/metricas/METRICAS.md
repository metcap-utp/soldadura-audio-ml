# Métricas de Clasificación SMAW - 50seg

**Fecha de evaluación:** 2026-02-19 18:40:04

**Configuración:**
- Duración de segmento: 50.0s
- Número de muestras (blind): 59
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.6780 | 0.6835 |
| Electrode Type | 0.8644 | 0.8837 |
| Current Type | 0.9831 | 0.9803 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.6780
- **Macro F1-Score:** 0.6835

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 15 | 0 | 1 |
| **Placa_3mm** | 1 | 12 | 2 |
| **Placa_6mm** | 5 | 10 | 13 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.7143 | 0.9375 | 0.8108 | 16 |
| Placa_3mm | 0.5455 | 0.8000 | 0.6486 | 15 |
| Placa_6mm | 0.8125 | 0.4643 | 0.5909 | 28 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8644
- **Macro F1-Score:** 0.8837

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 7 | 0 | 0 | 0 |
| **E6011** | 0 | 17 | 0 | 0 |
| **E6013** | 0 | 0 | 13 | 1 |
| **E7018** | 0 | 2 | 5 | 14 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 1.0000 | 1.0000 | 1.0000 | 7 |
| E6011 | 0.8947 | 1.0000 | 0.9444 | 17 |
| E6013 | 0.7222 | 0.9286 | 0.8125 | 14 |
| E7018 | 0.9333 | 0.6667 | 0.7778 | 21 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9831
- **Macro F1-Score:** 0.9803

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 18 | 0 |
| **DC** | 1 | 40 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9474 | 1.0000 | 0.9730 | 18 |
| DC | 1.0000 | 0.9756 | 0.9877 | 41 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
