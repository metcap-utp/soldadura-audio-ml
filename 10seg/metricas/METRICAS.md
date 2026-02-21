# Métricas de Clasificación SMAW - 10seg

**Fecha de evaluación:** 2026-02-19 18:37:58

**Configuración:**
- Duración de segmento: 10.0s
- Número de muestras (blind): 447
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7517 | 0.7578 |
| Electrode Type | 0.8702 | 0.8592 |
| Current Type | 0.9776 | 0.9759 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7517
- **Macro F1-Score:** 0.7578

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 107 | 2 | 20 |
| **Placa_3mm** | 6 | 106 | 2 |
| **Placa_6mm** | 47 | 34 | 123 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6687 | 0.8295 | 0.7405 | 129 |
| Placa_3mm | 0.7465 | 0.9298 | 0.8281 | 114 |
| Placa_6mm | 0.8483 | 0.6029 | 0.7049 | 204 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8702
- **Macro F1-Score:** 0.8592

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 53 | 0 | 2 | 0 |
| **E6011** | 2 | 140 | 2 | 2 |
| **E6013** | 1 | 3 | 105 | 2 |
| **E7018** | 22 | 17 | 5 | 91 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6795 | 0.9636 | 0.7970 | 55 |
| E6011 | 0.8750 | 0.9589 | 0.9150 | 146 |
| E6013 | 0.9211 | 0.9459 | 0.9333 | 111 |
| E7018 | 0.9579 | 0.6741 | 0.7913 | 135 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9776
- **Macro F1-Score:** 0.9759

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 158 | 0 |
| **DC** | 10 | 279 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9405 | 1.0000 | 0.9693 | 158 |
| DC | 1.0000 | 0.9654 | 0.9824 | 289 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
