# Métricas de Clasificación SMAW - 10seg

**Fecha de evaluación:** 2026-02-07 01:04:56

**Configuración:**
- Duración de segmento: 10.0s
- Número de muestras (blind): 878
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7711 | 0.7752 |
| Electrode Type | 0.8827 | 0.8758 |
| Current Type | 0.9647 | 0.9619 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7711
- **Macro F1-Score:** 0.7752

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 212 | 7 | 32 |
| **Placa_3mm** | 11 | 212 | 1 |
| **Placa_6mm** | 77 | 73 | 253 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.7067 | 0.8446 | 0.7695 | 251 |
| Placa_3mm | 0.7260 | 0.9464 | 0.8217 | 224 |
| Placa_6mm | 0.8846 | 0.6278 | 0.7344 | 403 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8827
- **Macro F1-Score:** 0.8758

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 108 | 0 | 0 | 0 |
| **E6011** | 2 | 279 | 2 | 2 |
| **E6013** | 1 | 2 | 208 | 6 |
| **E7018** | 35 | 35 | 18 | 180 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.7397 | 1.0000 | 0.8504 | 108 |
| E6011 | 0.8829 | 0.9789 | 0.9285 | 285 |
| E6013 | 0.9123 | 0.9585 | 0.9348 | 217 |
| E7018 | 0.9574 | 0.6716 | 0.7895 | 268 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9647
- **Macro F1-Score:** 0.9619

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 305 | 1 |
| **DC** | 30 | 542 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9104 | 0.9967 | 0.9516 | 306 |
| DC | 0.9982 | 0.9476 | 0.9722 | 572 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
