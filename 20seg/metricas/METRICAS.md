# Métricas de Clasificación SMAW - 20seg

**Fecha de evaluación:** 2026-02-07 01:11:27

**Configuración:**
- Duración de segmento: 20.0s
- Número de muestras (blind): 375
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7440 | 0.7488 |
| Electrode Type | 0.8587 | 0.8603 |
| Current Type | 0.9627 | 0.9602 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7440
- **Macro F1-Score:** 0.7488

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 90 | 3 | 12 |
| **Placa_3mm** | 5 | 86 | 1 |
| **Placa_6mm** | 39 | 36 | 103 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6716 | 0.8571 | 0.7531 | 105 |
| Placa_3mm | 0.6880 | 0.9348 | 0.7926 | 92 |
| Placa_6mm | 0.8879 | 0.5787 | 0.7007 | 178 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8587
- **Macro F1-Score:** 0.8603

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 45 | 0 | 0 | 0 |
| **E6011** | 1 | 119 | 2 | 0 |
| **E6013** | 1 | 0 | 82 | 4 |
| **E7018** | 7 | 16 | 22 | 76 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.8333 | 1.0000 | 0.9091 | 45 |
| E6011 | 0.8815 | 0.9754 | 0.9261 | 122 |
| E6013 | 0.7736 | 0.9425 | 0.8497 | 87 |
| E7018 | 0.9500 | 0.6281 | 0.7562 | 121 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9627
- **Macro F1-Score:** 0.9602

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 134 | 0 |
| **DC** | 14 | 227 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9054 | 1.0000 | 0.9504 | 134 |
| DC | 1.0000 | 0.9419 | 0.9701 | 241 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
