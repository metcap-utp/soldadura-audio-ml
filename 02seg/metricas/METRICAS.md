# Métricas de Clasificación SMAW - 02seg

**Fecha de evaluación:** 2026-02-07 00:48:16

**Configuración:**
- Duración de segmento: 2.0s
- Número de muestras (blind): 4912
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7152 | 0.7214 |
| Electrode Type | 0.7925 | 0.7806 |
| Current Type | 0.8758 | 0.8707 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7152
- **Macro F1-Score:** 0.7214

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 1060 | 64 | 301 |
| **Placa_3mm** | 65 | 1129 | 93 |
| **Placa_6mm** | 546 | 330 | 1324 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6344 | 0.7439 | 0.6848 | 1425 |
| Placa_3mm | 0.7413 | 0.8772 | 0.8036 | 1287 |
| Placa_6mm | 0.7707 | 0.6018 | 0.6759 | 2200 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.7925
- **Macro F1-Score:** 0.7806

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 508 | 37 | 54 | 12 |
| **E6011** | 86 | 1397 | 83 | 32 |
| **E6013** | 43 | 79 | 1075 | 64 |
| **E7018** | 146 | 170 | 213 | 913 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6488 | 0.8314 | 0.7288 | 611 |
| E6011 | 0.8301 | 0.8742 | 0.8516 | 1598 |
| E6013 | 0.7544 | 0.8525 | 0.8004 | 1261 |
| E7018 | 0.8942 | 0.6331 | 0.7414 | 1442 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.8758
- **Macro F1-Score:** 0.8707

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 1664 | 40 |
| **DC** | 570 | 2638 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.7449 | 0.9765 | 0.8451 | 1704 |
| DC | 0.9851 | 0.8223 | 0.8964 | 3208 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
