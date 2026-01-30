# Métricas de Clasificación SMAW - 1seg

**Fecha de evaluación:** 2026-01-23 22:44:47

**Configuración:**
- Duración de segmento: 1.0s
- Número de muestras (blind): 4988
- Número de modelos (ensemble): 5
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 29.05% | 0.1501 |
| Electrode Type | 12.49% | 0.0555 |
| Current Type | 34.64% | 0.2573 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 29.05%
- **Macro F1-Score:** 0.1501

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 1449 | 0 | 0 |
| **Placa_3mm** | 1313 | 0 | 0 |
| **Placa_6mm** | 2226 | 0 | 0 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.2905 | 1.0000 | 0.4502 | 1449 |
| Placa_3mm | 0.0000 | 0.0000 | 0.0000 | 1313 |
| Placa_6mm | 0.0000 | 0.0000 | 0.0000 | 2226 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 12.49%
- **Macro F1-Score:** 0.0555

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 623 | 0 | 0 | 0 |
| **E6011** | 1624 | 0 | 0 | 0 |
| **E6013** | 1285 | 0 | 0 | 0 |
| **E7018** | 1456 | 0 | 0 | 0 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.1249 | 1.0000 | 0.2221 | 623 |
| E6011 | 0.0000 | 0.0000 | 0.0000 | 1624 |
| E6013 | 0.0000 | 0.0000 | 0.0000 | 1285 |
| E7018 | 0.0000 | 0.0000 | 0.0000 | 1456 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 34.64%
- **Macro F1-Score:** 0.2573

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 1728 | 0 |
| **DC** | 3260 | 0 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.3464 | 1.0000 | 0.5146 | 1728 |
| DC | 0.0000 | 0.0000 | 0.0000 | 3260 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
