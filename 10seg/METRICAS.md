# Métricas de Clasificación SMAW - 10seg

**Fecha de evaluación:** 2026-01-21 22:17:34

**Configuración:**
- Duración de segmento: 10.0s
- Número de muestras (holdout): 447
- Número de modelos (ensemble): 5
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 75.39% | 0.7601 |
| Electrode Type | 86.13% | 0.8525 |
| Current Type | 97.09% | 0.9687 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 75.39%
- **Macro F1-Score:** 0.7601

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 102 | 3 | 24 |
| **Placa_3mm** | 3 | 111 | 0 |
| **Placa_6mm** | 46 | 34 | 124 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6755 | 0.7907 | 0.7286 | 129 |
| Placa_3mm | 0.7500 | 0.9737 | 0.8473 | 114 |
| Placa_6mm | 0.8378 | 0.6078 | 0.7045 | 204 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 86.13%
- **Macro F1-Score:** 0.8525

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 54 | 0 | 1 | 0 |
| **E6011** | 2 | 141 | 1 | 2 |
| **E6013** | 1 | 4 | 101 | 5 |
| **E7018** | 20 | 17 | 9 | 89 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.7013 | 0.9818 | 0.8182 | 55 |
| E6011 | 0.8704 | 0.9658 | 0.9156 | 146 |
| E6013 | 0.9018 | 0.9099 | 0.9058 | 111 |
| E7018 | 0.9271 | 0.6593 | 0.7706 | 135 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 97.09%
- **Macro F1-Score:** 0.9687

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 158 | 0 |
| **DC** | 13 | 276 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9240 | 1.0000 | 0.9605 | 158 |
| DC | 1.0000 | 0.9550 | 0.9770 | 289 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **holdout** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
