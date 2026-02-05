# Métricas de Clasificación SMAW - 10seg

**Fecha de evaluación:** 2026-02-01 01:31:06

**Configuración:**
- Duración de segmento: 10.0s
- Número de muestras (blind): 447
- Número de modelos (ensemble): 20
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 75.62% | 0.7622 |
| Electrode Type | 86.80% | 0.8600 |
| Current Type | 95.75% | 0.9546 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 75.62%
- **Macro F1-Score:** 0.7622

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 106 | 2 | 21 |
| **Placa_3mm** | 6 | 108 | 0 |
| **Placa_6mm** | 46 | 34 | 124 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6709 | 0.8217 | 0.7387 | 129 |
| Placa_3mm | 0.7500 | 0.9474 | 0.8372 | 114 |
| Placa_6mm | 0.8552 | 0.6078 | 0.7106 | 204 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 86.80%
- **Macro F1-Score:** 0.8600

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 54 | 0 | 1 | 0 |
| **E6011** | 2 | 140 | 2 | 2 |
| **E6013** | 0 | 3 | 106 | 2 |
| **E7018** | 19 | 18 | 10 | 88 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.7200 | 0.9818 | 0.8308 | 55 |
| E6011 | 0.8696 | 0.9589 | 0.9121 | 146 |
| E6013 | 0.8908 | 0.9550 | 0.9217 | 111 |
| E7018 | 0.9565 | 0.6519 | 0.7753 | 135 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 95.75%
- **Macro F1-Score:** 0.9546

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 158 | 0 |
| **DC** | 19 | 270 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.8927 | 1.0000 | 0.9433 | 158 |
| DC | 1.0000 | 0.9343 | 0.9660 | 289 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
