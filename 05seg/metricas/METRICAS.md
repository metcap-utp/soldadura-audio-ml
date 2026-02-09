# Métricas de Clasificación SMAW - 05seg

**Fecha de evaluación:** 2026-02-07 00:57:10

**Configuración:**
- Duración de segmento: 5.0s
- Número de muestras (blind): 1885
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7825 | 0.7888 |
| Electrode Type | 0.8568 | 0.8475 |
| Current Type | 0.9135 | 0.9091 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7825
- **Macro F1-Score:** 0.7888

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 467 | 11 | 67 |
| **Placa_3mm** | 13 | 467 | 8 |
| **Placa_6mm** | 199 | 112 | 541 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6878 | 0.8569 | 0.7631 | 545 |
| Placa_3mm | 0.7915 | 0.9570 | 0.8664 | 488 |
| Placa_6mm | 0.8782 | 0.6350 | 0.7371 | 852 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8568
- **Macro F1-Score:** 0.8475

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 213 | 9 | 8 | 2 |
| **E6011** | 14 | 585 | 9 | 4 |
| **E6013** | 4 | 14 | 445 | 16 |
| **E7018** | 62 | 82 | 46 | 372 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.7270 | 0.9181 | 0.8114 | 232 |
| E6011 | 0.8478 | 0.9559 | 0.8986 | 612 |
| E6013 | 0.8760 | 0.9290 | 0.9017 | 479 |
| E7018 | 0.9442 | 0.6619 | 0.7782 | 562 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9135
- **Macro F1-Score:** 0.9091

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 654 | 2 |
| **DC** | 161 | 1068 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.8025 | 0.9970 | 0.8892 | 656 |
| DC | 0.9981 | 0.8690 | 0.9291 | 1229 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
