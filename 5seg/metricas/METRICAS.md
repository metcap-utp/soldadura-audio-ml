# Métricas de Clasificación SMAW - 5seg

**Fecha de evaluación:** 2026-02-01 01:25:44

**Configuración:**
- Duración de segmento: 5.0s
- Número de muestras (blind): 951
- Número de modelos (ensemble): 20
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 74.13% | 0.7469 |
| Electrode Type | 85.70% | 0.8446 |
| Current Type | 93.69% | 0.9327 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 74.13%
- **Macro F1-Score:** 0.7469

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 207 | 13 | 55 |
| **Placa_3mm** | 8 | 232 | 7 |
| **Placa_6mm** | 101 | 62 | 266 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6551 | 0.7527 | 0.7005 | 275 |
| Placa_3mm | 0.7557 | 0.9393 | 0.8375 | 247 |
| Placa_6mm | 0.8110 | 0.6200 | 0.7028 | 429 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 85.70%
- **Macro F1-Score:** 0.8446

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 109 | 4 | 5 | 0 |
| **E6011** | 8 | 291 | 7 | 4 |
| **E6013** | 4 | 8 | 220 | 9 |
| **E7018** | 41 | 28 | 18 | 195 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6728 | 0.9237 | 0.7786 | 118 |
| E6011 | 0.8792 | 0.9387 | 0.9080 | 310 |
| E6013 | 0.8800 | 0.9129 | 0.8961 | 241 |
| E7018 | 0.9375 | 0.6915 | 0.7959 | 282 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 93.69%
- **Macro F1-Score:** 0.9327

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 326 | 4 |
| **DC** | 56 | 565 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.8534 | 0.9879 | 0.9157 | 330 |
| DC | 0.9930 | 0.9098 | 0.9496 | 621 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
