# Métricas de Clasificación SMAW - 5seg

**Fecha de evaluación:** 2026-01-23 22:35:36

**Configuración:**
- Duración de segmento: 5.0s
- Número de muestras (blind): 951
- Número de modelos (ensemble): 5
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 74.66% | 0.7523 |
| Electrode Type | 84.96% | 0.8370 |
| Current Type | 93.27% | 0.9284 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 74.66%
- **Macro F1-Score:** 0.7523

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 209 | 13 | 53 |
| **Placa_3mm** | 8 | 232 | 7 |
| **Placa_6mm** | 101 | 59 | 269 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6572 | 0.7600 | 0.7049 | 275 |
| Placa_3mm | 0.7632 | 0.9393 | 0.8421 | 247 |
| Placa_6mm | 0.8176 | 0.6270 | 0.7098 | 429 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 84.96%
- **Macro F1-Score:** 0.8370

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 109 | 4 | 4 | 1 |
| **E6011** | 13 | 287 | 7 | 3 |
| **E6013** | 2 | 8 | 222 | 9 |
| **E7018** | 41 | 27 | 24 | 190 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6606 | 0.9237 | 0.7703 | 118 |
| E6011 | 0.8804 | 0.9258 | 0.9025 | 310 |
| E6013 | 0.8638 | 0.9212 | 0.8916 | 241 |
| E7018 | 0.9360 | 0.6738 | 0.7835 | 282 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 93.27%
- **Macro F1-Score:** 0.9284

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 327 | 3 |
| **DC** | 61 | 560 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.8428 | 0.9909 | 0.9109 | 330 |
| DC | 0.9947 | 0.9018 | 0.9459 | 621 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
