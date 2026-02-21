# Métricas de Clasificación SMAW - 5seg

**Fecha de evaluación:** 2026-02-19 18:36:59

**Configuración:**
- Duración de segmento: 5.0s
- Número de muestras (blind): 951
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7497 | 0.7558 |
| Electrode Type | 0.8570 | 0.8464 |
| Current Type | 0.9369 | 0.9327 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7497
- **Macro F1-Score:** 0.7558

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 214 | 13 | 48 |
| **Placa_3mm** | 6 | 235 | 6 |
| **Placa_6mm** | 106 | 59 | 264 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6564 | 0.7782 | 0.7121 | 275 |
| Placa_3mm | 0.7655 | 0.9514 | 0.8484 | 247 |
| Placa_6mm | 0.8302 | 0.6154 | 0.7068 | 429 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8570
- **Macro F1-Score:** 0.8464

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 111 | 2 | 4 | 1 |
| **E6011** | 12 | 287 | 7 | 4 |
| **E6013** | 3 | 10 | 220 | 8 |
| **E7018** | 37 | 26 | 22 | 197 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6810 | 0.9407 | 0.7900 | 118 |
| E6011 | 0.8831 | 0.9258 | 0.9039 | 310 |
| E6013 | 0.8696 | 0.9129 | 0.8907 | 241 |
| E7018 | 0.9381 | 0.6986 | 0.8008 | 282 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9369
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
