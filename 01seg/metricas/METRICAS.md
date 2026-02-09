# Métricas de Clasificación SMAW - 01seg

**Fecha de evaluación:** 2026-02-07 00:35:22

**Configuración:**
- Duración de segmento: 1.0s
- Número de muestras (blind): 9958
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.2904 | 0.1500 |
| Electrode Type | 0.1248 | 0.0555 |
| Current Type | 0.3466 | 0.2574 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.2904
- **Macro F1-Score:** 0.1500

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 2892 | 0 | 0 |
| **Placa_3mm** | 2620 | 0 | 0 |
| **Placa_6mm** | 4446 | 0 | 0 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.2904 | 1.0000 | 0.4501 | 2892 |
| Placa_3mm | 0.0000 | 0.0000 | 0.0000 | 2620 |
| Placa_6mm | 0.0000 | 0.0000 | 0.0000 | 4446 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.1248
- **Macro F1-Score:** 0.0555

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 1243 | 0 | 0 | 0 |
| **E6011** | 3242 | 0 | 0 | 0 |
| **E6013** | 2566 | 0 | 0 | 0 |
| **E7018** | 2907 | 0 | 0 | 0 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.1248 | 1.0000 | 0.2219 | 1243 |
| E6011 | 0.0000 | 0.0000 | 0.0000 | 3242 |
| E6013 | 0.0000 | 0.0000 | 0.0000 | 2566 |
| E7018 | 0.0000 | 0.0000 | 0.0000 | 2907 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.3466
- **Macro F1-Score:** 0.2574

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 3451 | 0 |
| **DC** | 6507 | 0 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.3466 | 1.0000 | 0.5147 | 3451 |
| DC | 0.0000 | 0.0000 | 0.0000 | 6507 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
