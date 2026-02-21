# Métricas de Clasificación SMAW - 20seg

**Fecha de evaluación:** 2026-02-19 18:38:45

**Configuración:**
- Duración de segmento: 20.0s
- Número de muestras (blind): 199
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7337 | 0.7383 |
| Electrode Type | 0.8693 | 0.8640 |
| Current Type | 0.9648 | 0.9620 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7337
- **Macro F1-Score:** 0.7383

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 50 | 1 | 5 |
| **Placa_3mm** | 4 | 44 | 1 |
| **Placa_6mm** | 23 | 19 | 52 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6494 | 0.8929 | 0.7519 | 56 |
| Placa_3mm | 0.6875 | 0.8980 | 0.7788 | 49 |
| Placa_6mm | 0.8966 | 0.5532 | 0.6842 | 94 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8693
- **Macro F1-Score:** 0.8640

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 25 | 0 | 0 | 0 |
| **E6011** | 1 | 64 | 1 | 0 |
| **E6013** | 0 | 0 | 42 | 3 |
| **E7018** | 8 | 8 | 5 | 42 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.7353 | 1.0000 | 0.8475 | 25 |
| E6011 | 0.8889 | 0.9697 | 0.9275 | 66 |
| E6013 | 0.8750 | 0.9333 | 0.9032 | 45 |
| E7018 | 0.9333 | 0.6667 | 0.7778 | 63 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9648
- **Macro F1-Score:** 0.9620

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 69 | 1 |
| **DC** | 6 | 123 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9200 | 0.9857 | 0.9517 | 70 |
| DC | 0.9919 | 0.9535 | 0.9723 | 129 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
