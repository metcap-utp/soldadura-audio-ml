# Métricas de Clasificación SMAW - 30seg

**Fecha de evaluación:** 2026-02-07 01:16:53

**Configuración:**
- Duración de segmento: 30.0s
- Número de muestras (blind): 208
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7019 | 0.7069 |
| Electrode Type | 0.8750 | 0.8655 |
| Current Type | 0.9615 | 0.9578 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7019
- **Macro F1-Score:** 0.7069

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 49 | 1 | 5 |
| **Placa_3mm** | 3 | 48 | 0 |
| **Placa_6mm** | 25 | 28 | 49 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6364 | 0.8909 | 0.7424 | 55 |
| Placa_3mm | 0.6234 | 0.9412 | 0.7500 | 51 |
| Placa_6mm | 0.9074 | 0.4804 | 0.6282 | 102 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8750
- **Macro F1-Score:** 0.8655

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 25 | 0 | 0 | 0 |
| **E6011** | 0 | 66 | 0 | 0 |
| **E6013** | 0 | 0 | 41 | 3 |
| **E7018** | 13 | 5 | 5 | 50 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6579 | 1.0000 | 0.7937 | 25 |
| E6011 | 0.9296 | 1.0000 | 0.9635 | 66 |
| E6013 | 0.8913 | 0.9318 | 0.9111 | 44 |
| E7018 | 0.9434 | 0.6849 | 0.7937 | 73 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9615
- **Macro F1-Score:** 0.9578

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 69 | 3 |
| **DC** | 5 | 131 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9324 | 0.9583 | 0.9452 | 72 |
| DC | 0.9776 | 0.9632 | 0.9704 | 136 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
