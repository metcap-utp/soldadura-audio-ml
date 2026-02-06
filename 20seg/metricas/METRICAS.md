# Métricas de Clasificación SMAW - 20seg

**Fecha de evaluación:** 2026-02-05 13:25:28

**Configuración:**
- Duración de segmento: 20.0s
- Número de muestras (blind): 199
- Número de modelos (ensemble): 5
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 73.37% | 0.7376 |
| Electrode Type | 87.94% | 0.8745 |
| Current Type | 96.48% | 0.9620 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 73.37%
- **Macro F1-Score:** 0.7376

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 48 | 2 | 6 |
| **Placa_3mm** | 4 | 44 | 1 |
| **Placa_6mm** | 21 | 19 | 54 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6575 | 0.8571 | 0.7442 | 56 |
| Placa_3mm | 0.6769 | 0.8980 | 0.7719 | 49 |
| Placa_6mm | 0.8852 | 0.5745 | 0.6968 | 94 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 87.94%
- **Macro F1-Score:** 0.8745

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 25 | 0 | 0 | 0 |
| **E6011** | 0 | 65 | 1 | 0 |
| **E6013** | 0 | 0 | 41 | 4 |
| **E7018** | 8 | 7 | 4 | 44 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.7576 | 1.0000 | 0.8621 | 25 |
| E6011 | 0.9028 | 0.9848 | 0.9420 | 66 |
| E6013 | 0.8913 | 0.9111 | 0.9011 | 45 |
| E7018 | 0.9167 | 0.6984 | 0.7928 | 63 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 96.48%
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
