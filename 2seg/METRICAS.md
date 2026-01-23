# Métricas de Clasificación SMAW - 2seg

**Fecha de evaluación:** 2026-01-23 11:28:53

**Configuración:**
- Duración de segmento: 2.0s
- Número de muestras (holdout): 2465
- Número de modelos (ensemble): 5
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 69.53% | 0.7009 |
| Electrode Type | 76.96% | 0.7564 |
| Current Type | 88.15% | 0.8761 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 69.53%
- **Macro F1-Score:** 0.7009

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 541 | 41 | 133 |
| **Placa_3mm** | 40 | 560 | 47 |
| **Placa_6mm** | 293 | 197 | 613 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6190 | 0.7566 | 0.6809 | 715 |
| Placa_3mm | 0.7018 | 0.8655 | 0.7751 | 647 |
| Placa_6mm | 0.7730 | 0.5558 | 0.6466 | 1103 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 76.96%
- **Macro F1-Score:** 0.7564

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 254 | 21 | 26 | 6 |
| **E6011** | 52 | 695 | 41 | 14 |
| **E6013** | 20 | 46 | 514 | 53 |
| **E7018** | 92 | 99 | 98 | 434 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6077 | 0.8274 | 0.7007 | 307 |
| E6011 | 0.8072 | 0.8666 | 0.8358 | 802 |
| E6013 | 0.7570 | 0.8120 | 0.7835 | 633 |
| E7018 | 0.8560 | 0.6003 | 0.7057 | 723 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 88.15%
- **Macro F1-Score:** 0.8761

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 828 | 27 |
| **DC** | 265 | 1345 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.7575 | 0.9684 | 0.8501 | 855 |
| DC | 0.9803 | 0.8354 | 0.9021 | 1610 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **holdout** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
