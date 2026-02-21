# Métricas de Clasificación SMAW - 2seg

**Fecha de evaluación:** 2026-02-19 18:35:43

**Configuración:**
- Duración de segmento: 2.0s
- Número de muestras (blind): 2465
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7071 | 0.7126 |
| Electrode Type | 0.7797 | 0.7638 |
| Current Type | 0.8787 | 0.8732 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7071
- **Macro F1-Score:** 0.7126

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 534 | 41 | 140 |
| **Placa_3mm** | 37 | 567 | 43 |
| **Placa_6mm** | 281 | 180 | 642 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6268 | 0.7469 | 0.6816 | 715 |
| Placa_3mm | 0.7195 | 0.8764 | 0.7902 | 647 |
| Placa_6mm | 0.7782 | 0.5820 | 0.6660 | 1103 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.7797
- **Macro F1-Score:** 0.7638

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 251 | 25 | 24 | 7 |
| **E6011** | 38 | 709 | 40 | 15 |
| **E6013** | 21 | 45 | 531 | 36 |
| **E7018** | 105 | 73 | 114 | 431 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6048 | 0.8176 | 0.6953 | 307 |
| E6011 | 0.8322 | 0.8840 | 0.8573 | 802 |
| E6013 | 0.7489 | 0.8389 | 0.7914 | 633 |
| E7018 | 0.8814 | 0.5961 | 0.7112 | 723 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.8787
- **Macro F1-Score:** 0.8732

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 827 | 28 |
| **DC** | 271 | 1339 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.7532 | 0.9673 | 0.8469 | 855 |
| DC | 0.9795 | 0.8317 | 0.8996 | 1610 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
