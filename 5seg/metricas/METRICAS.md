# Métricas de Clasificación SMAW - 5seg

**Fecha de evaluación:** 2026-02-05 19:17:24

**Configuración:**
- Duración de segmento: 5.0s
- Número de muestras (blind): 951
- Número de modelos (ensemble): 5
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.5331 | 0.5421 |
| Electrode Type | 0.4995 | 0.5161 |
| Current Type | 0.6614 | 0.6613 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.5331
- **Macro F1-Score:** 0.5421

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 228 | 10 | 37 |
| **Placa_3mm** | 115 | 125 | 7 |
| **Placa_6mm** | 245 | 30 | 154 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.3878 | 0.8291 | 0.5284 | 275 |
| Placa_3mm | 0.7576 | 0.5061 | 0.6068 | 247 |
| Placa_6mm | 0.7778 | 0.3590 | 0.4912 | 429 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.4995
- **Macro F1-Score:** 0.5161

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 110 | 3 | 1 | 4 |
| **E6011** | 145 | 153 | 9 | 3 |
| **E6013** | 110 | 10 | 112 | 9 |
| **E7018** | 149 | 23 | 10 | 100 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.2140 | 0.9322 | 0.3481 | 118 |
| E6011 | 0.8095 | 0.4935 | 0.6132 | 310 |
| E6013 | 0.8485 | 0.4647 | 0.6005 | 241 |
| E7018 | 0.8621 | 0.3546 | 0.5025 | 282 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.6614
- **Macro F1-Score:** 0.6613

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 321 | 9 |
| **DC** | 313 | 308 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.5063 | 0.9727 | 0.6660 | 330 |
| DC | 0.9716 | 0.4960 | 0.6567 | 621 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
