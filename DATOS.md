# Estadísticas del Dataset SMAW

**Generado:** 2026-01-29 23:40:02

Este documento contiene las estadísticas de sesiones y segmentos por etiqueta para todas las duraciones de segmento disponibles.

---

## Resumen General

| Duración | Sesiones | Segmentos | Train | Test | Blind |
|----------|----------|-----------|-------|------|--------|
| 1seg | - | - | - | - | - |
| 2seg | - | - | - | - | - |
| 5seg | 373 | 8185 | 5713 | 1521 | 951 |
| 10seg | - | - | - | - | - |
| 20seg | - | - | - | - | - |
| 30seg | - | - | - | - | - |
| 50seg | - | - | - | - | - |

---

## Sesiones por Etiqueta (Globales)

Las sesiones son las mismas para todas las duraciones, solo cambia el número de segmentos.

### Espesor de Placa (Plate Thickness)

| Etiqueta | Sesiones |
|----------|----------|
| Placa_12mm | 120 |
| Placa_3mm | 127 |
| Placa_6mm | 126 |

### Tipo de Electrodo (Electrode)

| Etiqueta | Sesiones |
|----------|----------|
| E6010 | 57 |
| E6011 | 132 |
| E6013 | 120 |
| E7018 | 64 |

### Tipo de Corriente (Type of Current)

| Etiqueta | Sesiones |
|----------|----------|
| AC | 123 |
| DC | 250 |

---

## Segmentos por Duración y Etiqueta

### 1seg

*No hay datos disponibles. Ejecutar `python 1seg/generar_splits.py`*

### 2seg

*No hay datos disponibles. Ejecutar `python 2seg/generar_splits.py`*

### 5seg

**Total:** 8185 segmentos

#### Por Espesor de Placa

| Etiqueta | Total | Train | Test | Blind |
|----------|-------|-------|------|--------|
| Placa_12mm | 2709 | 1894 | 540 | 275 |
| Placa_3mm | 2747 | 1993 | 507 | 247 |
| Placa_6mm | 2729 | 1826 | 474 | 429 |

#### Por Tipo de Electrodo

| Etiqueta | Total | Train | Test | Blind |
|----------|-------|-------|------|--------|
| E6010 | 1233 | 876 | 239 | 118 |
| E6011 | 3193 | 2253 | 630 | 310 |
| E6013 | 2396 | 1724 | 431 | 241 |
| E7018 | 1363 | 860 | 221 | 282 |

#### Por Tipo de Corriente

| Etiqueta | Total | Train | Test | Blind |
|----------|-------|-------|------|--------|
| AC | 3278 | 2317 | 631 | 330 |
| DC | 4907 | 3396 | 890 | 621 |

### 10seg

*No hay datos disponibles. Ejecutar `python 10seg/generar_splits.py`*

### 20seg

*No hay datos disponibles. Ejecutar `python 20seg/generar_splits.py`*

### 30seg

*No hay datos disponibles. Ejecutar `python 30seg/generar_splits.py`*

### 50seg

*No hay datos disponibles. Ejecutar `python 50seg/generar_splits.py`*

---

## Notas

- Las **sesiones** representan grabaciones únicas de soldadura
- Los **segmentos** se generan on-the-fly dividiendo cada grabación según la duración especificada
- El split estratificado garantiza proporciones similares de etiquetas en cada conjunto
- **Blind** es el conjunto de validación final (nunca usado durante desarrollo)
- Los datos de cada duración se generan ejecutando `python Xseg/generar_splits.py`
