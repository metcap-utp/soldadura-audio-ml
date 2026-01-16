# Resultados del Modelo de Clasificación SMAW

## I. Validación Cruzada (5-Fold Cross-Validation)

### Audio de 5 segundos

**Fecha de ejecución:** 2026-01-12 22:21:53

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 92.09%   | 92.11%   | 92.14%    | 92.09% |
| **Electrodo** | 92.87%   | 92.87%   | 92.93%    | 92.87% |
| **Corriente** | 98.38%   | 98.39%   | 98.41%    | 98.38% |

**Mejora vs modelo individual:** +16.86% (Placa), +13.13% (Electrodo), +3.44% (Corriente)

---

### Audio de 10 segundos

**Fecha de ejecución:** 2026-01-12 19:30:36

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 98.99%   | 98.99%   | 98.99%    | 98.99% |
| **Electrodo** | 98.12%   | 98.12%   | 98.12%    | 98.12% |
| **Corriente** | 99.78%   | 99.78%   | 99.78%    | 99.78% |

**Mejora vs modelo individual:** +19.49% (Placa), +13.58% (Electrodo), +3.55% (Corriente)

---

### Audio de 30 segundos

**Fecha de ejecución:** 2025-12-25 03:48:33

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 99.88%   | 99.88%   | 99.89%    | 99.88% |
| **Electrodo** | 98.85%   | 98.85%   | 98.87%    | 98.85% |
| **Corriente** | 99.77%   | 99.77%   | 99.77%    | 99.77% |

**Mejora vs modelo individual:** +15.68% (Placa), +8.65% (Electrodo), +2.42% (Corriente)

---

## II. Evaluación en Conjunto Holdout (Datos No Vistos)

Los datos de holdout representan condiciones de uso real, nunca vistos durante el entrenamiento.

### Audio de 5 segundos

**Fecha de ejecución:** 2026-01-14 12:27:57  
**Tamaño del conjunto:** 430 muestras

| Parámetro     | Accuracy | F1-Score (macro avg) | Precision (macro avg) | Recall (macro avg) |
| ------------- | -------- | -------------------- | --------------------- | ------------------ |
| **Placa**     | 71.86%   | 72.00%               | 73.00%                | 72.00%             |
| **Electrodo** | 81.40%   | 80.00%               | 79.00%                | 81.00%             |
| **Corriente** | 95.58%   | 95.00%               | 95.00%                | 96.00%             |

---

### Audio de 10 segundos

**Fecha de ejecución:** 2026-01-14 12:47:57 (aproximada)  
**Tamaño del conjunto:** 224 muestras

| Parámetro     | Accuracy | F1-Score (macro avg) | Precision (macro avg) | Recall (macro avg) |
| ------------- | -------- | -------------------- | --------------------- | ------------------ |
| **Placa**     | 73.21%   | 74.00%               | 75.00%                | 73.00%             |
| **Electrodo** | 87.50%   | 86.00%               | 86.00%                | 87.00%             |
| **Corriente** | 97.77%   | 98.00%               | 98.00%                | 97.00%             |

---

### Audio de 30 segundos

**Fecha de ejecución:** 2026-01-14 12:52:33 (aproximada)  
**Tamaño del conjunto:** 87 muestras

| Parámetro     | Accuracy | F1-Score (macro avg) | Precision (macro avg) | Recall (macro avg) |
| ------------- | -------- | -------------------- | --------------------- | ------------------ |
| **Placa**     | 100.00%  | 100.00%              | 100.00%               | 100.00%            |
| **Electrodo** | 97.70%   | 97.00%               | 97.00%                | 98.00%             |
| **Corriente** | 100.00%  | 100.00%              | 100.00%               | 100.00%            |

---

## III. Comparación General

| Longitud               | Validación Cruzada | Holdout Test | Diferencia |
| ---------------------- | ------------------ | ------------ | ---------- |
| **5 seg - Placa**      | 92.09%             | 71.86%       | -20.23%    |
| **5 seg - Electrodo**  | 92.87%             | 81.40%       | -11.47%    |
| **5 seg - Corriente**  | 98.38%             | 95.58%       | -2.80%     |
| **10 seg - Placa**     | 98.99%             | 73.21%       | -25.78%    |
| **10 seg - Electrodo** | 98.12%             | 87.50%       | -10.62%    |
| **10 seg - Corriente** | 99.78%             | 97.77%       | -2.01%     |
| **30 seg - Placa**     | 99.88%             | 100.00%      | +0.12%     |
| **30 seg - Electrodo** | 98.85%             | 97.70%       | -1.15%     |
| **30 seg - Corriente** | 99.77%             | 100.00%      | +0.23%     |

---

## IV. Metodología

### Ensemble de Modelos

El sistema utiliza un **ensemble de 5 modelos** entrenados mediante validación cruzada K-Fold. Cada modelo se entrena con una partición diferente de los datos, lo que permite:

- Aprovechar toda la información disponible para entrenamiento
- Reducir la varianza y mejorar la robustez
- Obtener predicciones más confiables mediante votación

### Soft Voting

Las predicciones finales se obtienen mediante **soft voting**, que:

1. Cada modelo genera probabilidades para cada clase (no solo la predicción final)
2. Se promedian las probabilidades de todos los modelos
3. Se selecciona la clase con mayor probabilidad promedio

**Ventaja:** Aprovecha la confianza de cada modelo en sus predicciones, no solo su elección, resultando en decisiones más informadas y precisas que el hard voting (voto por mayoría simple).

---

## V. Fórmulas de Evaluación

Las métricas se calculan utilizando **scikit-learn**, biblioteca estándar validada en investigación científica.

### Métricas por Clase

Para cada clase individual:

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Donde:

- **TP** (True Positives): Predicciones correctas de la clase
- **FP** (False Positives): Predicciones incorrectas como esa clase
- **FN** (False Negatives): Casos de la clase no detectados
- **TN** (True Negatives): Correctamente no clasificados como esa clase

---

_Configuración: 5-fold cross-validation, soft voting, 100 epochs, batch size 32_
