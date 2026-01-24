# Resultados del Modelo de Clasificación SMAW

## I. Comparación: Audios Preprocesados vs Audios Crudos

Se realizaron dos series de experimentos para comparar el rendimiento del modelo con diferentes tipos de audio:

1. **Audios Preprocesados:** Audio con pipeline de preprocesamiento (reducción de ruido, normalización, filtrado)
2. **Audios Crudos:** Audio sin preprocesamiento, directamente desde la captura

### Resultados en Conjunto Holdout (Generalización Real)

El conjunto holdout contiene sesiones de soldadura nunca vistas durante el entrenamiento, representando condiciones de uso real.

#### Comparación por Duración de Segmento

| Duración   | Parámetro | Preprocesados (Acc) | Crudos (Acc) | Diferencia | Mejor         |
| ---------- | --------- | ------------------- | ------------ | ---------- | ------------- |
| **5 seg**  | Placa     | 72.79%              | 74.66%       | **+1.87%** | ✅ Crudos     |
| **5 seg**  | Electrodo | 82.33%              | 84.96%       | **+2.63%** | ✅ Crudos     |
| **5 seg**  | Corriente | 96.74%              | 93.27%       | -3.47%     | Preprocesados |
| **10 seg** | Placa     | 74.55%              | 75.39%       | **+0.84%** | ✅ Crudos     |
| **10 seg** | Electrodo | 87.05%              | 86.13%       | -0.92%     | Preprocesados |
| **10 seg** | Corriente | 98.21%              | 97.09%       | -1.12%     | Preprocesados |
| **30 seg** | Placa     | 80.46%              | 69.03%       | -11.43%    | Preprocesados |
| **30 seg** | Electrodo | 89.66%              | 88.50%       | -1.16%     | Preprocesados |
| **30 seg** | Corriente | 97.70%              | 95.58%       | -2.12%     | Preprocesados |

#### Comparación por F1-Score (Macro)

| Duración   | Parámetro | Preprocesados (F1) | Crudos (F1) | Diferencia | Mejor         |
| ---------- | --------- | ------------------ | ----------- | ---------- | ------------- |
| **5 seg**  | Placa     | 73.00%             | 75.23%      | **+2.23%** | ✅ Crudos     |
| **5 seg**  | Electrodo | 82.00%             | 83.70%      | **+1.70%** | ✅ Crudos     |
| **5 seg**  | Corriente | 97.00%             | 92.84%      | -4.16%     | Preprocesados |
| **10 seg** | Placa     | 75.00%             | 76.01%      | **+1.01%** | ✅ Crudos     |
| **10 seg** | Electrodo | 87.00%             | 85.25%      | -1.75%     | Preprocesados |
| **10 seg** | Corriente | 98.00%             | 96.87%      | -1.13%     | Preprocesados |
| **30 seg** | Placa     | 81.00%             | 69.46%      | -11.54%    | Preprocesados |
| **30 seg** | Electrodo | 90.00%             | 87.12%      | -2.88%     | Preprocesados |
| **30 seg** | Corriente | 98.00%             | 95.24%      | -2.76%     | Preprocesados |

### Análisis de Resultados

**Observaciones clave:**

1. **Segmentos cortos (5 seg):** Los audios crudos muestran ligera mejora en clasificación de Placa (+1.87%) y Electrodo (+2.63%), pero pierden rendimiento en Corriente (-3.47%).

2. **Segmentos medios (10 seg):** Resultados mixtos con diferencias mínimas (<1.5%), sugiriendo que ambos enfoques son comparables.

3. **Segmentos largos (30 seg):** Los audios preprocesados mantienen clara ventaja, especialmente en Placa (-11.43% para crudos).

4. **Tipo de Corriente:** El preprocesamiento beneficia consistentemente la clasificación AC/DC en todas las duraciones.

**Hipótesis:**

- El preprocesamiento elimina ruido ambiental que puede confundir características de tipo de corriente
- Para segmentos cortos, el preprocesamiento podría eliminar información acústica útil para distinguir grosor de placa
- Los segmentos largos se benefician más del preprocesamiento al tener más oportunidad de ruido acumulado

---

## II. Resultados Detallados con Audios Crudos (Configuración Actual)

### Validación Cruzada (5-Fold)

#### Audio de 1 segundo

**Fecha de ejecución:** 2026-01-22  
**Total de segmentos:** 38,182 | **Sesiones únicas:** 335

| Parámetro     | Accuracy Promedio Fold | Accuracy Ensemble | F1-Score Ensemble |
| ------------- | ---------------------- | ----------------- | ----------------- |
| **Placa**     | 0.00%                  | 33.58%            | 16.89%            |
| **Electrodo** | 0.00%                  | 15.43%            | 4.13%             |
| **Corriente** | 0.00%                  | 40.32%            | 23.17%            |

> **Nota:** El modelo con segmentos de 1 segundo presenta rendimiento muy bajo. VGGish produce solo 1 frame de embedding para 1s de audio, lo cual no provee suficiente contexto temporal para la clasificación multi-tarea. El modelo predice una sola clase (la mayoritaria) para todas las muestras.

#### Audio de 2 segundos

**Fecha de ejecución:** 2026-01-22  
**Total de segmentos:** 18,848 | **Sesiones únicas:** 335

| Parámetro     | Accuracy Promedio Fold | Accuracy Ensemble | F1-Score Ensemble |
| ------------- | ---------------------- | ----------------- | ----------------- |
| **Placa**     | 78.15%                 | 91.40%            | 91.38%            |
| **Electrodo** | 83.82%                 | 94.42%            | 94.44%            |
| **Corriente** | 96.04%                 | 99.28%            | 99.28%            |

#### Audio de 5 segundos

**Fecha de ejecución:** 2026-01-21  
**Total de segmentos:** 7,234 | **Sesiones únicas:** 335

| Parámetro     | Accuracy Promedio Fold | F1-Score Ensemble | Mejora Ensemble |
| ------------- | ---------------------- | ----------------- | --------------- |
| **Placa**     | 84.64%                 | 100.00%           | +15.36%         |
| **Electrodo** | 93.12%                 | 99.96%            | +6.84%          |
| **Corriente** | 99.03%                 | 100.00%           | +0.97%          |

#### Audio de 10 segundos

**Fecha de ejecución:** 2026-01-21  
**Total de segmentos:** 3,372 | **Sesiones únicas:** 335

| Parámetro     | Accuracy Promedio Fold | F1-Score Ensemble | Mejora Ensemble |
| ------------- | ---------------------- | ----------------- | --------------- |
| **Placa**     | 88.45%                 | 100.00%           | +11.55%         |
| **Electrodo** | 95.07%                 | 99.97%            | +4.90%          |
| **Corriente** | 98.88%                 | 100.00%           | +1.12%          |

#### Audio de 30 segundos

**Fecha de ejecución:** 2026-01-21  
**Total de segmentos:** 805 | **Sesiones únicas:** 335

| Parámetro     | Accuracy Promedio Fold | F1-Score Ensemble | Mejora Ensemble |
| ------------- | ---------------------- | ----------------- | --------------- |
| **Placa**     | 93.43%                 | 100.00%           | +6.57%          |
| **Electrodo** | 96.41%                 | 100.00%           | +3.59%          |
| **Corriente** | 99.03%                 | 100.00%           | +0.97%          |

### Evaluación en Conjunto Holdout

#### Audio de 1 segundo

**Tamaño del conjunto:** 4,988 segmentos (87 sesiones)

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 29.05%   | 15.01%   | 10.00%    | 33.00% |
| **Electrodo** | 12.49%   | 5.55%    | 3.00%     | 25.00% |
| **Corriente** | 34.64%   | 25.73%   | 17.00%    | 50.00% |

> **Nota:** Resultados muy pobres debido a contexto temporal insuficiente (1 frame VGGish). El modelo predice consistentemente la clase mayoritaria.

#### Audio de 2 segundos

**Tamaño del conjunto:** 2,465 segmentos (87 sesiones)

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 69.53%   | 70.09%   | 70.00%    | 73.00% |
| **Electrodo** | 76.96%   | 75.64%   | 76.00%    | 78.00% |
| **Corriente** | 88.15%   | 87.61%   | 87.00%    | 90.00% |

#### Audio de 5 segundos

**Tamaño del conjunto:** 958 segmentos (87 sesiones)

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 74.66%   | 75.23%   | 77.00%    | 75.00% |
| **Electrodo** | 84.96%   | 83.70%   | 86.00%    | 82.00% |
| **Corriente** | 93.27%   | 92.84%   | 93.00%    | 93.00% |

#### Audio de 10 segundos

**Tamaño del conjunto:** 447 segmentos (87 sesiones)

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 75.39%   | 76.01%   | 77.00%    | 79.00% |
| **Electrodo** | 86.13%   | 85.25%   | 87.00%    | 86.00% |
| **Corriente** | 97.09%   | 96.87%   | 97.00%    | 97.00% |

#### Audio de 30 segundos

**Tamaño del conjunto:** 113 segmentos (87 sesiones)

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 69.03%   | 69.46%   | 74.00%    | 69.00% |
| **Electrodo** | 88.50%   | 87.12%   | 91.00%    | 88.00% |
| **Corriente** | 95.58%   | 95.24%   | 96.00%    | 96.00% |

---

## III. Comparación General (Audios Crudos)

| Longitud               | Validación Cruzada | Holdout Test | Diferencia |
| ---------------------- | ------------------ | ------------ | ---------- |
| **1 seg - Placa**      | 33.58%             | 29.05%       | -4.53%     |
| **1 seg - Electrodo**  | 15.43%             | 12.49%       | -2.94%     |
| **1 seg - Corriente**  | 40.32%             | 34.64%       | -5.68%     |
| **2 seg - Placa**      | 91.40%             | 69.53%       | -21.87%    |
| **2 seg - Electrodo**  | 94.42%             | 76.96%       | -17.46%    |
| **2 seg - Corriente**  | 99.28%             | 88.15%       | -11.13%    |
| **5 seg - Placa**      | 100.00%            | 74.66%       | -25.34%    |
| **5 seg - Electrodo**  | 99.96%             | 84.96%       | -15.00%    |
| **5 seg - Corriente**  | 100.00%            | 93.27%       | -6.73%     |
| **10 seg - Placa**     | 100.00%            | 75.39%       | -24.61%    |
| **10 seg - Electrodo** | 99.97%             | 86.13%       | -13.84%    |
| **10 seg - Corriente** | 100.00%            | 97.09%       | -2.91%     |
| **30 seg - Placa**     | 100.00%            | 69.03%       | -30.97%    |
| **30 seg - Electrodo** | 100.00%            | 88.50%       | -11.50%    |
| **30 seg - Corriente** | 100.00%            | 95.58%       | -4.42%     |

**Observaciones:**

- **1 segundo:** Rendimiento inaceptable - VGGish produce solo 1 frame por segundo, insuficiente para clasificación multi-tarea
- **2 segundos:** Buen rendimiento considerando el contexto limitado; mejor relación cantidad/calidad de datos
- **5-10 segundos:** Balance óptimo entre contexto temporal y cantidad de muestras
- **30 segundos:** Menor cantidad de muestras pero mejor generalización en electrodo

**Nota:** La diferencia entre validación cruzada (ensemble perfecto en datos de entrenamiento) y holdout refleja la capacidad de generalización real del modelo.

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
