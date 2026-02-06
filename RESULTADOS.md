# Resultados del Modelo de Clasificacion SMAW

## I. Comparacion: Audios Preprocesados vs Audios Crudos

Se realizaron dos series de experimentos para comparar el rendimiento del modelo con diferentes tipos de audio:

1. **Audios Preprocesados:** Audio procesado con Spleeter para separacion de fuentes y aumento de ganancia (gain)
2. **Audios Crudos:** Audio sin preprocesamiento, directamente desde la captura

### Resultados en Conjunto Blind (Generalizacion Real)

El conjunto blind contiene sesiones de soldadura nunca vistas durante el entrenamiento, representando condiciones de uso real.

#### Comparacion por Duracion de Segmento (Accuracy)

| Duracion   | Parametro | Preprocesados | Crudos | Diferencia |
| ---------- | --------- | ------------- | ------ | ---------- |
| **5 seg**  | Placa     | 0.7279        | 0.7413 | +0.0134    |
| **5 seg**  | Electrodo | 0.8233        | 0.8570 | +0.0337    |
| **5 seg**  | Corriente | 0.9674        | 0.9369 | -0.0305    |
| **10 seg** | Placa     | 0.7455        | 0.7539 | +0.0084    |
| **10 seg** | Electrodo | 0.8705        | 0.8613 | -0.0092    |
| **10 seg** | Corriente | 0.9821        | 0.9709 | -0.0112    |
| **20 seg** | Placa     | -             | 0.7337 | -          |
| **20 seg** | Electrodo | -             | 0.8794 | -          |
| **20 seg** | Corriente | -             | 0.9648 | -          |
| **30 seg** | Placa     | 0.8046        | 0.6903 | -0.1143    |
| **30 seg** | Electrodo | 0.8966        | 0.8850 | -0.0116    |
| **30 seg** | Corriente | 0.9770        | 0.9558 | -0.0212    |
| **50 seg** | Placa     | -             | 0.6610 | -          |
| **50 seg** | Electrodo | -             | 0.8475 | -          |
| **50 seg** | Corriente | -             | 0.9661 | -          |

#### Comparacion por F1-Score (Macro)

| Duracion   | Parametro | Preprocesados | Crudos | Diferencia |
| ---------- | --------- | ------------- | ------ | ---------- |
| **5 seg**  | Placa     | 0.7300        | 0.7469 | +0.0169    |
| **5 seg**  | Electrodo | 0.8200        | 0.8446 | +0.0246    |
| **5 seg**  | Corriente | 0.9700        | 0.9327 | -0.0373    |
| **10 seg** | Placa     | 0.7500        | 0.7601 | +0.0101    |
| **10 seg** | Electrodo | 0.8700        | 0.8525 | -0.0175    |
| **10 seg** | Corriente | 0.9800        | 0.9687 | -0.0113    |
| **20 seg** | Placa     | -             | 0.7376 | -          |
| **20 seg** | Electrodo | -             | 0.8745 | -          |
| **20 seg** | Corriente | -             | 0.9620 | -          |
| **30 seg** | Placa     | 0.8100        | 0.6946 | -0.1154    |
| **30 seg** | Electrodo | 0.9000        | 0.8712 | -0.0288    |
| **30 seg** | Corriente | 0.9800        | 0.9524 | -0.0276    |
| **50 seg** | Placa     | -             | 0.6668 | -          |
| **50 seg** | Electrodo | -             | 0.8585 | -          |
| **50 seg** | Corriente | -             | 0.9612 | -          |



## II. Evaluacion durante Entrenamiento (Validacion Cruzada K-Fold)

En las metricas que mencionan "Fold", el valor corresponde al promedio de accuracy obtenido en cada particion (fold) durante la validacion cruzada K-Fold. Este promedio refleja el rendimiento del modelo individual en cada fold antes de combinarlos en el ensamble.

Los resultados de validacion cruzada representan el rendimiento del modelo evaluado en los datos de entrenamiento mediante K-Fold. Cada fold se entrena con K-1 particiones y se evalua en la particion restante, rotando hasta cubrir todos los datos.

### Audio de 1 segundo

**Fecha de ejecucion:** 2026-01-22  
**Total de segmentos:** 38,182 | **Sesiones unicas:** 335

| Parametro     | Accuracy Promedio Fold | Accuracy Ensemble | F1 Macro Ensemble |
| ------------- | ---------------------- | ----------------- | ----------------- |
| **Placa**     | 0.0000                 | 0.3358            | 0.1689            |
| **Electrodo** | 0.0000                 | 0.1543            | 0.0413            |
| **Corriente** | 0.0000                 | 0.4032            | 0.2317            |

El modelo con segmentos de 1 segundo presenta rendimiento muy bajo. VGGish produce solo 1 frame de embedding para 1s de audio, lo cual no provee suficiente contexto temporal para la clasificacion multi-tarea.

### Audio de 2 segundos

**Fecha de ejecucion:** 2026-01-22  
**Total de segmentos:** 18,848 | **Sesiones unicas:** 335

| Parametro     | Accuracy Promedio Fold | Accuracy Ensemble | F1 Macro Ensemble |
| ------------- | ---------------------- | ----------------- | ----------------- |
| **Placa**     | 0.7815                 | 0.9140            | 0.9138            |
| **Electrodo** | 0.8382                 | 0.9442            | 0.9444            |
| **Corriente** | 0.9604                 | 0.9928            | 0.9928            |

### Audio de 5 segundos

**Fecha de ejecucion:** 2026-01-21  
**Total de segmentos:** 7,234 | **Sesiones unicas:** 335

| Parametro     | Accuracy Promedio Fold | F1 Macro Ensemble | Mejora Ensemble |
| ------------- | ---------------------- | ----------------- | --------------- |
| **Placa**     | 0.8464                 | 1.0000            | +0.1536         |
| **Electrodo** | 0.9312                 | 0.9996            | +0.0684         |
| **Corriente** | 0.9903                 | 1.0000            | +0.0097         |

### Comparacion de K (5 segundos)

**Fecha de ejecucion:** 2026-02-01  
**Fuente:** [5seg/results.json](5seg/results.json)

| K  | Acc Fold (Placa) | Acc Fold (Electrodo) | Acc Fold (Corriente) | Acc Ensemble (Placa) | Acc Ensemble (Electrodo) | Acc Ensemble (Corriente) |
| -- | ---------------- | -------------------- | -------------------- | -------------------- | ------------------------ | ------------------------ |
| 3  | 0.8484           | 0.9225               | 0.9819               | 0.9961               | 0.9956                   | 0.9999                   |
| 5  | 0.8600           | 0.9180               | 0.9856               | 0.9981               | 0.9989                   | 1.0000                   |
| 7  | 0.8589           | 0.9258               | 0.9864               | 0.9989               | 0.9989                   | 1.0000                   |
| 10 | 0.8674           | 0.9281               | 0.9865               | 0.9997               | 0.9996                   | 1.0000                   |
| 15 | 0.8756           | 0.9277               | 0.9870               | 0.9997               | 0.9993                   | 1.0000                   |
| 20 | 0.8781           | 0.9331               | 0.9857               | 0.9975               | 0.9979                   | 1.0000                   |

### Audio de 10 segundos

**Fecha de ejecucion:** 2026-01-21  
**Total de segmentos:** 3,372 | **Sesiones unicas:** 335

| Parametro     | Accuracy Promedio Fold | F1 Macro Ensemble | Mejora Ensemble |
| ------------- | ---------------------- | ----------------- | --------------- |
| **Placa**     | 0.8845                 | 1.0000            | +0.1155         |
| **Electrodo** | 0.9507                 | 0.9997            | +0.0490         |
| **Corriente** | 0.9888                 | 1.0000            | +0.0112         |

### Comparacion de K (10 segundos)

**Fecha de ejecucion:** 2026-01-31  
**Fuente:** [10seg/results.json](10seg/results.json)

| K  | Acc Fold (Placa) | Acc Fold (Electrodo) | Acc Fold (Corriente) | Acc Ensemble (Placa) | Acc Ensemble (Electrodo) | Acc Ensemble (Corriente) |
| -- | ---------------- | -------------------- | -------------------- | -------------------- | ------------------------ | ------------------------ |
| 3  | 0.8833           | 0.9415               | 0.9911               | 0.9997               | 0.9994                   | 1.0000                   |
| 5  | 0.8845           | 0.9507               | 0.9888               | 1.0000               | 0.9997                   | 1.0000                   |
| 7  | 0.8961           | 0.9482               | 0.9908               | 1.0000               | 1.0000                   | 1.0000                   |
| 10 | 0.9045           | 0.9522               | 0.9907               | 1.0000               | 1.0000                   | 1.0000                   |
| 15 | 0.9136           | 0.9561               | 0.9924               | 1.0000               | 1.0000                   | 1.0000                   |
| 20 | 0.9063           | 0.9582               | 0.9939               | 1.0000               | 1.0000                   | 1.0000                   |

### Audio de 20 segundos

**Fecha de ejecucion:** 2026-01-30  
**Total de segmentos:** 1,441 | **Sesiones unicas:** 335

| Parametro     | Accuracy Promedio Fold | Accuracy Ensemble | F1 Macro Ensemble |
| ------------- | ---------------------- | ----------------- | ----------------- |
| **Placa**     | 0.9140                 | 1.0000            | 1.0000            |
| **Electrodo** | 0.9545                 | 1.0000            | 1.0000            |
| **Corriente** | 0.9895                 | 1.0000            | 1.0000            |

### Audio de 30 segundos

**Fecha de ejecucion:** 2026-01-21  
**Total de segmentos:** 805 | **Sesiones unicas:** 335

| Parametro     | Accuracy Promedio Fold | F1 Macro Ensemble | Mejora Ensemble |
| ------------- | ---------------------- | ----------------- | --------------- |
| **Placa**     | 0.9343                 | 1.0000            | +0.0657         |
| **Electrodo** | 0.9641                 | 1.0000            | +0.0359         |
| **Corriente** | 0.9903                 | 1.0000            | +0.0097         |

### Audio de 50 segundos

**Fecha de ejecucion:** 2026-01-31  
**Total de segmentos:** 389 | **Sesiones unicas:** 335

| Parametro     | Accuracy Promedio Fold | Accuracy Ensemble | F1 Macro Ensemble |
| ------------- | ---------------------- | ----------------- | ----------------- |
| **Placa**     | 0.9453                 | 1.0000            | 1.0000            |
| **Electrodo** | 0.9620                 | 1.0000            | 1.0000            |
| **Corriente** | 0.9901                 | 1.0000            | 1.0000            |

## III. Evaluacion en Conjunto Blind

El conjunto blind contiene sesiones de soldadura que nunca fueron vistas durante el entrenamiento, lo que permite medir la capacidad de generalizacion real del modelo ante datos nuevos.

### Audio de 1 segundo

**Tamano del conjunto:** 4,988 segmentos

| Parametro     | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
| ------------- | -------- | ---------- | ----------------- | -------------- |
| **Placa**     | 0.2905   | 0.1501     | 0.0968            | 0.3333         |
| **Electrodo** | 0.1249   | 0.0555     | 0.0312            | 0.2500         |
| **Corriente** | 0.3464   | 0.2573     | 0.1732            | 0.5000         |

**Matriz de confusion (1 segundo):**

![Matriz de confusion combinada - 1 segundo](1seg/confusion_matrices/combined_k10_2026-02-01_14-18-17.png)

### Audio de 2 segundos

**Tamano del conjunto:** 2,465 segmentos

| Parametro     | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
| ------------- | -------- | ---------- | ----------------- | -------------- |
| **Placa**     | 0.6953   | 0.7009     | 0.6979            | 0.7260         |
| **Electrodo** | 0.7696   | 0.7564     | 0.7570            | 0.7766         |
| **Corriente** | 0.8815   | 0.8761     | 0.8689            | 0.9019         |

**Matriz de confusion (2 segundos):**

![Matriz de confusion combinada - 2 segundos](2seg/confusion_matrices/combined_k5_2026-01-23_22-42-50.png)

### Audio de 5 segundos

**Tamano del conjunto:** 951 segmentos

| Parametro     | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
| ------------- | -------- | ---------- | ----------------- | -------------- |
| **Placa**     | 0.7413   | 0.7469     | 0.7406            | 0.7707         |
| **Electrodo** | 0.8570   | 0.8446     | 0.8424            | 0.8667         |
| **Corriente** | 0.9369   | 0.9327     | 0.9232            | 0.9489         |

**Matriz de confusion (5 segundos):**

![Matriz de confusion combinada - 5 segundos](5seg/confusion_matrices/combined_k20_2026-02-01_01-25-44.png)

### Audio de 10 segundos

**Tamano del conjunto:** 309 segmentos

| Parametro     | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
| ------------- | -------- | ---------- | ----------------- | -------------- |
| **Placa**     | 0.7539   | 0.7601     | 0.7544            | 0.7907         |
| **Electrodo** | 0.8613   | 0.8525     | 0.8501            | 0.8792         |
| **Corriente** | 0.9709   | 0.9687     | 0.9620            | 0.9775         |

**Matriz de confusion (10 segundos):**

![Matriz de confusion combinada - 10 segundos](10seg/confusion_matrices/combined_k20_2026-02-01_01-31-06.png)

### Audio de 20 segundos

**Tamano del conjunto:** 199 segmentos

| Parametro     | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
| ------------- | -------- | ---------- | ----------------- | -------------- |
| **Placa**     | 0.7337   | 0.7376     | 0.7399            | 0.7765         |
| **Electrodo** | 0.8794   | 0.8745     | 0.8703            | 0.8965         |
| **Corriente** | 0.9648   | 0.9620     | 0.9534            | 0.9747         |

**Matriz de confusion (20 segundos):**

![Matriz de confusion combinada - 20 segundos](20seg/confusion_matrices/combined_k5_2026-02-05_13-25-28.png)

### Audio de 30 segundos

**Tamano del conjunto:** 113 segmentos (87 sesiones)

| Parametro     | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
| ------------- | -------- | ---------- | ----------------- | -------------- |
| **Placa**     | 0.6903   | 0.6946     | 0.7066            | 0.7482         |
| **Electrodo** | 0.8850   | 0.8712     | 0.8627            | 0.9068         |
| **Corriente** | 0.9558   | 0.9524     | 0.9463            | 0.9601         |

**Matriz de confusion (30 segundos):**

![Matriz de confusion combinada - 30 segundos](30seg/confusion_matrices/combined_k5_2026-01-23_22-40-27.png)

### Audio de 50 segundos

**Tamano del conjunto:** 59 segmentos

| Parametro     | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
| ------------- | -------- | ---------- | ----------------- | -------------- |
| **Placa**     | 0.6610   | 0.6668     | 0.6787            | 0.7220         |
| **Electrodo** | 0.8475   | 0.8585     | 0.8773            | 0.8563         |
| **Corriente** | 0.9661   | 0.9612     | 0.9535            | 0.9720         |

**Matriz de confusion (50 segundos):**

![Matriz de confusion combinada - 50 segundos](50seg/confusion_matrices/combined_k5_2026-02-05_13-27-29.png)

## IV. Comparacion General (Audios Crudos)

### 1 segundo

| Parametro     | Evaluacion Entrenamiento (K-Fold) | Blind Test | Diferencia |
| ------------- | --------------------------------- | ---------- | ---------- |
| **Placa**     | 0.3358                            | 0.2905     | -0.0453    |
| **Electrodo** | 0.1543                            | 0.1249     | -0.0294    |
| **Corriente** | 0.4032                            | 0.3464     | -0.0568    |

### 2 segundos

| Parametro     | Evaluacion Entrenamiento (K-Fold) | Blind Test | Diferencia |
| ------------- | --------------------------------- | ---------- | ---------- |
| **Placa**     | 0.9140                            | 0.6953     | -0.2187    |
| **Electrodo** | 0.9442                            | 0.7696     | -0.1746    |
| **Corriente** | 0.9928                            | 0.8815     | -0.1113    |

### 5 segundos

| Parametro     | Evaluacion Entrenamiento (K-Fold) | Blind Test | Diferencia |
| ------------- | --------------------------------- | ---------- | ---------- |
| **Placa**     | 1.0000                            | 0.7413     | -0.2587    |
| **Electrodo** | 0.9996                            | 0.8570     | -0.1426    |
| **Corriente** | 1.0000                            | 0.9369     | -0.0631    |

### 10 segundos

| Parametro     | Evaluacion Entrenamiento (K-Fold) | Blind Test | Diferencia |
| ------------- | --------------------------------- | ---------- | ---------- |
| **Placa**     | 1.0000                            | 0.7539     | -0.2461    |
| **Electrodo** | 0.9997                            | 0.8613     | -0.1384    |
| **Corriente** | 1.0000                            | 0.9709     | -0.0291    |

### 20 segundos

| Parametro     | Evaluacion Entrenamiento (K-Fold) | Blind Test | Diferencia |
| ------------- | --------------------------------- | ---------- | ---------- |
| **Placa**     | 1.0000                            | 0.7337     | -0.2663    |
| **Electrodo** | 1.0000                            | 0.8794     | -0.1206    |
| **Corriente** | 1.0000                            | 0.9648     | -0.0352    |

### 30 segundos

| Parametro     | Evaluacion Entrenamiento (K-Fold) | Blind Test | Diferencia |
| ------------- | --------------------------------- | ---------- | ---------- |
| **Placa**     | 1.0000                            | 0.6903     | -0.3097    |
| **Electrodo** | 1.0000                            | 0.8850     | -0.1150    |
| **Corriente** | 1.0000                            | 0.9558     | -0.0442    |

### 50 segundos

| Parametro     | Evaluacion Entrenamiento (K-Fold) | Blind Test | Diferencia |
| ------------- | --------------------------------- | ---------- | ---------- |
| **Placa**     | 1.0000                            | 0.6610     | -0.3390    |
| **Electrodo** | 1.0000                            | 0.8475     | -0.1525    |
| **Corriente** | 1.0000                            | 0.9661     | -0.0339    |

La diferencia entre la evaluacion durante entrenamiento (ensamble con datos de K-Fold) y blind refleja la capacidad de generalizacion real del modelo ante datos nunca vistos.

## V. Analisis de K-Folds

### Metricas vs K-Folds en Conjunto Blind (5 segundos)

Se evaluo el impacto del numero de folds (modelos en el ensemble) en el rendimiento sobre el conjunto blind:

| K  | Accuracy (Placa) | Accuracy (Electrodo) | Accuracy (Corriente) | F1 (Placa) | F1 (Electrodo) | F1 (Corriente) |
| -- | ---------------- | -------------------- | -------------------- | ---------- | -------------- | -------------- |
| 3  | 0.7434           | 0.8465               | 0.9285               | 0.7479     | 0.8388         | 0.9242         |
| 5  | 0.7455           | 0.8559               | 0.9295               | 0.7490     | 0.8437         | 0.9254         |
| 7  | 0.7497           | 0.8486               | 0.9317               | 0.7551     | 0.8392         | 0.9276         |
| 10 | 0.7413           | 0.8444               | 0.9295               | 0.7472     | 0.8330         | 0.9252         |
| 15 | 0.7487           | 0.8601               | 0.9401               | 0.7535     | 0.8520         | 0.9372         |
| 20 | 0.7413           | 0.8570               | 0.9369               | 0.7469     | 0.8446         | 0.9327         |

![Plate Thickness vs K-Folds](5seg/metricas/plate_vs_kfolds.png)

![Electrode Type vs K-Folds](5seg/metricas/electrode_vs_kfolds.png)

![Current Type vs K-Folds](5seg/metricas/current_vs_kfolds.png)

### Comparacion de Overlap en Entrenamiento

Se comparo el efecto del solapamiento (overlap) entre segmentos durante el entrenamiento. Con overlap 0.5, cada segmento se superpone 50% con el anterior, generando mas datos de entrenamiento.

**Configuracion:** K=5, Segmentos de 5 segundos, Conjunto Blind

| Overlap | Segmentos Entrenamiento | Acc (Placa) | Acc (Electrodo) | Acc (Corriente) | F1 (Placa) | F1 (Electrodo) | F1 (Corriente) |
| ------- | ----------------------- | ----------- | --------------- | --------------- | ---------- | -------------- | -------------- |
| 0.0     | 3,617                   | 0.5331      | 0.4984          | 0.6614          | 0.5421     | 0.5148         | 0.6613         |
| 0.5     | 7,234                   | 0.7455      | 0.8559          | 0.9295          | 0.7509     | 0.8459         | 0.9251         |

**Diferencia (Overlap 0.5 - Overlap 0.0):**

| Parametro     | Δ Accuracy | Δ F1-Score |
| ------------- | ---------- | ---------- |
| **Placa**     | +0.2124    | +0.2088    |
| **Electrodo** | +0.3575    | +0.3311    |
| **Corriente** | +0.2681    | +0.2638    |

El overlap de 50% duplica la cantidad de segmentos de entrenamiento y mejora significativamente las metricas en el conjunto blind (mejora promedio de +28% en accuracy).

## VI. Tiempos de Ejecucion

### Tiempo de Entrenamiento vs K-Folds (5 segundos)

El tiempo de entrenamiento incluye solo el proceso de K-Fold CV y evaluacion del ensemble, sin incluir la extraccion de embeddings VGGish.

| K-Folds | Tiempo Entrenamiento (min) |
| ------- | -------------------------- |
| 3       | 2.96                       |
| 5       | 3.75                       |
| 7       | 6.91                       |
| 10      | 9.87                       |
| 15      | 14.80                      |
| 20      | 19.73                      |

![Tiempo vs K-Folds - 5 segundos](5seg/metricas/tiempo_vs_kfolds_5seg_2026-02-05_19-51-47.png)

El tiempo de entrenamiento crece aproximadamente de forma lineal con el numero de folds.

### Tiempo de Extraccion VGGish

El tiempo de extraccion de embeddings VGGish se ejecuta una sola vez por configuracion (duracion + overlap) y se cachea para entrenamientos posteriores.

#### Tiempo por Duracion de Segmento

| Duracion | Segmentos | Tiempo VGGish (min) | ms/segmento |
| -------- | --------- | ------------------- | ----------- |
| 2seg     | 18,848    | 5.36                | 17.1        |
| 5seg     | 7,234     | 5.78                | 47.9        |
| 10seg    | 3,372     | 23.71               | 421.9       |
| 50seg    | 389       | 22.70               | 3,500.9     |

El tiempo por segmento aumenta con la duracion porque VGGish procesa el audio en ventanas de 0.96s, generando mas frames para segmentos mas largos:
- **2s**: ~2 frames VGGish
- **5s**: ~5 frames VGGish  
- **10s**: ~10 frames VGGish
- **50s**: ~52 frames VGGish

### Tiempo de Entrenamiento K=10 por Duracion

| Duracion | Segmentos | Tiempo Entrenamiento (min) |
| -------- | --------- | -------------------------- |
| 2seg     | 18,848    | 21.59                      |
| 5seg     | 7,234     | 9.87                       |
| 10seg    | 3,372     | 9.84                       |
| 20seg    | 1,441     | 17.86                      |
| 30seg    | 805       | 5.20                       |
| 50seg    | 389       | 5.32                       |

El tiempo de entrenamiento no es directamente proporcional al numero de segmentos debido a factores como: tamano del batch, early stopping, y complejidad de los datos.









