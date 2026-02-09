# Ejemplos de comandos

Todos los comandos se ejecutan desde la raíz del proyecto (`soldadura/`).

## Preparación de datos y splits

- Generar splits para una duración específica:
  - `python generar_splits.py --duration 10 --overlap 0.5`
  - `python generar_splits.py --duration 30 --overlap 0.0`

- Generar splits con overlap de 75%:
  - `python generar_splits.py --duration 5 --overlap 0.75`

## Entrenamiento

- Entrenar ensemble con 5 folds y 50% de solapamiento:
  - `python entrenar.py --duration 10 --overlap 0.5 --k-folds 5`

- Entrenar ensemble con 10 folds y sin solapamiento:
  - `python entrenar.py --duration 30 --overlap 0.0 --k-folds 10`

- Entrenar sin usar caché de embeddings:
  - `python entrenar.py --duration 5 --overlap 0.5 --k-folds 5 --no-cache`

## Inferencia

- Inferencia de un archivo de audio:
  - `python inferir.py --duration 10 --overlap 0.5 --audio ruta/al/archivo.wav`

- Evaluar en blind (vida real):
  - `python inferir.py --duration 10 --overlap 0.5 --evaluar --k-folds 5`

- Cross-duration: modelo entrenado a 05seg evaluado con segmentos de 30seg:
  - `python inferir.py --duration 30 --overlap 0.5 --train-duration 5 --k-folds 5 --evaluar`

- Mostrar 20 predicciones aleatorias:
  - `python inferir.py --duration 5 --overlap 0.5 --n 20`

## Gráficas y métricas

- Generar matrices de confusión (todas las duraciones):
  - `python scripts/generar_confusion_matrices.py`

- Generar matrices de confusión solo para 30s y último resultado:
  - `python scripts/generar_confusion_matrices.py --duracion 30seg --ultimo`

- Graficar métricas vs duración:
  - `python scripts/graficar_duraciones.py`

- Graficar métricas vs folds para una duración:
  - `python scripts/graficar_folds.py 30seg`

- Comparar efectos del overlap:
  - `python scripts/graficar_overlap.py --save`
  - `python scripts/graficar_overlap.py --heatmap --save`
  - `python scripts/graficar_overlap.py --duration 5 --k-folds 10 --save`
