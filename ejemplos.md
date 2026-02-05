# Ejemplos de comandos

## Preparación de datos y splits

- Generar splits para una duración específica:
  - `python 10seg/generar_splits.py`
  - `python 30seg/generar_splits.py`

## Entrenamiento

- Entrenar ensemble con 5 folds y sin solapamiento:
  - `python 10seg/entrenar.py --k-folds 5 --overlap 0.0`

- Entrenar ensemble con 5 folds y 50% de solapamiento:
  - `python 30seg/entrenar.py --k-folds 5 --overlap 0.5`

## Inferencia

- Inferencia de un archivo de audio:
  - `python 10seg/infer.py --audio ruta/al/archivo.wav`

- Evaluar en blind (vida real):
  - `python 10seg/infer.py --evaluar --k-folds 5 --overlap 0.0`

- Evaluar modelo entrenado a 30s usando audios segmentados a 1s:
  - `python 30seg/infer.py --evaluar --k-folds 5 --overlap 0.5 --train-seconds 30 --test-seconds 1`

## Gráficas y métricas

- Generar matrices de confusión (todas las duraciones):
  - `python scripts/generar_confusion_matrices.py`

- Generar matrices de confusión solo para 30s y último resultado:
  - `python scripts/generar_confusion_matrices.py --duracion 30seg --ultimo`

- Graficar métricas vs duración:
  - `python scripts/graficar_duraciones.py`

- Graficar métricas vs folds para una duración:
  - `python scripts/graficar_folds.py 30seg`
