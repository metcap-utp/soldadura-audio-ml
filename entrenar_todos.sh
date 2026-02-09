#!/bin/bash
# =============================================================================
# Script de entrenamiento batch para todas las combinaciones de:
#   - Duraciones: 1, 2, 5, 10, 20, 30, 50 segundos
#   - Overlaps: 0.0, 0.25, 0.5, 0.75
#   - K-folds: 5, 10
#
# IMPORTANTE: Los CSVs (train.csv, blind.csv, etc.) dependen del overlap
# porque los Segment Index cambian. Por eso, generar_splits.py se ejecuta
# antes de cada grupo de overlap.
#
# Uso:
#   chmod +x entrenar_todos.sh
#   ./entrenar_todos.sh                          # Entrena todo lo que falta
#   ./entrenar_todos.sh --dry-run                # Solo muestra qué se haría
#   ./entrenar_todos.sh --duration 5             # Solo duración 5
#   ./entrenar_todos.sh --overlap 0.5            # Solo overlap 0.5
#   ./entrenar_todos.sh --k-folds 5              # Solo K=5
# =============================================================================

set -e

# Configuración
DURATIONS=(1 2 5 10 20 30 50)
OVERLAPS=(0.0 0.25 0.5 0.75)
KFOLDS=(5 10)

# Directorio del proyecto
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Parsear argumentos
DRY_RUN=false
FILTER_DURATION=""
FILTER_OVERLAP=""
FILTER_K=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --duration)
            FILTER_DURATION="$2"
            shift 2
            ;;
        --overlap)
            FILTER_OVERLAP="$2"
            shift 2
            ;;
        --k-folds)
            FILTER_K="$2"
            shift 2
            ;;
        *)
            echo "Argumento desconocido: $1"
            exit 1
            ;;
    esac
done

# Contadores
TOTAL=0
SKIPPED=0
TRAINED=0
FAILED=0
SPLITS_GENERATED=0

echo "============================================================"
echo "  ENTRENAMIENTO BATCH - SMAW Audio Classification"
echo "============================================================"
echo "  Proyecto: $PROJECT_DIR"
echo "  Duraciones: ${DURATIONS[*]}"
echo "  Overlaps: ${OVERLAPS[*]}"
echo "  K-folds: ${KFOLDS[*]}"
if [[ -n "$FILTER_DURATION" ]]; then echo "  Filtro duración: $FILTER_DURATION"; fi
if [[ -n "$FILTER_OVERLAP" ]]; then echo "  Filtro overlap: $FILTER_OVERLAP"; fi
if [[ -n "$FILTER_K" ]]; then echo "  Filtro K: $FILTER_K"; fi
if $DRY_RUN; then echo "  MODO: DRY RUN (no ejecuta nada)"; fi
echo "============================================================"
echo ""

# Iterar por cada combinación
for DUR in "${DURATIONS[@]}"; do
    # Aplicar filtro de duración
    if [[ -n "$FILTER_DURATION" && "$DUR" != "$FILTER_DURATION" ]]; then
        continue
    fi

    for OV in "${OVERLAPS[@]}"; do
        # Aplicar filtro de overlap
        if [[ -n "$FILTER_OVERLAP" && "$OV" != "$FILTER_OVERLAP" ]]; then
            continue
        fi

        # Verificar si necesitamos regenerar splits para este overlap
        NEEDS_SPLITS=false

        for K in "${KFOLDS[@]}"; do
            if [[ -n "$FILTER_K" && "$K" != "$FILTER_K" ]]; then
                continue
            fi

            MODEL_DIR="$(printf '%02d' $DUR)seg/modelos/k$(printf '%02d' $K)_overlap_${OV}"
            TOTAL=$((TOTAL + 1))

            if [[ -d "$MODEL_DIR" ]] && ls "$MODEL_DIR"/model_fold_*.pth &>/dev/null; then
                NUM_MODELS=$(ls "$MODEL_DIR"/model_fold_*.pth 2>/dev/null | wc -l)
                if [[ "$NUM_MODELS" -ge "$K" ]]; then
                    echo "[SKIP] $(printf '%02d' $DUR)seg K=$K overlap=$OV ($NUM_MODELS modelos encontrados)"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi
            fi

            NEEDS_SPLITS=true
        done

        # Generar splits si hay al menos un modelo faltante
        if $NEEDS_SPLITS; then
            echo ""
            echo "--- Generando splits: $(printf '%02d' $DUR)seg overlap=$OV ---"
            if $DRY_RUN; then
                echo "[DRY RUN] python generar_splits.py --duration $DUR --overlap $OV"
            else
                python generar_splits.py --duration "$DUR" --overlap "$OV" 2>&1 | tail -5
                SPLITS_GENERATED=$((SPLITS_GENERATED + 1))
            fi
        fi

        # Entrenar modelos faltantes
        for K in "${KFOLDS[@]}"; do
            if [[ -n "$FILTER_K" && "$K" != "$FILTER_K" ]]; then
                continue
            fi

            MODEL_DIR="$(printf '%02d' $DUR)seg/modelos/k$(printf '%02d' $K)_overlap_${OV}"

            # Verificar si ya existe un entrenamiento completo
            if [[ -d "$MODEL_DIR" ]] && ls "$MODEL_DIR"/model_fold_*.pth &>/dev/null; then
                NUM_MODELS=$(ls "$MODEL_DIR"/model_fold_*.pth 2>/dev/null | wc -l)
                if [[ "$NUM_MODELS" -ge "$K" ]]; then
                    continue  # Ya fue contado como SKIP arriba
                fi
            fi

            echo ""
            echo ">>> ENTRENANDO: $(printf '%02d' $DUR)seg K=$K overlap=$OV"
            echo "    Destino: $MODEL_DIR"

            if $DRY_RUN; then
                echo "[DRY RUN] python entrenar.py --duration $DUR --overlap $OV --k-folds $K"
                TRAINED=$((TRAINED + 1))
            else
                START_TIME=$(date +%s)
                if python entrenar.py --duration "$DUR" --overlap "$OV" --k-folds "$K" 2>&1 | tail -20; then
                    END_TIME=$(date +%s)
                    ELAPSED=$((END_TIME - START_TIME))
                    echo "    Completado en ${ELAPSED}s ($((ELAPSED / 60))min)"
                    TRAINED=$((TRAINED + 1))
                else
                    echo "    [ERROR] Falló el entrenamiento"
                    FAILED=$((FAILED + 1))
                fi
            fi
        done
    done
done

echo ""
echo "============================================================"
echo "  RESUMEN"
echo "============================================================"
echo "  Total combinaciones: $TOTAL"
echo "  Ya existían (skip):  $SKIPPED"
echo "  Entrenados:          $TRAINED"
echo "  Fallidos:            $FAILED"
echo "  Splits generados:    $SPLITS_GENERATED"
echo "============================================================"

if [[ $TRAINED -gt 0 ]] && ! $DRY_RUN; then
    echo ""
    echo "Para evaluar los modelos entrenados en blind:"
    echo "  python infer.py --duration <N> --overlap <ratio> --k-folds <K> --evaluar"
    echo ""
    echo "Para generar gráficas de comparación de overlap:"
    echo "  python scripts/graficar_overlap.py --save"
fi
