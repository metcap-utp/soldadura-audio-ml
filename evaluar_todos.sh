#!/bin/bash
# =============================================================================
# Script: Evalúa todos los modelos entrenados
#   - Modelos: xvector, ecapa_tdnn, feedforward
#   - Duraciones: 1, 2, 5, 10, 20, 30, 50 segundos
#   - K-folds: configurable (default: 10)
#   - Overlap: configurable (default: 0.5)
#
# Uso:
#   chmod +x evaluar_todos.sh
#   ./evaluar_todos.sh                          # k=10, overlap=0.5
#   ./evaluar_todos.sh --k-folds 10 --overlap 0.5
#   ./evaluar_todos.sh --model xvector          # Solo xvector
#   ./evaluar_todos.sh --duration 5             # Solo duración 5
# =============================================================================

set -e

# Configuración por defecto
DURATIONS=(1 2 5 10 20 30 50)
OVERLAP=0.5
KFOLDS=10
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Parsear argumentos
FILTER_DURATION=""
FILTER_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --k-folds)
            KFOLDS="$2"
            shift 2
            ;;
        --overlap)
            OVERLAP="$2"
            shift 2
            ;;
        --duration)
            FILTER_DURATION="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Argumento desconocido: $1"
            exit 1
            ;;
    esac
done

MODELS=("xvector" "ecapa_tdnn" "feedforward")

TOTAL=0
EVALUATED=0
SKIPPED=0
FAILED=0

echo "============================================================"
echo "  EVALUACIÓN BATCH - VGGish Backbone"
echo "============================================================"
echo "  Proyecto: $PROJECT_DIR"
echo "  Modelos: ${MODELS[*]}"
echo "  Duraciones: ${DURATIONS[*]}"
echo "  K-folds: $KFOLDS"
echo "  Overlap: $OVERLAP"
if [[ -n "$FILTER_DURATION" ]]; then echo "  Filtro duración: $FILTER_DURATION"; fi
if [[ -n "$FILTER_MODEL" ]]; then echo "  Filtro modelo: $FILTER_MODEL"; fi
echo "============================================================"
echo ""

models_exist() {
    local DUR=$1
    local MODEL=$2
    local MODEL_DIR="$(printf '%02d' $DUR)seg/modelos/${MODEL}/k$(printf '%02d' $KFOLDS)_overlap_${OVERLAP}"

    if [[ ! -d "$MODEL_DIR" ]]; then
        return 1
    fi

    local N_MODELS
    N_MODELS=$(find "$MODEL_DIR" -name "model_fold_*.pth" 2>/dev/null | wc -l)

    if [[ "$N_MODELS" -ge "$KFOLDS" ]]; then
        return 0
    fi

    return 1
}

for MODEL_NAME in "${MODELS[@]}"; do
    if [[ -n "$FILTER_MODEL" && "$MODEL_NAME" != "$FILTER_MODEL" ]]; then
        continue
    fi

    for DUR in "${DURATIONS[@]}"; do
        if [[ -n "$FILTER_DURATION" && "$DUR" != "$FILTER_DURATION" ]]; then
            continue
        fi

        TOTAL=$((TOTAL + 1))
        DUR_DIR="$(printf '%02d' $DUR)seg"

        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  [${TOTAL}] ${MODEL_NAME} | ${DUR_DIR} | K=${KFOLDS} | overlap=${OVERLAP}"
        echo "═══════════════════════════════════════════════════════════════"

        if ! models_exist "$DUR" "$MODEL_NAME"; then
            echo "  [SKIP] No hay modelos entrenados"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo "  [EVAL] Ejecutando inferencia..."
        START=$(date +%s)

        if python inferir.py --duration "$DUR" --overlap "$OVERLAP" --k-folds "$KFOLDS" --model "$MODEL_NAME" --evaluar 2>&1 | tail -20; then
            END=$(date +%s)
            ELAPSED=$((END - START))
            echo "  [EVAL] Completado en ${ELAPSED}s"
            EVALUATED=$((EVALUATED + 1))
        else
            echo "  [EVAL] ERROR"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  RESUMEN"
echo "═══════════════════════════════════════════════════════════════"
echo "  Total:     $TOTAL"
echo "  Evaluados: $EVALUATED"
echo "  Skip:      $SKIPPED"
echo "  Fallidos:  $FAILED"
echo "═══════════════════════════════════════════════════════════════"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
