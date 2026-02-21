#!/bin/bash
# =============================================================================
# Script unificado: Entrena y evalúa (inferencia) todos los modelos
#   - Modelos: xvector, ecapa_tdnn, feedforward
#   - Duraciones: 1, 2, 5, 10, 20, 30, 50 segundos
#   - K-folds: 10 (fijo)
#   - Overlap: 0.5 (fijo)
#
# Uso:
#   chmod +x entrenar_todos.sh
#   ./entrenar_todos.sh                          # Entrena y evalúa todo (k=10, overlap=0.5)
#   ./entrenar_todos.sh --k-folds 10 --overlap 0.5 --no-cache  # EXPLÍCITO
#   ./entrenar_todos.sh --dry-run                # Solo muestra qué se haría
#   ./entrenar_todos.sh --duration 5             # Solo duración 5
#   ./entrenar_todos.sh --model xvector          # Solo xvector
#   ./entrenar_todos.sh --model ecapa_tdnn       # Solo ecapa_tdnn
#   ./entrenar_todos.sh --model feedforward      # Solo feedforward
#   ./entrenar_todos.sh --skip-train             # Solo evaluar (saltar entreno)
#   ./entrenar_todos.sh --skip-eval              # Solo entrenar (saltar eval)
#   ./entrenar_todos.sh --force                  # Re-evaluar incluso si existe
#   ./entrenar_todos.sh --force-train            # Re-entrenar incluso si existe
#   ./entrenar_todos.sh --no-cache               # Recalcular embeddings VGGish (ignorar caché)
# =============================================================================

set -e

# Configuración FIJA según requerimientos
DURATIONS=(1 2 5 10 20 30 50)
OVERLAP=0.5
KFOLDS=10

# Directorio del proyecto
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Parsear argumentos
DRY_RUN=false
SKIP_TRAIN=false
SKIP_EVAL=false
FORCE=false
FORCE_TRAIN=false
NO_CACHE=false
FILTER_DURATION=""
FILTER_MODEL=""
FILTER_K=""
FILTER_OVERLAP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --force-train)
            FORCE_TRAIN=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --duration)
            FILTER_DURATION="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --k-folds)
            FILTER_K="$2"
            shift 2
            ;;
        --overlap)
            FILTER_OVERLAP="$2"
            shift 2
            ;;
        *)
            echo "Argumento desconocido: $1"
            exit 1
            ;;
    esac
done

# Aplicar filtros de k-folds y overlap si se especificaron
if [[ -n "$FILTER_K" ]]; then
    KFOLDS=$FILTER_K
fi
if [[ -n "$FILTER_OVERLAP" ]]; then
    OVERLAP=$FILTER_OVERLAP
fi

# Modelos a entrenar
MODELS=("xvector" "ecapa_tdnn" "feedforward")

# Contadores
TOTAL=0
TRAINED=0
TRAIN_SKIPPED=0
TRAIN_FAILED=0
EVALUATED=0
EVAL_SKIPPED=0
EVAL_FAILED=0

echo "============================================================"
echo "  ENTRENAMIENTO + EVALUACIÓN BATCH - VGGish Backbone"
echo "============================================================"
echo "  Proyecto: $PROJECT_DIR"
echo "  Modelos: ${MODELS[*]}"
echo "  Duraciones: ${DURATIONS[*]}"
if [[ -n "$FILTER_K" ]]; then
    echo "  K-folds: $KFOLDS (modificado)"
else
    echo "  K-folds: $KFOLDS (por defecto)"
fi
if [[ -n "$FILTER_OVERLAP" ]]; then
    echo "  Overlap: $OVERLAP (modificado)"
else
    echo "  Overlap: $OVERLAP (por defecto)"
fi
if [[ -n "$FILTER_DURATION" ]]; then echo "  Filtro duración: $FILTER_DURATION"; fi
if [[ -n "$FILTER_MODEL" ]]; then echo "  Filtro modelo: $FILTER_MODEL"; fi
if $SKIP_TRAIN; then echo "  MODO: Solo evaluación (skip entreno)"; fi
if $SKIP_EVAL; then echo "  MODO: Solo entrenamiento (skip eval)"; fi
if $DRY_RUN; then echo "  MODO: DRY RUN (no ejecuta nada)"; fi
if $FORCE; then echo "  Force: Re-evaluar incluso si ya existe"; fi
if $FORCE_TRAIN; then echo "  Force-train: Re-entrenar incluso si ya existe"; fi
if $NO_CACHE; then echo "  No-cache: Recalcular embeddings VGGish (ignorar caché)"; fi
echo "============================================================"
echo ""

# ─── Función: verificar si ya existe evaluación ──────────────────────────────
result_exists() {
    local DUR=$1
    local MODEL=$2
    local INFER_JSON="$(printf '%02d' $DUR)seg/inferencia.json"
    local EXPECTED_ID="$(printf '%02d' $DUR)seg_${MODEL}_k$(printf '%02d' $KFOLDS)_overlap_${OVERLAP}"

    if [[ ! -f "$INFER_JSON" ]]; then
        return 1
    fi

    if grep -q "\"id\": \"${EXPECTED_ID}\"" "$INFER_JSON" 2>/dev/null; then
        return 0
    fi

    return 1
}

# ─── Función: verificar si existen modelos entrenados ────────────────────────
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

# Iterar por cada combinación
for MODEL_NAME in "${MODELS[@]}"; do
    # Aplicar filtro de modelo
    if [[ -n "$FILTER_MODEL" && "$MODEL_NAME" != "$FILTER_MODEL" ]]; then
        continue
    fi

    for DUR in "${DURATIONS[@]}"; do
        # Aplicar filtro de duración
        if [[ -n "$FILTER_DURATION" && "$DUR" != "$FILTER_DURATION" ]]; then
            continue
        fi

        TOTAL=$((TOTAL + 1))
        
        MODEL_DIR="$(printf '%02d' $DUR)seg/modelos/${MODEL_NAME}/k$(printf '%02d' $KFOLDS)_overlap_${OVERLAP}"
        
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  [${TOTAL}] ${MODEL_NAME} | $(printf '%02d' $DUR)seg | K=${KFOLDS} | overlap=${OVERLAP}"
        echo "═══════════════════════════════════════════════════════════════"

        # Determinar script de entrenamiento
        case "$MODEL_NAME" in
            xvector)
                TRAIN_SCRIPT="entrenar.py"
                ;;
            ecapa_tdnn)
                TRAIN_SCRIPT="entrenar_ecapa.py"
                ;;
            feedforward)
                TRAIN_SCRIPT="entrenar_feedforward.py"
                ;;
        esac

        # =============================================================
        # FASE 1: ENTRENAMIENTO
        # =============================================================
        if ! $SKIP_TRAIN; then
            if ! $FORCE_TRAIN && models_exist "$DUR" "$MODEL_NAME"; then
                echo "  [ENTRENO] Ya existe - saltando"
                TRAIN_SKIPPED=$((TRAIN_SKIPPED + 1))
            else
                if $FORCE_TRAIN && models_exist "$DUR" "$MODEL_NAME"; then
                    echo "  [ENTRENO] Ya existe pero --force-train activado - re-entrenando..."
                else
                    echo "  [ENTRENO] Iniciando entrenamiento..."
                fi
                echo "    Script: $TRAIN_SCRIPT"
                echo "    Destino: $MODEL_DIR"

                # Construir comando base
                TRAIN_CMD="python $TRAIN_SCRIPT --duration $DUR --overlap $OVERLAP --k-folds $KFOLDS"
                if $NO_CACHE; then
                    TRAIN_CMD="$TRAIN_CMD --no-cache"
                fi
                
                if $DRY_RUN; then
                    echo "    [DRY RUN] $TRAIN_CMD"
                    TRAINED=$((TRAINED + 1))
                else
                    START_TIME=$(date +%s)
                    if $TRAIN_CMD 2>&1 | tail -20; then
                        END_TIME=$(date +%s)
                        ELAPSED=$((END_TIME - START_TIME))
                        echo "    [ENTRENO] Completado en ${ELAPSED}s ($((ELAPSED / 60))min)"
                        TRAINED=$((TRAINED + 1))
                    else
                        echo "    [ENTRENO] ERROR - Falló el entrenamiento"
                        TRAIN_FAILED=$((TRAIN_FAILED + 1))
                        continue
                    fi
                fi
            fi
        fi

        # =============================================================
        # FASE 2: EVALUACIÓN (INFERENCIA)
        # =============================================================
        if ! $SKIP_EVAL; then
            if [[ "$FORCE" == "false" ]] && result_exists "$DUR" "$MODEL_NAME"; then
                echo "  [EVAL] Ya evaluado - saltando (usa --force para re-evaluar)"
                EVAL_SKIPPED=$((EVAL_SKIPPED + 1))
            else
                # Verificar que existan modelos para evaluar
                if ! models_exist "$DUR" "$MODEL_NAME"; then
                    echo "  [EVAL] No hay modelos entrenados - saltando evaluación"
                    continue
                fi

                echo "  [EVAL] Iniciando inferencia/evaluación..."

                if $DRY_RUN; then
                    echo "    [DRY RUN] python inferir.py --duration $DUR --overlap $OVERLAP --k-folds $KFOLDS --model $MODEL_NAME --evaluar"
                    EVALUATED=$((EVALUATED + 1))
                else
                    EVAL_START=$(date +%s)
                    
                    if python inferir.py --duration "$DUR" --overlap "$OVERLAP" --k-folds "$KFOLDS" --model "$MODEL_NAME" --evaluar 2>&1 | tail -30; then
                        EVAL_END=$(date +%s)
                        EVAL_TIME=$((EVAL_END - EVAL_START))
                        echo "    [EVAL] Completado en ${EVAL_TIME}s"
                        EVALUATED=$((EVALUATED + 1))
                    else
                        EVAL_END=$(date +%s)
                        EVAL_TIME=$((EVAL_END - EVAL_START))
                        echo "    [EVAL] ERROR - Falló la evaluación después de ${EVAL_TIME}s"
                        EVAL_FAILED=$((EVAL_FAILED + 1))
                    fi
                fi
            fi
        fi
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  RESUMEN FINAL"
echo "═══════════════════════════════════════════════════════════════"
echo "  Total combinaciones: $TOTAL"
echo ""
if ! $SKIP_TRAIN; then
    echo "  ENTRENAMIENTO:"
    echo "    Completados:     $TRAINED"
    echo "    Ya existían:     $TRAIN_SKIPPED"
    echo "    Fallidos:        $TRAIN_FAILED"
    echo ""
fi
if ! $SKIP_EVAL; then
    echo "  EVALUACIÓN:"
    echo "    Completadas:     $EVALUATED"
    echo "    Ya existían:     $EVAL_SKIPPED"
    echo "    Fallidas:        $EVAL_FAILED"
fi
echo "═══════════════════════════════════════════════════════════════"

if [[ $TRAIN_FAILED -gt 0 || $EVAL_FAILED -gt 0 ]]; then
    exit 1
fi
