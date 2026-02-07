#!/bin/bash
# ==============================================================================
# evaluar_todos.sh - Evaluación ciega (blind) de todos los modelos entrenados
# ==============================================================================
# Ejecuta inferir.py --evaluar para todas las combinaciones de:
#   - Duraciones: 1, 2, 5, 10, 20, 30, 50 segundos
#   - Overlaps: 0.0, 0.25, 0.5, 0.75
#   - K-Folds: 5, 10
#
# Uso:
#   bash evaluar_todos.sh                          # Evaluar todo lo faltante
#   bash evaluar_todos.sh --dry-run                # Solo mostrar qué se ejecutaría
#   bash evaluar_todos.sh --duration 5             # Solo duración 5
#   bash evaluar_todos.sh --overlap 0.5            # Solo overlap 0.5
#   bash evaluar_todos.sh --k-folds 10             # Solo K=10
#   bash evaluar_todos.sh --force                  # Re-evaluar incluso si ya existe
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Configuración ───────────────────────────────────────────────────────────
DURATIONS=(1 2 5 10 20 30 50)
OVERLAPS=(0.0 0.25 0.5 0.75)
KFOLDS=(5 10)

# ─── Argumentos ──────────────────────────────────────────────────────────────
DRY_RUN=false
FORCE=false
FILTER_DUR=""
FILTER_OV=""
FILTER_K=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --force)      FORCE=true; shift ;;
        --duration)   FILTER_DUR="$2"; shift 2 ;;
        --overlap)    FILTER_OV="$2"; shift 2 ;;
        --k-folds)    FILTER_K="$2"; shift 2 ;;
        *)            echo "Argumento desconocido: $1"; exit 1 ;;
    esac
done

# Aplicar filtros
if [[ -n "$FILTER_DUR" ]]; then DURATIONS=($FILTER_DUR); fi
if [[ -n "$FILTER_OV" ]];  then OVERLAPS=($FILTER_OV); fi
if [[ -n "$FILTER_K" ]];   then KFOLDS=($FILTER_K); fi

# ─── Contadores ──────────────────────────────────────────────────────────────
TOTAL=0
SKIPPED=0
EVALUATED=0
FAILED=0
NO_MODELS=0

# ─── Log ─────────────────────────────────────────────────────────────────────
LOG_FILE="evaluar_todos.log"
echo "═══════════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo " Evaluación ciega masiva - $(date)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "Duraciones: ${DURATIONS[*]}" | tee -a "$LOG_FILE"
echo "Overlaps:   ${OVERLAPS[*]}" | tee -a "$LOG_FILE"
echo "K-Folds:    ${KFOLDS[*]}" | tee -a "$LOG_FILE"
echo "Dry-run:    $DRY_RUN" | tee -a "$LOG_FILE"
echo "Force:      $FORCE" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# ─── Función: verificar si ya existe resultado en inferencia.json ───────────────
result_exists() {
    local DUR=$1
    local OV=$2
    local K=$3
    local INFER_JSON="$(printf '%02d' $DUR)seg/inferencia.json"
    local EXPECTED_ID="$(printf '%02d' $DUR)seg_k$(printf '%02d' $K)_overlap_${OV}"

    if [[ ! -f "$INFER_JSON" ]]; then
        return 1  # No existe el archivo
    fi

    # Buscar el ID exacto en inferencia.json
    if grep -q "\"id\": \"${EXPECTED_ID}\"" "$INFER_JSON" 2>/dev/null; then
        return 0  # Existe
    fi

    return 1  # No existe
}

# ─── Función: verificar si existen modelos ───────────────────────────────────
models_exist() {
    local DUR=$1
    local OV=$2
    local K=$3
    local MODEL_DIR="$(printf '%02d' $DUR)seg/modelos/k$(printf '%02d' $K)_overlap_${OV}"

    if [[ ! -d "$MODEL_DIR" ]]; then
        return 1
    fi

    local N_MODELS
    N_MODELS=$(find "$MODEL_DIR" -name "model_fold_*.pth" 2>/dev/null | wc -l)

    if [[ "$N_MODELS" -ge "$K" ]]; then
        return 0
    fi

    return 1
}

# ─── Función: verificar si existe CSV de blind ───────────────────────────────
blind_csv_exists() {
    local DUR=$1
    local OV=$2
    local CSV="$(printf '%02d' $DUR)seg/blind_overlap_${OV}.csv"

    if [[ -f "$CSV" ]]; then
        return 0
    fi

    # Fallback al genérico
    if [[ -f "$(printf '%02d' $DUR)seg/blind.csv" ]]; then
        return 0
    fi

    return 1
}

# ─── Bucle principal ─────────────────────────────────────────────────────────
GLOBAL_START=$(date +%s)

for DUR in "${DURATIONS[@]}"; do
    for OV in "${OVERLAPS[@]}"; do
        for K in "${KFOLDS[@]}"; do
            TOTAL=$((TOTAL + 1))
            COMBO="dur=${DUR}s ov=${OV} K=${K}"

            # Verificar si existen modelos
            if ! models_exist "$DUR" "$OV" "$K"; then
                echo "  ⊘ [$COMBO] Sin modelos - saltando" | tee -a "$LOG_FILE"
                NO_MODELS=$((NO_MODELS + 1))
                continue
            fi

            # Verificar si ya tiene resultado (a menos que --force)
            if [[ "$FORCE" == "false" ]] && result_exists "$DUR" "$OV" "$K"; then
                echo "  ✓ [$COMBO] Ya evaluado - saltando" | tee -a "$LOG_FILE"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            # Verificar CSV de blind
            if ! blind_csv_exists "$DUR" "$OV"; then
                echo "  ⚠ [$COMBO] Sin CSV blind - generando splits..." | tee -a "$LOG_FILE"
                if [[ "$DRY_RUN" == "false" ]]; then
                    python generar_splits.py --duration "$DUR" --overlap "$OV" 2>&1 | tail -1 | tee -a "$LOG_FILE"
                fi
            fi

            # Ejecutar evaluación
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "  → [$COMBO] Se ejecutaría: python inferir.py --duration $DUR --overlap $OV --k-folds $K --evaluar" | tee -a "$LOG_FILE"
            else
                echo "" | tee -a "$LOG_FILE"
                echo "───────────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
                echo "  ▶ [$COMBO] Evaluando..." | tee -a "$LOG_FILE"
                echo "───────────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"

                EVAL_START=$(date +%s)

                if python inferir.py --duration "$DUR" --overlap "$OV" --k-folds "$K" --evaluar 2>&1 | tee -a "$LOG_FILE"; then
                    EVAL_END=$(date +%s)
                    EVAL_TIME=$((EVAL_END - EVAL_START))
                    echo "  ✓ [$COMBO] Completado en ${EVAL_TIME}s" | tee -a "$LOG_FILE"
                    EVALUATED=$((EVALUATED + 1))
                else
                    EVAL_END=$(date +%s)
                    EVAL_TIME=$((EVAL_END - EVAL_START))
                    echo "  ✗ [$COMBO] FALLÓ después de ${EVAL_TIME}s" | tee -a "$LOG_FILE"
                    FAILED=$((FAILED + 1))
                fi
            fi
        done
    done
done

GLOBAL_END=$(date +%s)
GLOBAL_TIME=$((GLOBAL_END - GLOBAL_START))

# ─── Resumen ─────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo " RESUMEN DE EVALUACIÓN" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  Total combinaciones:  $TOTAL" | tee -a "$LOG_FILE"
echo "  Evaluadas:            $EVALUATED" | tee -a "$LOG_FILE"
echo "  Ya existentes:        $SKIPPED" | tee -a "$LOG_FILE"
echo "  Sin modelos:          $NO_MODELS" | tee -a "$LOG_FILE"
echo "  Fallidas:             $FAILED" | tee -a "$LOG_FILE"
echo "  Tiempo total:         ${GLOBAL_TIME}s ($(echo "scale=1; $GLOBAL_TIME/60" | bc)m)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

if [[ $FAILED -gt 0 ]]; then
    echo "" | tee -a "$LOG_FILE"
    echo "⚠ Hubo $FAILED evaluaciones fallidas. Revisa el log: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "✓ Proceso completado exitosamente" | tee -a "$LOG_FILE"
