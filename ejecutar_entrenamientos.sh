#!/bin/bash
#
# Script para ejecutar todos los entrenamientos con diferentes valores de K
# 
# Uso: ./ejecutar_entrenamientos.sh
#
# Este script ejecuta:
# 1. Primero: Entrenamiento para 20seg y 30seg con k=5
# 2. Después: Entrenamiento para TODAS las duraciones con k=3, 7, 10, 15, 20
#

set -e  # Salir si hay error

BASE_DIR="/home/luis/projects/tesis/audio/soldadura"
DURACIONES=("1seg" "2seg" "5seg" "10seg" "20seg" "30seg" "50seg")
K_VALUES=(3 7 10 15 20)

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "       ENTRENAMIENTO MASIVO DE MODELOS SMAW"
echo "========================================================================"
echo ""
echo "Duraciones: ${DURACIONES[*]}"
echo "Valores de K adicionales: ${K_VALUES[*]}"
echo ""

# Función para verificar si ya existe el modelo
modelo_existe() {
    local duracion=$1
    local k=$2
    local model_dir="${BASE_DIR}/${duracion}/models/${k}-fold"
    if [ -d "$model_dir" ] && [ "$(ls -A $model_dir 2>/dev/null)" ]; then
        return 0  # Existe
    else
        return 1  # No existe
    fi
}

# Función para entrenar
entrenar() {
    local duracion=$1
    local k=$2
    
    echo -e "${YELLOW}----------------------------------------${NC}"
    echo -e "${YELLOW}Entrenando: ${duracion} con k=${k}${NC}"
    echo -e "${YELLOW}Hora inicio: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${YELLOW}----------------------------------------${NC}"
    
    cd "${BASE_DIR}/${duracion}"
    
    # Verificar si existen los splits
    if [ ! -f "train.csv" ]; then
        echo -e "${YELLOW}  -> Generando splits para ${duracion}...${NC}"
        python generar_splits.py
    fi
    
    # Verificar si ya existe el modelo
    if modelo_existe "$duracion" "$k"; then
        echo -e "${GREEN}  -> Modelo ya existe para ${duracion} k=${k}, saltando...${NC}"
    else
        echo -e "${YELLOW}  -> Ejecutando entrenamiento...${NC}"
        python entrenar.py --k-folds $k
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  -> ✓ Completado: ${duracion} k=${k}${NC}"
        else
            echo -e "${RED}  -> ✗ Error: ${duracion} k=${k}${NC}"
        fi
    fi
    
    echo -e "${CYAN}Hora fin: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
}

# Contador de tiempo
START_TIME=$(date +%s)

echo ""
echo "========================================================================"
echo "FASE 1: Entrenamiento 20seg y 30seg con k=5"
echo "========================================================================"
echo ""

# Primero: 20seg y 30seg con k=5
for duracion in "20seg" "30seg"; do
    entrenar "$duracion" 5
    echo ""
done

echo ""
echo "========================================================================"
echo "FASE 2: Entrenamiento TODAS las duraciones con k=3,7,10,15,20"
echo "========================================================================"
echo ""

# Ejecutar entrenamientos para cada duración y k
for duracion in "${DURACIONES[@]}"; do
    for k in "${K_VALUES[@]}"; do
        entrenar "$duracion" "$k"
        echo ""
    done
done

# Tiempo total
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "========================================================================"
echo -e "${GREEN}ENTRENAMIENTO COMPLETADO${NC}"
echo "========================================================================"
echo "Tiempo total: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
