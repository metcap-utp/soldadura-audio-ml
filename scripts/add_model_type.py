#!/usr/bin/env python3
"""
Script para agregar el campo 'model_type': 'xvector' a todos los archivos JSON
de resultados e inferencia en los directorios de segmentos.
"""

import json
import os
from pathlib import Path

# Directorios a procesar
SEGMENT_DIRS = ['01seg', '02seg', '05seg', '10seg', '20seg', '30seg', '50seg']
BASE_DIR = Path('/home/luis/projects/vggish-backbone')

def add_model_type_to_json(filepath, model_type='xvector'):
    """Agrega el campo model_type a cada entry en el archivo JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Si es una lista, procesar cada entry
        if isinstance(data, list):
            modified = False
            for entry in data:
                if isinstance(entry, dict) and 'model_type' not in entry:
                    entry['model_type'] = model_type
                    modified = True
            
            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"✓ Actualizado: {filepath}")
            else:
                print(f"  Ya tiene model_type: {filepath}")
        else:
            print(f"  Formato inesperado (no es lista): {filepath}")
            
    except FileNotFoundError:
        print(f"  No existe: {filepath}")
    except json.JSONDecodeError as e:
        print(f"  Error JSON en {filepath}: {e}")
    except Exception as e:
        print(f"  Error en {filepath}: {e}")

def main():
    print("Agregando 'model_type': 'xvector' a los archivos JSON...\n")
    
    for seg_dir in SEGMENT_DIRS:
        dir_path = BASE_DIR / seg_dir
        if not dir_path.exists():
            print(f"Directorio no existe: {dir_path}")
            continue
        
        print(f"\nProcesando {seg_dir}/")
        
        # Actualizar resultados.json
        resultados_path = dir_path / 'resultados.json'
        add_model_type_to_json(resultados_path)
        
        # Actualizar inferencia.json
        inferencia_path = dir_path / 'inferencia.json'
        add_model_type_to_json(inferencia_path)
    
    print("\n✓ Proceso completado!")

if __name__ == '__main__':
    main()
