#!/usr/bin/env python3
"""
Script para actualizar los archivos resultados.json existentes con:
- confusion_matrix_plate, confusion_matrix_electrode, confusion_matrix_current por fold
- time_seconds por fold
"""

import json
from pathlib import Path

SEGMENT_DIRS = ['01seg', '02seg', '05seg', '10seg', '20seg', '30seg', '50seg']
BASE_DIR = Path('/home/luis/projects/vggish-backbone')

def update_resultados_json(filepath):
    """Actualiza resultados.json con campos faltantes."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"  Formato no es lista: {filepath}")
            return False
        
        modified = False
        for entry in data:
            if not isinstance(entry, dict):
                continue
            
            # Verificar fold_results
            fold_results = entry.get('fold_results', [])
            fold_training_times = entry.get('fold_training_times_seconds', [])
            
            if not fold_results:
                continue
            
            for i, fold in enumerate(fold_results):
                # Agregar time_seconds si existe en fold_training_times
                if i < len(fold_training_times) and 'time_seconds' not in fold:
                    fold['time_seconds'] = fold_training_times[i]
                    modified = True
                
                # Agregar confusion matrices vacías si no existen
                if 'confusion_matrix_plate' not in fold:
                    fold['confusion_matrix_plate'] = []
                    modified = True
                if 'confusion_matrix_electrode' not in fold:
                    fold['confusion_matrix_electrode'] = []
                    modified = True
                if 'confusion_matrix_current' not in fold:
                    fold['confusion_matrix_current'] = []
                    modified = True
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✓ Actualizado: {filepath}")
            return True
        else:
            print(f"  Sin cambios: {filepath}")
            return False
            
    except FileNotFoundError:
        print(f"  No existe: {filepath}")
        return False
    except json.JSONDecodeError as e:
        print(f"  Error JSON en {filepath}: {e}")
        return False
    except Exception as e:
        print(f"  Error en {filepath}: {e}")
        return False

def update_inferencia_json(filepath):
    """Asegura que inferencia.json tenga model_type."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"  Formato no es lista: {filepath}")
            return False
        
        modified = False
        for entry in data:
            if isinstance(entry, dict) and 'model_type' not in entry:
                entry['model_type'] = 'xvector'
                modified = True
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✓ Actualizado inferencia.json: {filepath}")
            return True
        else:
            print(f"  Sin cambios: {filepath}")
            return False
            
    except FileNotFoundError:
        print(f"  No existe: {filepath}")
        return False
    except Exception as e:
        print(f"  Error en {filepath}: {e}")
        return False

def main():
    print("Actualizando archivos JSON con campos faltantes...\n")
    
    for seg_dir in SEGMENT_DIRS:
        dir_path = BASE_DIR / seg_dir
        if not dir_path.exists():
            print(f"Directorio no existe: {dir_path}")
            continue
        
        print(f"\nProcesando {seg_dir}/")
        
        # Actualizar resultados.json
        resultados_path = dir_path / 'resultados.json'
        update_resultados_json(resultados_path)
        
        # Actualizar inferencia.json
        inferencia_path = dir_path / 'inferencia.json'
        update_inferencia_json(inferencia_path)
    
    print("\n✓ Proceso completado!")

if __name__ == '__main__':
    main()
