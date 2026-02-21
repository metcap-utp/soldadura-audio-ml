"""
Script para normalizar el formato de resultados.json de VGGish
al mismo formato que Spectral-analysis.
"""
import json
from pathlib import Path
from typing import Dict, List, Any

# Mapeo de claves de VGGish a Spectral
KEY_MAPPING = {
    'acc_plate': 'accuracy_plate',
    'acc_electrode': 'accuracy_electrode', 
    'acc_current': 'accuracy_current',
    'f1_plate': 'f1_plate',
    'f1_electrode': 'f1_electrode',
    'f1_current': 'f1_current',
}

def transform_fold_results(vggish_folds: List[Dict]) -> List[Dict]:
    """Transforma fold_results de VGGish a formato Spectral."""
    spectral_folds = []
    
    for i, fold in enumerate(vggish_folds):
        new_fold = {'fold': i}
        
        # Mapear métricas
        for vggish_key, spectral_key in KEY_MAPPING.items():
            if vggish_key in fold:
                new_fold[spectral_key] = fold[vggish_key]
        
        # Agregar tiempo
        if 'time_seconds' in fold:
            new_fold['time_seconds'] = fold['time_seconds']
        
        spectral_folds.append(new_fold)
    
    return spectral_folds

def create_blind_evaluation_from_inferencia(inferencia_path: Path) -> Dict:
    """Crea blind_evaluation desde inferencia.json."""
    if not inferencia_path.exists():
        return {}
    
    with open(inferencia_path) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Buscar entrada con k10 overlap 0.5
        for entry in data:
            config = entry.get('config', {})
            models_dir = config.get('models_dir', '')
            if 'k10_overlap_0.5' in models_dir:
                # Transformar a formato blind_evaluation
                blind = {}
                
                if 'accuracy' in entry:
                    acc = entry['accuracy']
                    blind['plate'] = {
                        'accuracy': acc.get('plate_thickness', 0),
                        'f1': entry.get('macro_f1', {}).get('plate_thickness', 0)
                    }
                    blind['electrode'] = {
                        'accuracy': acc.get('electrode', 0),
                        'f1': entry.get('macro_f1', {}).get('electrode', 0)
                    }
                    blind['current'] = {
                        'accuracy': acc.get('current_type', 0),
                        'f1': entry.get('macro_f1', {}).get('current_type', 0)
                    }
                
                if 'global_metrics' in entry:
                    gm = entry['global_metrics']
                    blind['global'] = {
                        'exact_match': gm.get('exact_match', gm.get('exact_match_accuracy', 0)),
                        'hamming_accuracy': gm.get('hamming_accuracy', 0)
                    }
                
                return blind
    
    return {}

def process_duration(duration: str):
    """Procesa una duración específica."""
    vggish_base = Path('/home/luis/projects/vggish-backbone')
    resultados_path = vggish_base / duration / 'resultados.json'
    inferencia_path = vggish_base / duration / 'inferencia.json'
    
    if not resultados_path.exists():
        print(f"{duration}: resultados.json not found")
        return
    
    with open(resultados_path) as f:
        data = json.load(f)
    
    modified = False
    
    for entry in data:
        config = entry.get('config', {})
        models_dir = config.get('models_dir', '')
        
        # Solo procesar k10 overlap 0.5
        if 'k10_overlap_0.5' in models_dir:
            print(f"{duration}: Processing k10_overlap_0.5...")
            
            # 1. Transformar fold_results si existe
            if 'fold_results' in entry:
                entry['fold_results'] = transform_fold_results(entry['fold_results'])
                print(f"  - Transformed {len(entry['fold_results'])} folds")
            
            # 2. Agregar blind_evaluation si no existe
            if 'blind_evaluation' not in entry:
                blind = create_blind_evaluation_from_inferencia(inferencia_path)
                if blind:
                    entry['blind_evaluation'] = blind
                    print(f"  - Added blind_evaluation")
                else:
                    print(f"  - WARNING: Could not create blind_evaluation")
            
            modified = True
    
    if modified:
        # Guardar resultado
        with open(resultados_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"{duration}: Saved changes")
    else:
        print(f"{duration}: No k10_overlap_0.5 found to process")

def main():
    durations = ['01seg', '02seg', '05seg', '10seg', '20seg', '30seg', '50seg']
    
    print("=" * 70)
    print("NORMALIZANDO FORMATO DE VGGish A SPECTRAL")
    print("=" * 70)
    print()
    
    for dur in durations:
        process_duration(dur)
        print()
    
    print("=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    main()
