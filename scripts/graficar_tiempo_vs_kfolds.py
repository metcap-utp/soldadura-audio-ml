#!/usr/bin/env python3
"""
Script para graficar el tiempo de entrenamiento vs la cantidad de k-folds.
Estilos consistentes con graficar_folds.py
"""

import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime

# Colores consistentes con otros scripts
COLORS = {
    "training": "#3498db",  # Azul
    "vggish": "#e74c3c",    # Rojo
    "total": "#2c3e50",     # Gris oscuro
}


def extract_kfold_times(results_file: Path) -> dict:
    """
    Extrae los tiempos por k-folds de un archivo resultados.json.
    Prioriza el campo 'training_time' si existe, sino calcula a partir de execution_time.
    
    Returns:
        dict con keys: k_values, training_times, vggish_times
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    kfold_data = {}
    for entry in results:
        if 'config' not in entry:
            continue
            
        k = entry['config'].get('n_folds')
        if k is None:
            continue
        
        # Tiempo de extraccion VGGish (si existe)
        vggish_info = entry.get('vggish_extraction', {})
        vggish_time_seconds = vggish_info.get('extraction_time_seconds')
        from_cache = vggish_info.get('from_cache', True)
        
        # Priorizar training_time si existe (tiempo puro de entrenamiento)
        if 'training_time' in entry and entry['training_time'] is not None:
            training_only_seconds = entry['training_time'].get('seconds', 0)
            total_seconds = entry.get('execution_time', {}).get('seconds', training_only_seconds)
            
            kfold_data[k] = {
                'total_time': total_seconds / 60,
                'training_time': training_only_seconds / 60,
                'vggish_time': vggish_time_seconds / 60 if vggish_time_seconds else None,
                'from_cache': from_cache,
            }
        elif 'execution_time' in entry:
            time_seconds = entry['execution_time'].get('seconds')
            if time_seconds is not None:
                # Calcular tiempo de entrenamiento puro (sin VGGish) - fallback
                if not from_cache and vggish_time_seconds:
                    training_only_seconds = time_seconds - vggish_time_seconds
                else:
                    training_only_seconds = time_seconds
                
                kfold_data[k] = {
                    'total_time': time_seconds / 60,
                    'training_time': training_only_seconds / 60,
                    'vggish_time': vggish_time_seconds / 60 if vggish_time_seconds else None,
                    'from_cache': from_cache,
                }
    
    return dict(sorted(kfold_data.items()))


def plot_training_time_vs_kfolds(data: dict, duration: str, output_dir: Path):
    """
    Genera el grafico de tiempo de entrenamiento vs k-folds.
    Solo grafica el tiempo de entrenamiento puro (sin extraccion VGGish).
    """
    if not data:
        print("No hay datos para graficar")
        return
    
    k_values = list(data.keys())
    training_times = [data[k]['training_time'] for k in k_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Grafico de tiempo de entrenamiento puro
    ax.plot(k_values, training_times, 
            color=COLORS['training'],
            marker='o', 
            linewidth=2, 
            markersize=8, 
            label='Tiempo de entrenamiento')
    
    # Agregar etiquetas de valor en cada punto
    for k, t in zip(k_values, training_times):
        ax.annotate(f'{t:.1f}', 
                   (k, t), 
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center',
                   fontsize=9,
                   color=COLORS['training'])
    
    ax.set_xlabel('Numero de K-Folds', fontsize=12)
    ax.set_ylabel('Tiempo (minutos)', fontsize=12)
    ax.set_title(f'Tiempo de Entrenamiento vs K-Folds\nSegmentos de {duration}', fontsize=14)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Ajustar limites
    ax.set_xlim(min(k_values) - 1, max(k_values) + 1)
    ax.set_ylim(0, max(training_times) * 1.2)
    
    plt.tight_layout()
    
    # Guardar
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"tiempo_vs_kfolds_{duration}_{timestamp}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Grafico guardado en: {output_file}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Graficar tiempo de entrenamiento vs k-folds')
    parser.add_argument('--duration', '-d', type=str, default='05seg',
                       help='Duracion de los segmentos (ej: 05seg, 10seg)'))
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Directorio de salida para el grafico')
    
    args = parser.parse_args()
    
    # Determinar rutas
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    results_file = project_dir / args.duration / 'resultados.json'
    
    if not results_file.exists():
        print(f"Error: No se encontro {results_file}")
        return
    
    output_dir = Path(args.output) if args.output else project_dir / args.duration / 'metricas'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extraer datos
    print(f"Leyendo datos de: {results_file}")
    data = extract_kfold_times(results_file)
    
    if not data:
        print("No se encontraron entrenamientos con tiempos registrados")
        return
    
    print(f"\nDatos encontrados:")
    print("-" * 50)
    print(f"{'K-Folds':<10} {'Entrenamiento':<18} {'VGGish':<15} {'Cache'}")
    print("-" * 50)
    for k, info in data.items():
        vggish_str = f"{info['vggish_time']:.2f} min" if info['vggish_time'] else "-"
        cache_str = "Si" if info['from_cache'] else "No"
        print(f"  K={k:<6} {info['training_time']:>10.2f} min    {vggish_str:<15} {cache_str}")
    print("-" * 50)
    
    # Graficar
    plot_training_time_vs_kfolds(data, args.duration, output_dir)


if __name__ == '__main__':
    main()
