"""
Grafica métricas vs cantidad de folds para una duración específica.

Uso:
    python graficar_folds.py 5seg              # Grafica para 5 segundos
    python graficar_folds.py 10seg --save      # Guarda la imagen
    python graficar_folds.py 5seg --metric f1  # Solo métrica F1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent


def load_results(duration_dir: str) -> list:
    """Carga los resultados de entrenamiento."""
    results_path = ROOT_DIR / duration_dir / "results.json"

    if not results_path.exists():
        print(f"Error: No se encontró {results_path}")
        print(f"Ejecuta primero: cd {duration_dir} && python entrenar.py --k-folds N")
        sys.exit(1)

    with open(results_path, "r") as f:
        results = json.load(f)

    if not isinstance(results, list):
        results = [results]

    return results


def extract_metrics_by_folds(results: list) -> dict:
    """Extrae métricas agrupadas por número de folds."""
    metrics_by_k = {}

    for entry in results:
        k = entry.get("config", {}).get("n_folds", 5)

        # Usar ensemble_results si existe, sino results
        res = entry.get("ensemble_results", entry.get("results", {}))

        if k not in metrics_by_k:
            metrics_by_k[k] = {
                "plate": {"accuracy": [], "f1": [], "precision": [], "recall": []},
                "electrode": {"accuracy": [], "f1": [], "precision": [], "recall": []},
                "current": {"accuracy": [], "f1": [], "precision": [], "recall": []},
            }

        for task in ["plate", "electrode", "current"]:
            if task in res:
                for metric in ["accuracy", "f1", "precision", "recall"]:
                    if metric in res[task]:
                        metrics_by_k[k][task][metric].append(res[task][metric])

    return metrics_by_k


def plot_metrics_vs_folds(
    metrics_by_k: dict,
    duration: str,
    metric: str = "all",
    save: bool = False,
    output_dir: Path = None,
):
    """Grafica métricas vs número de folds."""

    # Ordenar por k
    k_values = sorted(metrics_by_k.keys())

    if len(k_values) < 2:
        print(f"Advertencia: Solo hay datos para k={k_values}")
        print("Ejecuta entrenar.py con diferentes valores de --k-folds para comparar.")

    tasks = ["plate", "electrode", "current"]
    task_names = {
        "plate": "Plate Thickness",
        "electrode": "Electrode Type",
        "current": "Current Type",
    }
    colors = {"plate": "#2ecc71", "electrode": "#3498db", "current": "#e74c3c"}

    metrics_to_plot = (
        ["accuracy", "f1", "precision", "recall"] if metric == "all" else [metric]
    )

    fig, axes = plt.subplots(
        1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5)
    )

    if len(metrics_to_plot) == 1:
        axes = [axes]

    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]

        for task in tasks:
            # Obtener valores promedio por k
            y_values = []
            y_stds = []

            for k in k_values:
                values = metrics_by_k[k][task][metric_name]
                if values:
                    y_values.append(np.mean(values))
                    y_stds.append(np.std(values) if len(values) > 1 else 0)
                else:
                    y_values.append(np.nan)
                    y_stds.append(0)

            # Graficar línea con error bars si hay múltiples corridas
            ax.errorbar(
                k_values,
                y_values,
                yerr=y_stds if max(y_stds) > 0 else None,
                marker="o",
                label=task_names[task],
                color=colors[task],
                linewidth=2,
                markersize=8,
                capsize=3,
            )

        ax.set_xlabel("Número de Folds (K)", fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f"{metric_name.capitalize()} vs K-Folds", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)

        # Ajustar límites del eje Y
        ax.set_ylim([0.0, 1.05])

    plt.suptitle(f"Métricas vs Número de Folds - {duration}", fontsize=16, y=1.02)
    plt.tight_layout()

    if save:
        if output_dir is None:
            output_dir = ROOT_DIR / duration
        output_path = output_dir / f"metricas_vs_folds_{metric}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Gráfica guardada en: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Grafica métricas vs cantidad de folds para una duración específica"
    )
    parser.add_argument(
        "duration",
        type=str,
        help="Directorio de duración (ej: 5seg, 10seg, 30seg)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["all", "accuracy", "f1", "precision", "recall"],
        help="Métrica a graficar (default: all)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guardar gráfica como imagen",
    )

    args = parser.parse_args()

    # Verificar que el directorio existe
    duration_dir = ROOT_DIR / args.duration
    if not duration_dir.exists():
        print(f"Error: No se encontró el directorio {duration_dir}")
        print(f"Directorios disponibles: 1seg, 2seg, 5seg, 10seg, 20seg, 30seg, 50seg")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Graficando métricas vs folds para: {args.duration}")
    print(f"{'=' * 60}")

    # Cargar y procesar resultados
    results = load_results(args.duration)
    print(f"Cargadas {len(results)} entradas de resultados")

    metrics_by_k = extract_metrics_by_folds(results)
    print(f"Valores de K encontrados: {sorted(metrics_by_k.keys())}")

    # Graficar
    plot_metrics_vs_folds(
        metrics_by_k,
        args.duration,
        metric=args.metric,
        save=args.save,
        output_dir=duration_dir,
    )


if __name__ == "__main__":
    main()
