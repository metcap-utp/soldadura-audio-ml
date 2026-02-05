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
from matplotlib.ticker import PercentFormatter

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


def load_global_metrics(duration_dir: str) -> dict:
    """Carga métricas globales (Exact Match y Hamming) desde infer.json."""
    infer_path = ROOT_DIR / duration_dir / "infer.json"
    if not infer_path.exists():
        return {}

    with open(infer_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    global_by_k = {}
    for entry in data:
        gm = entry.get("global_metrics") or {}
        if not gm:
            continue
        k = (
            entry.get("k_folds")
            or entry.get("n_models")
            or entry.get("config", {}).get("n_folds")
        )
        if not k:
            continue
        global_by_k[k] = {
            "exact_match": gm.get("exact_match_accuracy"),
            "hamming": gm.get("hamming_accuracy"),
        }

    return global_by_k


def extract_metrics_by_folds(results: list) -> dict:
    """Extrae métricas por número de folds usando la última ejecución por K."""
    metrics_by_k = {}

    for entry in results:
        k = entry.get("config", {}).get("n_folds", 5)

        # Usar ensemble_results si existe, sino results
        res = entry.get("ensemble_results", entry.get("results", {}))

        metrics_by_k[k] = {
            "plate": {},
            "electrode": {},
            "current": {},
        }

        for task in ["plate", "electrode", "current"]:
            if task in res:
                for metric in ["accuracy", "f1", "precision", "recall"]:
                    if metric in res[task]:
                        metrics_by_k[k][task][metric] = res[task][metric]

    return metrics_by_k


def plot_metrics_vs_folds(
    metrics_by_k: dict,
    duration: str,
    metric: str = "all",
    save: bool = False,
    output_dir: Path = None,
    global_by_k: dict | None = None,
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
    colors = {
        "plate": "#2ecc71",
        "electrode": "#3498db",
        "current": "#e74c3c",
        "global": "#2c3e50",
    }

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

        all_values = []

        for task in tasks:
            # Obtener valores promedio por k
            y_values = []

            for k in k_values:
                value = metrics_by_k[k][task].get(metric_name)
                if value is not None:
                    y_values.append(value)
                else:
                    y_values.append(np.nan)

            all_values.extend([v for v in y_values if not np.isnan(v)])

            # Graficar línea simple (sin velas)
            ax.plot(
                k_values,
                y_values,
                marker="o",
                label=task_names[task],
                color=colors[task],
                linewidth=2,
                markersize=6,
            )

        # Métrica global (promedio de las 3 tareas)
        global_values = []
        for k in k_values:
            vals = []
            for task in tasks:
                value = metrics_by_k[k][task].get(metric_name)
                if value is not None:
                    vals.append(value)
            global_values.append(np.mean(vals) if vals else np.nan)

        ax.plot(
            k_values,
            global_values,
            marker="o",
            label="Global (avg)",
            color=colors["global"],
            linewidth=2,
            markersize=6,
            linestyle="--",
        )

        all_values.extend([v for v in global_values if not np.isnan(v)])

        ax.set_xlabel("Número de Folds (K)", fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f"{metric_name.capitalize()} vs K-Folds", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1, alpha=0.6)

        # Ajustar límites del eje Y (zoom dinámico para resaltar diferencias)
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            span = y_max - y_min
            pad = max(0.005, span * 0.2)
            # Mantener 100% cerca del borde superior
            y_upper = 1.0002
            y_lower = max(0.0, y_min - pad)
            # Si el rango es muy pequeño, acercar aún más el techo
            if y_upper - y_lower > 0.15:
                y_lower = max(0.0, 1.0 - 0.15)
            ax.set_ylim([y_lower, y_upper])
        else:
            ax.set_ylim([0.85, 1.0002])

        # Mantener etiquetas hasta 100% y dejar margen mínimo arriba
        tick_min = max(0.0, ax.get_ylim()[0])
        ax.set_yticks(np.linspace(tick_min, 1.0, 5))

    plt.suptitle(f"Métricas vs Número de Folds - {duration}", fontsize=16, y=1.02)
    plt.tight_layout()

    if save:
        if output_dir is None:
            output_dir = ROOT_DIR / duration / "metricas"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"metricas_vs_folds_{metric}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Gráfica guardada en: {output_path}")

    plt.show()

    # Gráfica adicional: métricas globales (Exact Match y Hamming)
    if global_by_k:
        k_global = sorted(global_by_k.keys())
        exact_values = [global_by_k[k].get("exact_match") for k in k_global]
        hamming_values = [global_by_k[k].get("hamming") for k in k_global]

        fig_g, ax_g = plt.subplots(1, 1, figsize=(6, 5))
        ax_g.plot(
            k_global,
            exact_values,
            marker="o",
            label="Exact Match",
            color="#8e44ad",
            linewidth=2,
            markersize=6,
        )
        ax_g.plot(
            k_global,
            hamming_values,
            marker="o",
            label="Hamming",
            color="#16a085",
            linewidth=2,
            markersize=6,
        )

        ax_g.set_xlabel("Número de Folds (K)")
        ax_g.set_ylabel("Métrica Global")
        ax_g.set_title(f"Métricas Globales vs K-Folds - {duration}")
        ax_g.grid(True, alpha=0.3)
        ax_g.legend(loc="best")
        ax_g.set_xticks(k_global)
        ax_g.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax_g.axhline(1.0, color="#666666", linestyle="--", linewidth=1, alpha=0.6)

        # Ajustar límites del eje Y (mismo criterio de margen)
        vals = [v for v in exact_values + hamming_values if v is not None]
        if vals:
            y_min = min(vals)
            y_max = max(vals)
            span = y_max - y_min
            pad = max(0.005, span * 0.2)
            y_upper = 1.0002
            y_lower = max(0.0, y_min - pad)
            if y_upper - y_lower > 0.15:
                y_lower = max(0.0, 1.0 - 0.15)
            ax_g.set_ylim([y_lower, y_upper])
        else:
            ax_g.set_ylim([0.85, 1.0002])

        tick_min = max(0.0, ax_g.get_ylim()[0])
        ax_g.set_yticks(np.linspace(tick_min, 1.0, 5))

        plt.tight_layout()

        if save:
            output_path = output_dir / "metricas_globales_vs_folds.png"
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
    global_by_k = load_global_metrics(args.duration)
    print(f"Valores de K encontrados: {sorted(metrics_by_k.keys())}")

    # Graficar
    metrics_dir = duration_dir / "metricas"
    metrics_dir.mkdir(exist_ok=True)
    plot_metrics_vs_folds(
        metrics_by_k,
        args.duration,
        metric=args.metric,
        save=args.save,
        output_dir=metrics_dir,
        global_by_k=global_by_k,
    )


if __name__ == "__main__":
    main()
