"""
Grafica métricas vs duración de clips de audio.

Compara el rendimiento del modelo entre diferentes duraciones de segmento
(01seg, 02seg, 05seg, 10seg, 20seg, 30seg, 50seg).

Uso:
    python graficar_duraciones.py                    # Grafica todas las duraciones
    python graficar_duraciones.py --k-folds 5        # Solo resultados de 5-fold
    python graficar_duraciones.py --save             # Guarda la imagen
    python graficar_duraciones.py --metric accuracy  # Solo accuracy
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent

# Duraciones disponibles
DURATION_DIRS = ["01seg", "02seg", "05seg", "10seg", "20seg", "30seg", "50seg"]

# ── i18n ──────────────────────────────────────────────────────────────
I18N = {
    "es": {
        "task_names": {
            "plate": "Espesor de Placa",
            "electrode": "Tipo de Electrodo",
            "current": "Tipo de Corriente",
        },
        "xlabel_dur": "Duración del Clip (segundos)",
        "title_metric": "{metric} vs Duración",
        "suptitle": "Métricas vs Duración del Clip de Audio",
    },
    "en": {
        "task_names": {
            "plate": "Plate Thickness",
            "electrode": "Electrode Type",
            "current": "Current Type",
        },
        "xlabel_dur": "Clip Duration (seconds)",
        "title_metric": "{metric} vs Duration",
        "suptitle": "Metrics vs Audio Clip Duration",
    },
}


def get_duration_value(dir_name: str) -> float:
    """Extrae el valor numérico de la duración del nombre del directorio."""
    match = re.match(r"(\d+)seg", dir_name)
    if match:
        return float(match.group(1))
    return 0


def load_all_results(k_folds: int = None) -> dict:
    """Carga resultados de todas las duraciones disponibles."""
    results_by_duration = {}

    for duration_dir in DURATION_DIRS:
        results_path = ROOT_DIR / duration_dir / "resultados.json"

        if not results_path.exists():
            continue

        with open(results_path, "r") as f:
            results = json.load(f)

        if not isinstance(results, list):
            results = [results]

        # Filtrar por k_folds si se especifica
        if k_folds is not None:
            results = [
                r for r in results if r.get("config", {}).get("n_folds", 5) == k_folds
            ]

        if results:
            duration_value = get_duration_value(duration_dir)
            results_by_duration[duration_value] = {
                "dir": duration_dir,
                "results": results,
            }

    return results_by_duration


def extract_best_metrics(results_by_duration: dict, k_folds: int = None) -> dict:
    """Extrae las mejores métricas para cada duración."""
    metrics = {
        "durations": [],
        "duration_dirs": [],
        "plate": {"accuracy": [], "f1": [], "precision": [], "recall": []},
        "electrode": {"accuracy": [], "f1": [], "precision": [], "recall": []},
        "current": {"accuracy": [], "f1": [], "precision": [], "recall": []},
    }

    for duration in sorted(results_by_duration.keys()):
        data = results_by_duration[duration]
        results = data["results"]

        # Filtrar por k_folds si se especifica
        if k_folds is not None:
            results = [
                r for r in results if r.get("config", {}).get("n_folds", 5) == k_folds
            ]

        if not results:
            continue

        metrics["durations"].append(duration)
        metrics["duration_dirs"].append(data["dir"])

        # Tomar el mejor resultado (más reciente o mejor accuracy promedio)
        best_result = None
        best_avg_acc = 0

        for entry in results:
            res = entry.get("ensemble_results", entry.get("results", {}))
            avg_acc = np.mean(
                [
                    res.get("plate", {}).get("accuracy", 0),
                    res.get("electrode", {}).get("accuracy", 0),
                    res.get("current", {}).get("accuracy", 0),
                ]
            )
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_result = res

        if best_result:
            for task in ["plate", "electrode", "current"]:
                for metric in ["accuracy", "f1", "precision", "recall"]:
                    value = best_result.get(task, {}).get(metric, np.nan)
                    metrics[task][metric].append(value)
        else:
            for task in ["plate", "electrode", "current"]:
                for metric in ["accuracy", "f1", "precision", "recall"]:
                    metrics[task][metric].append(np.nan)

    return metrics


def plot_metrics_vs_duration(
    metrics: dict,
    metric: str = "all",
    k_folds: int = None,
    save: bool = False,
    lang: str = "es",
):
    """Grafica métricas vs duración de clips."""

    durations = metrics["durations"]

    if len(durations) == 0:
        print("Error: No hay datos para graficar.")
        print("Ejecuta entrenar.py en al menos una carpeta de duración.")
        sys.exit(1)

    if len(durations) < 2:
        print(f"Advertencia: Solo hay datos para {durations[0]}seg")
        print("Ejecuta entrenar.py en más carpetas para comparar duraciones.")

    L = I18N[lang]
    tasks = ["plate", "electrode", "current"]
    task_names = L["task_names"]
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
            y_values = metrics[task][metric_name]

            ax.plot(
                durations,
                y_values,
                marker="o",
                label=task_names[task],
                color=colors[task],
                linewidth=2,
                markersize=8,
            )

            # Agregar anotaciones encima de cada punto
            for x, y in zip(durations, y_values):
                # Offset vertical para evitar solapamiento
                offset = 0.02
                ax.annotate(
                    f"{y:.2f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7,
                    color=colors[task],
                    fontweight="bold",
                )

        ax.set_xlabel(L["xlabel_dur"], fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(
            L["title_metric"].format(metric=metric_name.capitalize()), fontsize=14
        )
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(durations)

        # Ajustar límites del eje Y
        ax.set_ylim([0.0, 1.12])

    title = L["suptitle"]
    if k_folds:
        title += f" (K={k_folds})"

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save:
        suffix = f"_k{k_folds}" if k_folds else ""
        filename = f"metricas_vs_duracion{suffix}_{metric}.png"
        duration_dirs = metrics.get("duration_dirs", [])
        if not duration_dirs:
            output_path = ROOT_DIR / filename
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Gráfica guardada en: {output_path}")
        else:
            for duration_dir in duration_dirs:
                output_dir = ROOT_DIR / duration_dir / "metricas"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / filename
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"Gráfica guardada en: {output_path}")

    plt.show()


def print_summary_table(metrics: dict, k_folds: int = None):
    """Imprime tablas resumen de métricas: por tarea y globales."""

    durations = metrics["durations"]

    # ============= MÉTRICAS GLOBALES =============
    print(f"\n{'=' * 90}")
    title = "MÉTRICAS GLOBALES POR DURACIÓN"
    if k_folds:
        title += f" (K={k_folds})"
    print(title)
    print(f"{'=' * 90}")

    # Header
    header = f"{'Duración':>10} | {'Plate Acc':>10} | {'Electrode Acc':>13} | {'Current Acc':>11} | {'Promedio':>10}"
    print(header)
    print("-" * 90)

    avg_accs = []
    for i, duration in enumerate(durations):
        plate_acc = metrics["plate"]["accuracy"][i]
        electrode_acc = metrics["electrode"]["accuracy"][i]
        current_acc = metrics["current"]["accuracy"][i]
        avg_acc = np.mean([plate_acc, electrode_acc, current_acc])
        avg_accs.append(avg_acc)

        row = f"{duration:>8}s | {plate_acc:>10.4f} | {electrode_acc:>13.4f} | {current_acc:>11.4f} | {avg_acc:>10.4f}"
        print(row)

    print("-" * 90)
    best_idx = np.argmax(avg_accs)
    print(
        f"Mejor duración: {durations[best_idx]}s (accuracy promedio: {avg_accs[best_idx]:.4f})"
    )

    # ============= MÉTRICAS POR TAREA =============
    tasks = ["plate", "electrode", "current"]
    task_names = {
        "plate": "PLATE THICKNESS (Espesor de Placa)",
        "electrode": "ELECTRODE TYPE (Tipo de Electrodo)",
        "current": "CURRENT TYPE (Tipo de Corriente)",
    }

    for task in tasks:
        print(f"\n{'=' * 90}")
        title = f"MÉTRICAS: {task_names[task]}"
        if k_folds:
            title += f" (K={k_folds})"
        print(title)
        print(f"{'=' * 90}")

        header = f"{'Duración':>10} | {'Accuracy':>10} | {'F1-Score':>10} | {'Precision':>10} | {'Recall':>10}"
        print(header)
        print("-" * 90)

        for i, duration in enumerate(durations):
            acc = metrics[task]["accuracy"][i]
            f1 = metrics[task]["f1"][i]
            prec = metrics[task]["precision"][i]
            rec = metrics[task]["recall"][i]

            row = f"{duration:>8}s | {acc:>10.4f} | {f1:>10.4f} | {prec:>10.4f} | {rec:>10.4f}"
            print(row)

        print("-" * 90)

        # Mejor duración para esta tarea
        task_accs = metrics[task]["accuracy"]
        best_task_idx = np.argmax(task_accs)
        print(
            f"Mejor duración para {task}: {durations[best_task_idx]}s (accuracy: {task_accs[best_task_idx]:.4f})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Grafica métricas vs duración de clips de audio"
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=None,
        help="Filtrar por número de folds (default: mejor de cada duración)",
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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Solo mostrar tabla, sin gráfica",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="es",
        choices=["es", "en"],
        help="Idioma de las gráficas (default: es)",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("Comparando métricas entre diferentes duraciones de clip")
    print(f"{'=' * 60}")

    # Cargar resultados de todas las duraciones
    results_by_duration = load_all_results(k_folds=args.k_folds)

    if not results_by_duration:
        print("Error: No se encontraron resultados en ninguna carpeta.")
        print("Ejecuta primero: cd Xseg && python entrenar.py")
        sys.exit(1)

    print(f"Duraciones encontradas: {sorted(results_by_duration.keys())}s")

    # Extraer métricas
    metrics = extract_best_metrics(results_by_duration, k_folds=args.k_folds)

    # Mostrar tabla resumen
    print_summary_table(metrics, k_folds=args.k_folds)

    # Graficar
    if not args.no_plot:
        plot_metrics_vs_duration(
            metrics,
            metric=args.metric,
            k_folds=args.k_folds,
            save=args.save,
            lang=args.lang,
        )


if __name__ == "__main__":
    main()
