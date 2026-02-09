"""
Grafica métricas vs cantidad de folds para una duración específica.

Genera UNA sola figura con Accuracy y F1 para las 3 etiquetas
(Plate Thickness, Electrode Type, Current Type) con una leyenda compartida.

Los datos se obtienen de inferencia.json (evaluación ciega / blind) cuando están
disponibles, lo que refleja la capacidad de generalización real.

Uso:
    python scripts/graficar_folds.py 05seg              # Grafica para 5 segundos
    python scripts/graficar_folds.py 10seg --save       # Guarda la imagen
    python scripts/graficar_folds.py 05seg --metric f1  # Solo métrica F1
    python scripts/graficar_folds.py 05seg --source val # Usar datos de validación cruzada
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent

TASKS = ["plate", "electrode", "current"]
COLORS = {
    "plate": "#2ecc71",
    "electrode": "#3498db",
    "current": "#e74c3c",
}
METRIC_MARKERS = {
    "accuracy": "o",
    "f1": "s",
}
METRIC_STYLES = {
    "accuracy": "-",
    "f1": "--",
}

# ── i18n ────────────────────────────────────────────────────────────────
I18N = {
    "es": {
        "task_names": {
            "plate": "Espesor de Placa",
            "electrode": "Tipo de Electrodo",
            "current": "Tipo de Corriente",
        },
        "xlabel_folds": "Número de Folds (K)",
        "ylabel_metric": "Valor de Métrica",
        "ylabel_global": "Métrica Global",
        "title_folds": "Métricas vs K-Folds — {duration} ({source})",
        "title_global": "Métricas Globales vs K-Folds — {duration} ({source})",
        "source_blind": "Blind",
        "source_val": "Val. Cruzada",
        "exact_match": "Coincidencia Exacta",
        "hamming": "Hamming Accuracy",
    },
    "en": {
        "task_names": {
            "plate": "Plate Thickness",
            "electrode": "Electrode Type",
            "current": "Current Type",
        },
        "xlabel_folds": "Number of Folds (K)",
        "ylabel_metric": "Metric Value",
        "ylabel_global": "Global Metric",
        "title_folds": "Metrics vs K-Folds — {duration} ({source})",
        "title_global": "Global Metrics vs K-Folds — {duration} ({source})",
        "source_blind": "Blind",
        "source_val": "Cross-Val",
        "exact_match": "Exact Match",
        "hamming": "Hamming Accuracy",
    },
}


# =============================================================================
# Carga de datos
# =============================================================================


def load_blind_metrics(duration_dir: str) -> dict:
    """
    Carga métricas de evaluación ciega (inferencia.json) agrupadas por K.

    Retorna:
        dict[k] -> {
            "plate": {"accuracy": ..., "f1": ...},
            "electrode": {"accuracy": ..., "f1": ...},
            "current": {"accuracy": ..., "f1": ...},
            "exact_match": ...,
            "hamming": ...,
        }
    """
    infer_path = ROOT_DIR / duration_dir / "inferencia.json"
    if not infer_path.exists():
        return {}

    with open(infer_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    # Agrupar por K, tomando la entrada más reciente para cada K
    entries_by_k = {}

    for entry in data:
        if entry.get("mode") != "blind_evaluation":
            continue

        # Determinar K
        config = entry.get("config", {})
        k = config.get("k_folds", entry.get("n_models"))
        if not k:
            continue

        if k not in entries_by_k:
            entries_by_k[k] = []
        entries_by_k[k].append(entry)

    metrics_by_k = {}

    for k, entries in entries_by_k.items():
        # Preferir entradas con id (nuevo formato) sobre las sin id
        with_id = [e for e in entries if "id" in e]
        if with_id:
            entry = sorted(with_id, key=lambda e: e.get("timestamp", ""), reverse=True)[
                0
            ]
        else:
            entry = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[
                0
            ]

        acc = entry.get("accuracy", {})
        f1 = entry.get("macro_f1", {})
        gm = entry.get("global_metrics", {})

        metrics_by_k[k] = {
            "plate": {
                "accuracy": acc.get("plate_thickness", 0),
                "f1": f1.get("plate_thickness", 0),
            },
            "electrode": {
                "accuracy": acc.get("electrode", 0),
                "f1": f1.get("electrode", 0),
            },
            "current": {
                "accuracy": acc.get("current_type", 0),
                "f1": f1.get("current_type", 0),
            },
            "exact_match": gm.get("exact_match_accuracy"),
            "hamming": gm.get("hamming_accuracy"),
        }

    return metrics_by_k


def load_cv_metrics(duration_dir: str) -> dict:
    """
    Carga métricas de validación cruzada (resultados.json) agrupadas por K.

    Calcula el promedio de las métricas de cada fold.

    Retorna:
        dict[k] -> {
            "plate": {"accuracy": ..., "f1": ...},
            "electrode": {"accuracy": ..., "f1": ...},
            "current": {"accuracy": ..., "f1": ...},
        }
    """
    results_path = ROOT_DIR / duration_dir / "resultados.json"
    if not results_path.exists():
        return {}

    with open(results_path, "r") as f:
        results = json.load(f)

    if not isinstance(results, list):
        results = [results]

    entries_by_k = {}

    for entry in results:
        config = entry.get("config", {})
        k = config.get("n_folds", 5)

        if k not in entries_by_k:
            entries_by_k[k] = []
        entries_by_k[k].append(entry)

    metrics_by_k = {}

    for k, entries in entries_by_k.items():
        # Preferir entradas con id (nuevo formato)
        with_id = [e for e in entries if "id" in e]
        if with_id:
            entry = sorted(with_id, key=lambda e: e.get("timestamp", ""), reverse=True)[
                0
            ]
        else:
            entry = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[
                0
            ]

        fold_results = entry.get("fold_results", [])

        if fold_results:
            metrics_by_k[k] = {}
            for task in TASKS:
                acc_key = f"acc_{task}"
                f1_key = f"f1_{task}"

                accs = [fr[acc_key] for fr in fold_results if acc_key in fr]
                f1s = [fr[f1_key] for fr in fold_results if f1_key in fr]

                metrics_by_k[k][task] = {
                    "accuracy": np.mean(accs) if accs else 0,
                    "f1": np.mean(f1s) if f1s else 0,
                }
        else:
            # Fallback: usar ensemble_results
            res = entry.get("ensemble_results", entry.get("results", {}))
            metrics_by_k[k] = {}
            for task in TASKS:
                metrics_by_k[k][task] = {
                    "accuracy": res.get(task, {}).get("accuracy", 0),
                    "f1": res.get(task, {}).get("f1", 0),
                }

    return metrics_by_k


# =============================================================================
# Gráficas
# =============================================================================


def plot_metrics_vs_folds(
    metrics_by_k: dict,
    duration: str,
    metric: str = "all",
    save: bool = False,
    output_dir: Path = None,
    source: str = "blind",
    lang: str = "es",
):
    """
    Grafica métricas vs número de folds.

    metric='all': una sola figura con Accuracy (línea continua) y F1 (línea
    punteada) para las 3 etiquetas. Leyenda compartida.
    """
    L = I18N[lang]
    TASK_NAMES = L["task_names"]
    k_values = sorted(metrics_by_k.keys())

    if len(k_values) < 2:
        print(f"  Advertencia: Solo hay datos para K={k_values}. Se necesitan ≥2.")

    source_label = L["source_blind"] if source == "blind" else L["source_val"]

    if metric == "all":
        # ── Una sola figura con acc + f1 para las 3 tareas ──────────────
        fig, ax = plt.subplots(figsize=(10, 6))

        all_values = []

        for task in TASKS:
            for m_name in ["accuracy", "f1"]:
                y_values = []
                for k in k_values:
                    val = metrics_by_k[k].get(task, {}).get(m_name)
                    y_values.append(val if val is not None else np.nan)

                all_values.extend([v for v in y_values if not np.isnan(v)])

                label = f"{TASK_NAMES[task]} — {m_name.capitalize()}"
                ax.plot(
                    k_values,
                    y_values,
                    marker=METRIC_MARKERS[m_name],
                    linestyle=METRIC_STYLES[m_name],
                    color=COLORS[task],
                    linewidth=2,
                    markersize=6,
                    label=label,
                )

        ax.set_xlabel(L["xlabel_folds"], fontsize=12)
        ax.set_ylabel(L["ylabel_metric"], fontsize=12)
        ax.set_title(
            L["title_folds"].format(duration=duration, source=source_label), fontsize=14
        )
        ax.legend(loc="best", fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)

        _adjust_ylim(ax, all_values)

        plt.tight_layout()

        if save:
            if output_dir is None:
                output_dir = ROOT_DIR / duration / "metricas"
            output_dir.mkdir(exist_ok=True)
            out = output_dir / "metricas_vs_folds.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Guardada: {out}")
        plt.close(fig)

    else:
        # ── Una métrica específica ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))

        all_values = []
        for task in TASKS:
            y_values = []
            for k in k_values:
                val = metrics_by_k[k].get(task, {}).get(metric)
                y_values.append(val if val is not None else np.nan)

            all_values.extend([v for v in y_values if not np.isnan(v)])

            ax.plot(
                k_values,
                y_values,
                marker="o",
                color=COLORS[task],
                linewidth=2,
                markersize=6,
                label=TASK_NAMES[task],
            )

        ax.set_xlabel(L["xlabel_folds"], fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(
            f"{metric.capitalize()} vs K-Folds — {duration} ({source_label})",
            fontsize=14,
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)

        _adjust_ylim(ax, all_values)

        plt.tight_layout()

        if save:
            if output_dir is None:
                output_dir = ROOT_DIR / duration / "metricas"
            output_dir.mkdir(exist_ok=True)
            out = output_dir / f"metricas_vs_folds_{metric}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Guardada: {out}")
        plt.close(fig)

    # Gráfica adicional: métricas globales (Exact Match y Hamming)
    if source == "blind":
        plot_global_vs_folds(
            metrics_by_k,
            duration,
            save=save,
            output_dir=output_dir,
            source=source,
            lang=lang,
        )


def plot_global_vs_folds(
    metrics_by_k: dict,
    duration: str,
    save: bool = False,
    output_dir: Path = None,
    source: str = "blind",
    lang: str = "es",
):
    """Grafica Exact Match y Hamming Accuracy vs K (solo blind)."""
    L = I18N[lang]
    k_values = sorted(metrics_by_k.keys())
    exact = [metrics_by_k[k].get("exact_match") for k in k_values]
    hamming = [metrics_by_k[k].get("hamming") for k in k_values]

    # Verificar que haya datos
    if all(v is None for v in exact + hamming):
        return

    source_label = L["source_blind"] if source == "blind" else L["source_val"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        k_values,
        exact,
        marker="o",
        label=L["exact_match"],
        color="#8e44ad",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        k_values,
        hamming,
        marker="s",
        label=L["hamming"],
        color="#16a085",
        linewidth=2,
        markersize=6,
    )

    ax.set_xlabel(L["xlabel_folds"], fontsize=12)
    ax.set_ylabel(L["ylabel_global"], fontsize=12)
    ax.set_title(
        L["title_global"].format(duration=duration, source=source_label), fontsize=14
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    vals = [v for v in exact + hamming if v is not None]
    _adjust_ylim(ax, vals)

    plt.tight_layout()

    if save:
        if output_dir is None:
            output_dir = ROOT_DIR / duration / "metricas"
        output_dir.mkdir(exist_ok=True)
        out = output_dir / "metricas_globales_vs_folds.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Guardada: {out}")
    plt.close(fig)


def _adjust_ylim(ax, values):
    """Ajusta los límites del eje Y según los datos."""
    if values:
        y_min = min(values)
        y_max = max(values)
        span = y_max - y_min
        pad = max(0.02, span * 0.15)
        ax.set_ylim(max(0.0, y_min - pad), min(1.05, y_max + pad))
    else:
        ax.set_ylim(0, 1.05)

    if ax.get_ylim()[1] >= 1.0:
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1, alpha=0.5)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Grafica métricas vs cantidad de folds para una duración"
    )
    parser.add_argument(
        "duration",
        type=str,
        help="Directorio de duración (ej: 05seg, 10seg, 30seg)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["all", "accuracy", "f1"],
        help="Métrica a graficar (default: all = acc + f1 juntas)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="blind",
        choices=["blind", "val"],
        help="Fuente de datos: blind (inferencia.json) o val (resultados.json)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guardar gráfica como imagen",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="es",
        choices=["es", "en"],
        help="Idioma de la gráfica (default: es)",
    )
    args = parser.parse_args()

    duration_dir = ROOT_DIR / args.duration
    if not duration_dir.exists():
        print(f"Error: No se encontró el directorio {duration_dir}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Graficando métricas vs K-Folds: {args.duration}")
    print(
        f"Fuente: {'Evaluación ciega (blind)' if args.source == 'blind' else 'Validación cruzada'}"
    )
    print(f"{'=' * 60}")

    # Cargar datos
    if args.source == "blind":
        metrics_by_k = load_blind_metrics(args.duration)
    else:
        metrics_by_k = load_cv_metrics(args.duration)

    if not metrics_by_k:
        print(f"No se encontraron datos para {args.duration} (fuente: {args.source})")
        sys.exit(1)

    print(f"  Valores de K: {sorted(metrics_by_k.keys())}")

    metrics_dir = duration_dir / "metricas"
    metrics_dir.mkdir(exist_ok=True)

    # Graficar
    plot_metrics_vs_folds(
        metrics_by_k,
        args.duration,
        metric=args.metric,
        save=args.save,
        output_dir=metrics_dir,
        source=args.source,
        lang=args.lang,
    )

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
