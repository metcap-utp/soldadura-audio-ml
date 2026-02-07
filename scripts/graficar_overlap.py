"""
Grafica métricas vs overlap ratio para comparar el efecto del solapamiento.

Genera gráficas que comparan el rendimiento del modelo con diferentes
niveles de overlap (0.0, 0.25, 0.5, 0.75) para cada duración de segmento.

Uso:
    python graficar_overlap.py                          # Todas las duraciones, K=5
    python graficar_overlap.py --k-folds 10             # Usa modelos de 10-fold
    python graficar_overlap.py --duration 5             # Solo 05seg
    python graficar_overlap.py --save                   # Guarda las imágenes
    python graficar_overlap.py --heatmap                # Genera heatmaps duración×overlap
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

# Duraciones y overlaps disponibles
DURATIONS = [1, 2, 5, 10, 20, 30, 50]
OVERLAPS = [0.0, 0.25, 0.5, 0.75]

TASK_LABELS = {
    "plate_thickness": "Plate Thickness",
    "electrode": "Electrode",
    "current_type": "Current Type",
}

# ── i18n ──────────────────────────────────────────────────────────────
I18N = {
    "es": {
        "task_names": {
            "plate": "Espesor de Placa",
            "electrode": "Tipo de Electrodo",
            "current": "Tipo de Corriente",
        },
        "xlabel_overlap": "Overlap Ratio",
        "ylabel_metric": "Valor de Métrica",
        "title_overlap": "Métricas vs Overlap — {dur}seg (K={k}, {source})",
        "title_all": "{task} - {metric} vs Overlap (K={k})",
        "title_heatmap_task": "{task} Accuracy",
        "title_heatmap_avg": "Accuracy Promedio",
        "title_heatmap_sup": "Heatmap: Accuracy por Duración × Overlap (K={k})",
        "ylabel_dur": "Duración del Segmento",
        "title_segments": "Segmentos vs Overlap Ratio (estimado)",
        "ylabel_segments": "Segmentos Estimados",
        "source_blind": "Blind",
        "source_val": "Val. Cruzada",
    },
    "en": {
        "task_names": {
            "plate": "Plate Thickness",
            "electrode": "Electrode Type",
            "current": "Current Type",
        },
        "xlabel_overlap": "Overlap Ratio",
        "ylabel_metric": "Metric Value",
        "title_overlap": "Metrics vs Overlap — {dur}s (K={k}, {source})",
        "title_all": "{task} - {metric} vs Overlap (K={k})",
        "title_heatmap_task": "{task} Accuracy",
        "title_heatmap_avg": "Average Accuracy",
        "title_heatmap_sup": "Heatmap: Accuracy by Duration × Overlap (K={k})",
        "ylabel_dur": "Segment Duration",
        "title_segments": "Segments vs Overlap Ratio (estimated)",
        "ylabel_segments": "Estimated Segments",
        "source_blind": "Blind",
        "source_val": "Cross-Val",
    },
}

COLORS = {
    0.0: "#2196F3",
    0.25: "#4CAF50",
    0.5: "#FF9800",
    0.75: "#F44336",
}

MARKERS = {
    0.0: "o",
    0.25: "s",
    0.5: "D",
    0.75: "^",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Grafica métricas vs overlap ratio")
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Número de folds a buscar (default: 5)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        choices=DURATIONS,
        help="Duración específica a graficar (default: todas)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guardar imágenes en vez de mostrar",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Generar heatmaps duración × overlap",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="both",
        choices=["accuracy", "f1", "both"],
        help="Métrica a graficar (default: both)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="es",
        choices=["es", "en"],
        help="Idioma de las gráficas (default: es)",
    )
    return parser.parse_args()


def load_results_for_overlap(duration: int, k_folds: int) -> dict:
    """
    Carga resultados de resultados.json para una duración dada,
    buscando modelos con diferentes overlap ratios.

    Retorna: dict[overlap_ratio] -> resultado
    """
    results_path = ROOT_DIR / f"{duration}seg" / "resultados.json"

    if not results_path.exists():
        return {}

    with open(results_path, "r") as f:
        results = json.load(f)

    if not isinstance(results, list):
        results = [results]

    overlap_results = {}

    for entry in results:
        config = entry.get("config", {})
        entry_k = config.get("n_folds", 5)
        entry_overlap = config.get("overlap_ratio", 0.0)

        # Compatibilidad: si no tiene overlap_ratio, inferir de overlap_seconds
        if "overlap_ratio" not in config:
            overlap_sec = config.get("overlap_seconds", 0.0)
            seg_dur = config.get("segment_duration", float(duration))
            entry_overlap = round(overlap_sec / seg_dur, 2) if seg_dur > 0 else 0.0

        if entry_k != k_folds:
            continue

        # Tomar el mejor resultado por overlap (mayor hamming accuracy)
        if entry_overlap not in overlap_results:
            overlap_results[entry_overlap] = entry
        else:
            existing = overlap_results[entry_overlap]
            existing_acc = _avg_accuracy(existing)
            new_acc = _avg_accuracy(entry)
            if new_acc > existing_acc:
                overlap_results[entry_overlap] = entry

    return overlap_results


def load_infer_for_overlap(duration: int, k_folds: int) -> dict:
    """
    Carga resultados de inferencia.json (blind evaluation) para una duración dada.

    Retorna: dict[overlap_ratio] -> resultado
    """
    infer_path = ROOT_DIR / f"{duration}seg" / "inferencia.json"

    if not infer_path.exists():
        return {}

    with open(infer_path, "r") as f:
        results = json.load(f)

    if not isinstance(results, list):
        results = [results]

    overlap_results = {}

    for entry in results:
        if entry.get("mode") != "blind_evaluation":
            continue

        config = entry.get("config", {})
        entry_k = config.get("k_folds", entry.get("n_models", 5))
        entry_overlap = config.get("overlap_ratio", 0.0)

        if "overlap_ratio" not in config:
            overlap_sec = config.get("overlap_seconds", 0.0)
            seg_dur = config.get("test_seconds", float(duration))
            entry_overlap = round(overlap_sec / seg_dur, 2) if seg_dur > 0 else 0.0

        if entry_k != k_folds:
            continue

        if entry_overlap not in overlap_results:
            overlap_results[entry_overlap] = entry
        else:
            existing = overlap_results[entry_overlap]
            existing_acc = _avg_accuracy_infer(existing)
            new_acc = _avg_accuracy_infer(entry)
            if new_acc > existing_acc:
                overlap_results[entry_overlap] = entry

    return overlap_results


def _avg_accuracy(entry: dict) -> float:
    """Calcula accuracy promedio de un resultado de entrenamiento."""
    val_metrics = entry.get("best_val_metrics", {})
    accs = []
    for task in ["plate_thickness", "electrode", "current_type"]:
        acc = val_metrics.get(f"acc_{task}", val_metrics.get(f"{task}_accuracy", 0.0))
        accs.append(acc)
    return np.mean(accs) if accs else 0.0


def _avg_accuracy_infer(entry: dict) -> float:
    """Calcula accuracy promedio de un resultado de inferencia."""
    acc = entry.get("accuracy", {})
    values = [
        acc.get("plate_thickness", 0),
        acc.get("electrode", 0),
        acc.get("current_type", 0),
    ]
    return np.mean(values)


def extract_metrics_from_results(entry: dict) -> dict:
    """Extrae métricas de un resultado de resultados.json (entrenamiento)."""
    val = entry.get("best_val_metrics", {})
    return {
        "plate_accuracy": val.get("acc_plate_thickness", 0),
        "plate_f1": val.get("f1_plate_thickness", 0),
        "electrode_accuracy": val.get("acc_electrode", 0),
        "electrode_f1": val.get("f1_electrode", 0),
        "current_accuracy": val.get("acc_current_type", 0),
        "current_f1": val.get("f1_current_type", 0),
    }


def extract_metrics_from_infer(entry: dict) -> dict:
    """Extrae métricas de un resultado de inferencia.json (blind)."""
    acc = entry.get("accuracy", {})
    f1 = entry.get("macro_f1", {})
    return {
        "plate_accuracy": acc.get("plate_thickness", 0),
        "plate_f1": f1.get("plate_thickness", 0),
        "electrode_accuracy": acc.get("electrode", 0),
        "electrode_f1": f1.get("electrode", 0),
        "current_accuracy": acc.get("current_type", 0),
        "current_f1": f1.get("current_type", 0),
    }


# =============================================================================
# Gráficas
# =============================================================================


def plot_overlap_comparison(
    duration: int, k_folds: int, metric: str, save: bool, lang: str = "es"
):
    """
    Grafica métricas vs overlap ratio para una duración específica.

    Genera UNA sola figura con Accuracy (línea continua) y F1 (línea
    punteada) para las 3 etiquetas con leyenda compartida.
    Usa datos de inferencia.json (blind) si disponible, sino resultados.json (val).
    """
    # Intentar cargar datos de blind primero
    infer_results = load_infer_for_overlap(duration, k_folds)
    train_results = load_results_for_overlap(duration, k_folds)

    # Priorizar infer (blind) sobre results (val)
    if infer_results:
        overlap_data = {
            ov: extract_metrics_from_infer(r) for ov, r in infer_results.items()
        }
        source = "blind"
    elif train_results:
        overlap_data = {
            ov: extract_metrics_from_results(r) for ov, r in train_results.items()
        }
        source = "val"
    else:
        print(f"  No hay datos para {duration}seg con K={k_folds}")
        return

    overlaps = sorted(overlap_data.keys())

    if len(overlaps) < 2:
        print(
            f"  Solo hay {len(overlaps)} nivel(es) de overlap para {duration}seg K={k_folds}. Se necesitan al menos 2."
        )
        return

    L = I18N[lang]
    tasks = ["plate", "electrode", "current"]
    task_names = L["task_names"]
    task_colors = {
        "plate": "#2ecc71",
        "electrode": "#3498db",
        "current": "#e74c3c",
    }
    metric_markers = {"accuracy": "o", "f1": "s"}
    metric_styles = {"accuracy": "-", "f1": "--"}

    source_label = L["source_blind"] if source == "blind" else L["source_val"]

    show_acc = metric in ("accuracy", "both")
    show_f1 = metric in ("f1", "both")

    fig, ax = plt.subplots(figsize=(10, 6))
    all_values = []

    for task in tasks:
        metrics_to_plot = []
        if show_acc:
            metrics_to_plot.append("accuracy")
        if show_f1:
            metrics_to_plot.append("f1")

        for m_name in metrics_to_plot:
            key = f"{task}_{m_name}"
            values = [overlap_data[ov].get(key, 0) for ov in overlaps]
            all_values.extend(values)

            label = f"{task_names[task]} — {m_name.capitalize()}"
            ax.plot(
                overlaps,
                values,
                marker=metric_markers[m_name],
                linestyle=metric_styles[m_name],
                color=task_colors[task],
                linewidth=2,
                markersize=6,
                label=label,
            )

    ax.set_xlabel(L["xlabel_overlap"], fontsize=12)
    ax.set_ylabel(L["ylabel_metric"], fontsize=12)
    ax.set_title(
        L["title_overlap"].format(dur=duration, k=k_folds, source=source_label),
        fontsize=14,
    )
    ax.set_xticks(overlaps)
    ax.set_xticklabels([str(ov) for ov in overlaps])
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Ajustar eje Y según datos
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        span = y_max - y_min
        pad = max(0.02, span * 0.15)
        ax.set_ylim(max(0.0, y_min - pad), min(1.05, y_max + pad))
    else:
        ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save:
        out_dir = ROOT_DIR / f"{duration}seg" / "metricas"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"overlap_comparison_k{k_folds:02d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {out_path}")
        plt.close()
    else:
        plt.show()


def plot_overlap_all_durations(k_folds: int, metric: str, save: bool, lang: str = "es"):
    """
    Grafica comparación de overlap para cada tarea, con una línea por duración.
    Una sola figura con 3 subplots (uno por tarea), con Accuracy y F1 juntas.
    """
    tasks = ["plate", "electrode", "current"]
    L = I18N[lang]
    task_names = L["task_names"]
    metric_suffix = "accuracy" if metric == "accuracy" else "f1"
    metric_label = "Accuracy" if metric == "accuracy" else "Macro F1"

    # Recopilar datos
    all_data = {}
    for dur in DURATIONS:
        infer_results = load_infer_for_overlap(dur, k_folds)
        train_results = load_results_for_overlap(dur, k_folds)

        if infer_results:
            all_data[dur] = {
                ov: extract_metrics_from_infer(r) for ov, r in infer_results.items()
            }
        elif train_results:
            all_data[dur] = {
                ov: extract_metrics_from_results(r) for ov, r in train_results.items()
            }

    if not all_data:
        print(f"No hay datos disponibles para K={k_folds}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cmap = plt.cm.viridis
    dur_colors = {
        dur: cmap(i / max(len(DURATIONS) - 1, 1)) for i, dur in enumerate(DURATIONS)
    }

    for task_idx, task in enumerate(tasks):
        ax = axes[task_idx]
        key = f"{task}_{metric_suffix}"

        for dur in DURATIONS:
            if dur not in all_data:
                continue

            data = all_data[dur]
            overlaps = sorted(data.keys())

            if len(overlaps) < 2:
                continue

            values = [data[ov].get(key, 0) for ov in overlaps]
            ax.plot(
                overlaps,
                values,
                marker="o",
                linewidth=2,
                color=dur_colors[dur],
                label=f"{dur}seg",
            )

        ax.set_xlabel(L["xlabel_overlap"], fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(
            L["title_all"].format(
                task=task_names[task], metric=metric_label, k=k_folds
            ),
            fontsize=12,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(OVERLAPS)
        ax.set_xticklabels([str(ov) for ov in OVERLAPS])

    plt.tight_layout()

    if save:
        out_dir = ROOT_DIR / "scripts"
        out_path = out_dir / f"overlap_all_durations_{metric_suffix}_k{k_folds:02d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Guardado: {out_path}")
        plt.close()
    else:
        plt.show()


def plot_heatmap(k_folds: int, save: bool, lang: str = "es"):
    """
    Genera heatmaps de duración × overlap para cada tarea.
    Muestra accuracy promedio (hamming) o por tarea.
    """
    tasks = ["plate", "electrode", "current"]
    L = I18N[lang]
    task_names = L["task_names"]

    # Construir matrices
    matrices_acc = {
        task: np.full((len(DURATIONS), len(OVERLAPS)), np.nan) for task in tasks
    }
    matrices_f1 = {
        task: np.full((len(DURATIONS), len(OVERLAPS)), np.nan) for task in tasks
    }
    matrix_avg = np.full((len(DURATIONS), len(OVERLAPS)), np.nan)

    for i, dur in enumerate(DURATIONS):
        infer_results = load_infer_for_overlap(dur, k_folds)
        train_results = load_results_for_overlap(dur, k_folds)

        if infer_results:
            data = {
                ov: extract_metrics_from_infer(r) for ov, r in infer_results.items()
            }
        elif train_results:
            data = {
                ov: extract_metrics_from_results(r) for ov, r in train_results.items()
            }
        else:
            continue

        for j, ov in enumerate(OVERLAPS):
            if ov not in data:
                continue

            metrics = data[ov]
            for task in tasks:
                matrices_acc[task][i, j] = metrics.get(f"{task}_accuracy", np.nan)
                matrices_f1[task][i, j] = metrics.get(f"{task}_f1", np.nan)

            accs = [metrics.get(f"{t}_accuracy", np.nan) for t in tasks]
            matrix_avg[i, j] = np.nanmean(accs)

    # Plot: 4 heatmaps (3 tareas + promedio)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    duration_labels = [f"{d}s" for d in DURATIONS]
    overlap_labels = [str(o) for o in OVERLAPS]

    all_matrices = list(matrices_acc.values()) + [matrix_avg]
    all_titles = [L["title_heatmap_task"].format(task=task_names[t]) for t in tasks] + [
        L["title_heatmap_avg"]
    ]

    for idx, (matrix, title) in enumerate(zip(all_matrices, all_titles)):
        ax = axes[idx // 2][idx % 2]

        # Crear heatmap
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(OVERLAPS)))
        ax.set_xticklabels(overlap_labels)
        ax.set_yticks(range(len(DURATIONS)))
        ax.set_yticklabels(duration_labels)
        ax.set_xlabel(L["xlabel_overlap"])
        ax.set_ylabel(L["ylabel_dur"])
        ax.set_title(f"{title} (K={k_folds})")

        # Anotar valores
        for i in range(len(DURATIONS)):
            for j in range(len(OVERLAPS)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )
                else:
                    ax.text(
                        j, i, "N/A", ha="center", va="center", color="gray", fontsize=8
                    )

        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(
        L["title_heatmap_sup"].format(k=k_folds),
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        out_dir = ROOT_DIR / "scripts"
        out_path = out_dir / f"heatmap_overlap_k{k_folds:02d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Guardado: {out_path}")
        plt.close()
    else:
        plt.show()


def plot_segments_vs_overlap(k_folds: int, save: bool, lang: str = "es"):
    """
    Grafica número de segmentos vs overlap ratio para cada duración.
    Muestra cómo el overlap multiplica los datos disponibles.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    L = I18N[lang]
    cmap = plt.cm.viridis
    dur_colors = {
        dur: cmap(i / max(len(DURATIONS) - 1, 1)) for i, dur in enumerate(DURATIONS)
    }

    for dur in DURATIONS:
        stats_path = ROOT_DIR / f"{dur}seg" / "data_stats.json"
        if not stats_path.exists():
            continue

        with open(stats_path, "r") as f:
            stats = json.load(f)

        total_segments = stats.get("totals", {}).get("segments", 0)
        overlap_ratio = stats.get("overlap_ratio", 0.0)

        # Estimar segmentos para otros overlaps (basado en factor teórico)
        # Con overlap=r y duración=d, hop=d*(1-r), segments ≈ total_audio / hop
        # Factor multiplicativo respecto a overlap=0: 1/(1-r)
        base_segments = (
            total_segments * (1 - overlap_ratio)
            if overlap_ratio < 1
            else total_segments
        )

        segments_by_overlap = {}
        for ov in OVERLAPS:
            factor = 1 / (1 - ov) if ov < 1 else float("inf")
            segments_by_overlap[ov] = int(base_segments * factor)

        overlaps = sorted(segments_by_overlap.keys())
        values = [segments_by_overlap[ov] for ov in overlaps]

        ax.plot(
            overlaps,
            values,
            marker="o",
            linewidth=2,
            color=dur_colors[dur],
            label=f"{dur}seg (real={total_segments} @ {overlap_ratio})",
        )

    ax.set_xlabel(L["xlabel_overlap"], fontsize=12)
    ax.set_ylabel(L["ylabel_segments"], fontsize=12)
    ax.set_title(L["title_segments"], fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(OVERLAPS)
    ax.set_xticklabels([str(ov) for ov in OVERLAPS])

    plt.tight_layout()

    if save:
        out_dir = ROOT_DIR / "scripts"
        out_path = out_dir / "segments_vs_overlap.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Guardado: {out_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    print("=" * 70)
    print("COMPARACIÓN DE OVERLAP")
    print("=" * 70)
    print(f"K-folds: {args.k_folds}")
    print(f"Métrica: {args.metric}")

    lang = args.lang

    if args.heatmap:
        print("\nGenerando heatmaps...")
        plot_heatmap(args.k_folds, args.save, lang)
        plot_segments_vs_overlap(args.k_folds, args.save, lang)
    elif args.duration:
        print(f"\nGraficando overlap para {args.duration}seg...")
        plot_overlap_comparison(
            args.duration, args.k_folds, args.metric, args.save, lang
        )
    else:
        # Graficar comparación por duración individual
        print("\nGraficando overlap por duración individual...")
        for dur in DURATIONS:
            print(f"\n  {dur}seg:")
            plot_overlap_comparison(dur, args.k_folds, args.metric, args.save, lang)

        # Graficar todas las duraciones juntas
        if args.metric == "both":
            for m in ["accuracy", "f1"]:
                print(f"\nGraficando todas las duraciones ({m})...")
                plot_overlap_all_durations(args.k_folds, m, args.save, lang)
        else:
            print(f"\nGraficando todas las duraciones ({args.metric})...")
            plot_overlap_all_durations(args.k_folds, args.metric, args.save, lang)

        # Heatmaps
        print("\nGenerando heatmaps...")
        plot_heatmap(args.k_folds, args.save, lang)

        # Segmentos vs overlap
        print("\nSegmentos vs overlap...")
        plot_segments_vs_overlap(args.k_folds, args.save, lang)


if __name__ == "__main__":
    main()
