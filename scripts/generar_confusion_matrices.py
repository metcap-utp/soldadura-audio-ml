#!/usr/bin/env python3
"""
Genera gráficas de matrices de confusión a partir de los resultados de inferencia.

Uso:
    python scripts/generar_confusion_matrices.py              # Todas las duraciones
    python scripts/generar_confusion_matrices.py --duracion 10seg  # Solo 10seg
    python scripts/generar_confusion_matrices.py --ultimo     # Solo el último resultado de cada duración
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent

# Duraciones disponibles
DURACIONES = ["01seg", "02seg", "05seg", "10seg", "20seg", "30seg", "50seg"]

# Nombres legibles para las tareas
TASK_NAMES = {
    "plate_thickness": "Grosor de Placa",
    "electrode": "Tipo de Electrodo",
    "current_type": "Tipo de Corriente",
}

# Colores para cada tarea
TASK_COLORS = {
    "plate_thickness": "Blues",
    "electrode": "Greens",
    "current_type": "Oranges",
}


def cargar_resultados(duracion: str) -> list:
    """Carga los resultados de inferencia de una duración específica."""
    infer_json = ROOT_DIR / duracion / "inferencia.json"

    if not infer_json.exists():
        print(f"  No se encontró {infer_json}")
        return []

    with open(infer_json, "r") as f:
        data = json.load(f)

    # Filtrar solo evaluaciones blind con matrices de confusión
    resultados = [
        r
        for r in data
        if r.get("mode") == "blind_evaluation" and "confusion_matrices" in r
    ]

    return resultados


def generar_grafica_confusion(
    cm: np.ndarray,
    clases: list,
    titulo: str,
    output_path: Path,
    cmap: str = "Blues",
    accuracy: float = None,
    f1_macro: float = None,
):
    """Genera y guarda una gráfica de matriz de confusión."""

    fig, ax = plt.subplots(figsize=(8, 6))

    # Crear heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=clases,
        yticklabels=clases,
        ax=ax,
        cbar_kws={"label": "Cantidad"},
    )

    # Configurar etiquetas
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight="bold")

    # Agregar métricas si están disponibles
    if accuracy is not None or f1_macro is not None:
        metrics_text = []
        if accuracy is not None:
            metrics_text.append(f"Accuracy: {accuracy:.4f}")
        if f1_macro is not None:
            metrics_text.append(f"F1 Macro: {f1_macro:.4f}")

        ax.text(
            0.5,
            -0.12,
            " | ".join(metrics_text),
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            style="italic",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generar_grafica_combinada(
    resultado: dict,
    duracion: str,
    output_path: Path,
):
    """Genera una gráfica combinada con las 3 matrices de confusión."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tasks = ["plate_thickness", "electrode", "current_type"]

    for ax, task in zip(axes, tasks):
        cm = np.array(resultado["confusion_matrices"][task])
        clases = resultado["classes"][task]

        # Simplificar nombres de clases
        clases_cortas = [c.replace("Placa_", "").replace("mm", " mm") for c in clases]

        acc = resultado["accuracy"].get(task, None)
        f1 = resultado.get("macro_f1", {}).get(task, None)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=TASK_COLORS[task],
            xticklabels=clases_cortas,
            yticklabels=clases_cortas,
            ax=ax,
            cbar=False,
        )

        ax.set_xlabel("Predicción", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)

        # Título con métricas
        if acc is not None and f1 is not None:
            titulo = f"{TASK_NAMES[task]}\nAcc: {acc:.3f} | F1: {f1:.3f}"
        elif acc is not None:
            titulo = f"{TASK_NAMES[task]}\nAcc: {acc:.3f}"
        else:
            titulo = TASK_NAMES[task]
        ax.set_title(titulo, fontsize=11, fontweight="bold")

    # Título general
    config = resultado.get("config", {})
    segment_dur = resultado.get("segment_duration", duracion)
    n_samples = resultado.get("n_samples", "?")
    k_folds = config.get("k_folds", resultado.get("n_models", "?"))
    train_seconds = config.get("train_seconds", "?")
    test_seconds = config.get("test_seconds", segment_dur)
    overlap_seconds = config.get("overlap_seconds", None)

    overlap_text = ""
    if overlap_seconds is not None and test_seconds not in (None, "?"):
        try:
            overlap_ratio = float(overlap_seconds) / float(test_seconds)
            overlap_text = (
                f" | Solapamiento: {overlap_seconds}s ({overlap_ratio * 100:.0f}%)"
            )
        except (ValueError, ZeroDivisionError):
            overlap_text = f" | Solapamiento: {overlap_seconds}s"

    # Métricas globales si existen
    global_metrics = resultado.get("global_metrics", {})
    exact_match = global_metrics.get("exact_match_accuracy", None)
    hamming = global_metrics.get("hamming_accuracy", None)

    title_parts = [
        f"Matrices de Confusión - Audio de {test_seconds} segundos",
        f"K={k_folds} ({n_samples} muestras)",
    ]
    if exact_match is not None and hamming is not None:
        title_parts.append(f"Exact Match: {exact_match:.3f} | Hamming: {hamming:.3f}")

    fig.suptitle("\n".join(title_parts), fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def timestamp_to_filename(timestamp: str) -> str:
    """Convierte un timestamp ISO a un nombre de archivo legible."""
    # "2026-01-21T22:17:34.283909" -> "2026-01-21_22-17-34"
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d_%H-%M-%S")
    except (ValueError, TypeError):
        return "unknown"


def procesar_duracion(duracion: str, solo_ultimo: bool = False):
    """Procesa una duración y genera sus gráficas."""
    print(f"\nProcesando {duracion}...")

    resultados = cargar_resultados(duracion)

    if not resultados:
        print(f"  No hay resultados de blind para {duracion}")
        return

    # Crear carpeta de salida
    output_dir = ROOT_DIR / duracion / "matrices_confusion"
    output_dir.mkdir(exist_ok=True)

    # Si solo_ultimo, procesar solo el último resultado
    if solo_ultimo:
        resultados = [resultados[-1]]

    for i, resultado in enumerate(resultados):
        timestamp = resultado.get("timestamp", f"result_{i}")
        k_folds = resultado.get("config", {}).get(
            "k_folds", resultado.get("n_models", 5)
        )
        filename_base = f"k{k_folds}_{timestamp_to_filename(timestamp)}"

        print(f"  Generando graficas para {filename_base}...")

        # Generar gráfica combinada
        output_combined = output_dir / f"combined_{filename_base}.png"
        generar_grafica_combinada(resultado, duracion, output_combined)
        print(f"    - {output_combined.name}")

        # Generar gráficas individuales
        for task in ["plate_thickness", "electrode", "current_type"]:
            cm = np.array(resultado["confusion_matrices"][task])
            clases = resultado["classes"][task]
            clases_cortas = [
                c.replace("Placa_", "").replace("mm", " mm") for c in clases
            ]

            acc = resultado["accuracy"].get(task, None)
            f1 = resultado.get("macro_f1", {}).get(task, None)

            config = resultado.get("config", {})
            segment_dur = resultado.get("segment_duration", duracion)
            k_folds = config.get("k_folds", resultado.get("n_models", "?"))
            train_seconds = config.get("train_seconds", "?")
            test_seconds = config.get("test_seconds", segment_dur)
            overlap_seconds = config.get("overlap_seconds", None)

            overlap_text = ""
            if overlap_seconds is not None and test_seconds not in (None, "?"):
                try:
                    overlap_ratio = float(overlap_seconds) / float(test_seconds)
                    overlap_text = f", Solapamiento: {overlap_seconds}s ({overlap_ratio * 100:.0f}%)"
                except (ValueError, ZeroDivisionError):
                    overlap_text = f", Solapamiento: {overlap_seconds}s"

            titulo = f"{TASK_NAMES[task]} (Audio: {test_seconds}s, K={k_folds})"

            output_path = output_dir / f"{task}_{filename_base}.png"
            generar_grafica_confusion(
                cm,
                clases_cortas,
                titulo,
                output_path,
                cmap=TASK_COLORS[task],
                accuracy=acc,
                f1_macro=f1,
            )

        print(f"    - 3 graficas individuales")

    print(f"  Guardadas en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera gráficas de matrices de confusión"
    )
    parser.add_argument(
        "--duracion",
        "-d",
        choices=DURACIONES,
        help="Procesar solo una duración específica",
    )
    parser.add_argument(
        "--ultimo",
        "-u",
        action="store_true",
        help="Procesar solo el último resultado de cada duración",
    )
    parser.add_argument(
        "--todas",
        "-a",
        action="store_true",
        help="Procesar todas las duraciones (por defecto)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  GENERADOR DE MATRICES DE CONFUSIÓN")
    print("=" * 60)

    # Determinar qué duraciones procesar
    if args.duracion:
        duraciones = [args.duracion]
    else:
        duraciones = DURACIONES

    for duracion in duraciones:
        procesar_duracion(duracion, solo_ultimo=args.ultimo)

    print("\n" + "=" * 60)
    print("  Proceso completado")
    print("=" * 60)


if __name__ == "__main__":
    main()
