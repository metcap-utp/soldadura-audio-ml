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
DURACIONES = ["1seg", "2seg", "5seg", "10seg", "30seg"]

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
    infer_json = ROOT_DIR / duracion / "infer.json"

    if not infer_json.exists():
        print(f"  ⚠ No se encontró {infer_json}")
        return []

    with open(infer_json, "r") as f:
        data = json.load(f)

    # Filtrar solo evaluaciones holdout con matrices de confusión
    resultados = [
        r
        for r in data
        if r.get("mode") == "holdout_evaluation" and "confusion_matrices" in r
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
            metrics_text.append(f"Accuracy: {accuracy * 100:.2f}%")
        if f1_macro is not None:
            metrics_text.append(f"F1 Macro: {f1_macro * 100:.2f}%")

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
        f1 = resultado["macro_f1"].get(task, None)

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
        titulo = f"{TASK_NAMES[task]}\nAcc: {acc * 100:.1f}% | F1: {f1 * 100:.1f}%"
        ax.set_title(titulo, fontsize=11, fontweight="bold")

    # Título general
    segment_dur = resultado.get("segment_duration", duracion)
    n_samples = resultado.get("n_samples", "?")

    # Métricas globales si existen
    global_metrics = resultado.get("global_metrics", {})
    exact_match = global_metrics.get("exact_match_accuracy", None)
    hamming = global_metrics.get("hamming_accuracy", None)

    title_parts = [f"Matrices de Confusión - {segment_dur}s ({n_samples} muestras)"]
    if exact_match is not None and hamming is not None:
        title_parts.append(
            f"Exact Match: {exact_match * 100:.1f}% | Hamming: {hamming * 100:.1f}%"
        )

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
    print(f"\n📊 Procesando {duracion}...")

    resultados = cargar_resultados(duracion)

    if not resultados:
        print(f"  No hay resultados de holdout para {duracion}")
        return

    # Crear carpeta de salida
    output_dir = ROOT_DIR / duracion / "confusion_matrices"
    output_dir.mkdir(exist_ok=True)

    # Si solo_ultimo, procesar solo el último resultado
    if solo_ultimo:
        resultados = [resultados[-1]]

    for i, resultado in enumerate(resultados):
        timestamp = resultado.get("timestamp", f"result_{i}")
        filename_base = timestamp_to_filename(timestamp)

        print(f"  📈 Generando gráficas para {filename_base}...")

        # Generar gráfica combinada
        output_combined = output_dir / f"combined_{filename_base}.png"
        generar_grafica_combinada(resultado, duracion, output_combined)
        print(f"    ✓ {output_combined.name}")

        # Generar gráficas individuales
        for task in ["plate_thickness", "electrode", "current_type"]:
            cm = np.array(resultado["confusion_matrices"][task])
            clases = resultado["classes"][task]
            clases_cortas = [
                c.replace("Placa_", "").replace("mm", " mm") for c in clases
            ]

            acc = resultado["accuracy"].get(task, None)
            f1 = resultado["macro_f1"].get(task, None)

            segment_dur = resultado.get("segment_duration", duracion)
            titulo = f"{TASK_NAMES[task]} ({segment_dur}s)"

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

        print(f"    ✓ 3 gráficas individuales")

    print(f"  📁 Guardadas en: {output_dir}")


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
    print("  ✅ Proceso completado")
    print("=" * 60)


if __name__ == "__main__":
    main()
