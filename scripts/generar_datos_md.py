"""
Genera documento DATOS.md con estadísticas de todas las duraciones.

Este script lee los data_stats.json de cada carpeta de duración
y genera un único documento Markdown en el directorio base.

Uso:
    python scripts/generar_datos_md.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

# Directorios
ROOT_DIR = Path(__file__).parent.parent
DURACIONES = ["01seg", "02seg", "05seg", "10seg", "20seg", "30seg", "50seg"]


def load_stats_from_duration(duration: str) -> dict | None:
    """Carga estadísticas de una duración específica."""
    stats_path = ROOT_DIR / duration / "data_stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def generate_global_document(all_stats: dict) -> str:
    """Genera documento Markdown con estadísticas de todas las duraciones."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = f"""# Estadísticas del Dataset SMAW

**Generado:** {timestamp}

Este documento contiene las estadísticas de sesiones y segmentos por etiqueta para todas las duraciones de segmento disponibles.

---

## Resumen General

"""

    # Tabla resumen de todas las duraciones
    doc += "| Duración | Sesiones | Segmentos | Train | Test | Blind |\n"
    doc += "|----------|----------|-----------|-------|------|--------|\n"

    for duration, stats in sorted(
        all_stats.items(), key=lambda x: float(x[0].replace("seg", ""))
    ):
        if stats is None:
            doc += f"| {duration} | - | - | - | - | - |\n"
            continue

        train_segs = stats.get("splits", {}).get("train", {}).get("segments", 0)
        test_segs = stats.get("splits", {}).get("test", {}).get("segments", 0)
        blind_segs = stats.get("splits", {}).get("blind", {}).get("segments", 0)

        doc += f"| {duration} | {stats['totals']['sessions']} | {stats['totals']['segments']} | {train_segs} | {test_segs} | {blind_segs} |\n"

    doc += """
---

## Sesiones por Etiqueta (Globales)

Las sesiones son las mismas para todas las duraciones, solo cambia el número de segmentos.

"""

    # Usar las estadísticas de cualquier duración disponible para sesiones
    sample_stats = next((s for s in all_stats.values() if s is not None), None)

    if sample_stats:
        doc += "### Espesor de Placa (Plate Thickness)\n\n"
        doc += "| Etiqueta | Sesiones |\n"
        doc += "|----------|----------|\n"
        for label, count in sorted(
            sample_stats["sessions_by_label"]["Plate Thickness"].items()
        ):
            doc += f"| {label} | {count} |\n"

        doc += "\n### Tipo de Electrodo (Electrode)\n\n"
        doc += "| Etiqueta | Sesiones |\n"
        doc += "|----------|----------|\n"
        for label, count in sorted(
            sample_stats["sessions_by_label"]["Electrode"].items()
        ):
            doc += f"| {label} | {count} |\n"

        doc += "\n### Tipo de Corriente (Type of Current)\n\n"
        doc += "| Etiqueta | Sesiones |\n"
        doc += "|----------|----------|\n"
        for label, count in sorted(
            sample_stats["sessions_by_label"]["Type of Current"].items()
        ):
            doc += f"| {label} | {count} |\n"

    doc += """
---

## Segmentos por Duración y Etiqueta

"""

    for duration in sorted(all_stats.keys(), key=lambda x: float(x.replace("seg", ""))):
        stats = all_stats[duration]
        if stats is None:
            doc += f"### {duration}\n\n*No hay datos disponibles. Ejecutar `python {duration}/generar_splits.py`*\n\n"
            continue

        doc += f"### {duration}\n\n"
        doc += f"**Total:** {stats['totals']['segments']} segmentos\n\n"

        # Tabla de segmentos por etiqueta
        doc += "#### Por Espesor de Placa\n\n"
        doc += "| Etiqueta | Total | Train | Test | Blind |\n"
        doc += "|----------|-------|-------|------|--------|\n"

        for label in sorted(stats["segments_by_label"]["Plate Thickness"].keys()):
            total = stats["segments_by_label"]["Plate Thickness"].get(label, 0)
            train = (
                stats["splits"]
                .get("train", {})
                .get("segments_by_label", {})
                .get("Plate Thickness", {})
                .get(label, 0)
            )
            test = (
                stats["splits"]
                .get("test", {})
                .get("segments_by_label", {})
                .get("Plate Thickness", {})
                .get(label, 0)
            )
            blind = (
                stats["splits"]
                .get("blind", {})
                .get("segments_by_label", {})
                .get("Plate Thickness", {})
                .get(label, 0)
            )
            doc += f"| {label} | {total} | {train} | {test} | {blind} |\n"

        doc += "\n#### Por Tipo de Electrodo\n\n"
        doc += "| Etiqueta | Total | Train | Test | Blind |\n"
        doc += "|----------|-------|-------|------|--------|\n"

        for label in sorted(stats["segments_by_label"]["Electrode"].keys()):
            total = stats["segments_by_label"]["Electrode"].get(label, 0)
            train = (
                stats["splits"]
                .get("train", {})
                .get("segments_by_label", {})
                .get("Electrode", {})
                .get(label, 0)
            )
            test = (
                stats["splits"]
                .get("test", {})
                .get("segments_by_label", {})
                .get("Electrode", {})
                .get(label, 0)
            )
            blind = (
                stats["splits"]
                .get("blind", {})
                .get("segments_by_label", {})
                .get("Electrode", {})
                .get(label, 0)
            )
            doc += f"| {label} | {total} | {train} | {test} | {blind} |\n"

        doc += "\n#### Por Tipo de Corriente\n\n"
        doc += "| Etiqueta | Total | Train | Test | Blind |\n"
        doc += "|----------|-------|-------|------|--------|\n"

        for label in sorted(stats["segments_by_label"]["Type of Current"].keys()):
            total = stats["segments_by_label"]["Type of Current"].get(label, 0)
            train = (
                stats["splits"]
                .get("train", {})
                .get("segments_by_label", {})
                .get("Type of Current", {})
                .get(label, 0)
            )
            test = (
                stats["splits"]
                .get("test", {})
                .get("segments_by_label", {})
                .get("Type of Current", {})
                .get(label, 0)
            )
            blind = (
                stats["splits"]
                .get("blind", {})
                .get("segments_by_label", {})
                .get("Type of Current", {})
                .get(label, 0)
            )
            doc += f"| {label} | {total} | {train} | {test} | {blind} |\n"

        doc += "\n"

    doc += """---

## Notas

- Las **sesiones** representan grabaciones únicas de soldadura
- Los **segmentos** se generan on-the-fly dividiendo cada grabación según la duración especificada
- El split estratificado garantiza proporciones similares de etiquetas en cada conjunto
- **Blind** es el conjunto de validación final (nunca usado durante desarrollo)
- Los datos de cada duración se generan ejecutando `python Xseg/generar_splits.py`
"""

    return doc


def main():
    """Genera el documento DATOS.md en el directorio base."""
    start_time = time.time()

    print("=" * 70)
    print("GENERANDO DOCUMENTO DATOS.md GLOBAL")
    print("=" * 70)

    # Cargar estadísticas de todas las duraciones
    all_stats = {}
    for duration in DURACIONES:
        print(f"  Cargando {duration}...", end=" ")
        stats = load_stats_from_duration(duration)
        if stats:
            print(f"OK ({stats['totals']['segments']} segmentos)")
        else:
            print("No encontrado")
        all_stats[duration] = stats

    # Generar documento
    doc = generate_global_document(all_stats)

    # Guardar
    output_path = ROOT_DIR / "DATOS.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc)

    elapsed_time = time.time() - start_time

    print(f"\nDocumento generado: {output_path}")
    print(f"Tiempo de ejecución: {elapsed_time:.2f}s")

    # También guardar JSON consolidado
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "duraciones": all_stats,
        "execution_time": {
            "seconds": round(elapsed_time, 2),
        },
    }

    json_path = ROOT_DIR / "data_stats_global.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"JSON consolidado: {json_path}")


if __name__ == "__main__":
    main()
