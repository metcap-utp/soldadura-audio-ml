"""
Genera CSVs de splits (train/test/blind) usando split estratificado por sesión.

Este sistema:
1. Agrupa archivos por SESION (carpeta de grabación) para evitar data leakage
2. Estratifica por combinación de etiquetas (plate + electrode + current)
3. Usa semilla fija para reproducibilidad
4. Permite configurar fracciones de train/test/blind
5. SEGMENTA ON-THE-FLY según la duración del directorio (5seg, 10seg, 30seg)
   NO hay archivos segmentados en disco - usa audios del directorio base.

Estructura de archivos (carpeta base audio/):
    audio/Placa_Xmm/EXXXX/{AC,DC}/YYMMDD-HHMMSS_Audio/*.wav

Qué es una SESIÓN?
    Una sesión es una grabación continua de soldadura, identificada por
    su carpeta con timestamp (ej: 240912-143741_Audio). De cada grabación
    se extraen múltiples segmentos. Es importante que todos los segmentos
    de la misma grabación vayan al mismo split para evitar data leakage.

Genera:
    - train.csv (para entrenamiento con K-Fold CV)
    - test.csv (para evaluación durante desarrollo)
    - blind.csv (para validación final en vida real)
    - completo.csv (todos los archivos con columna Split)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Agregar directorio padre para imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils.audio_utils import (
    PROJECT_ROOT,
    discover_sessions,
    get_all_segments_from_session,
    get_script_segment_duration,
)
from utils.timing import timer

# =============================================================================
# CONFIGURACION - Modificar estas variables segun necesidad
# =============================================================================

# Semilla para reproducibilidad (mismo valor = mismo split)
RANDOM_SEED = 42

# Fraccion de datos para blind (0.0 - 1.0)
# Este conjunto NUNCA se usa durante entrenamiento ni desarrollo
# Solo para validacion final en vida real
BLIND_FRACTION = 0.10

# Fraccion de datos para test (0.0 - 1.0, del total original)
TEST_FRACTION = 0.18

# Fraccion de datos para validacion durante entrenamiento (0.0 - 1.0)
# Si es 0.0, no se genera conjunto de validacion (se usa K-Fold CV interno)
VAL_FRACTION = 0.0

# =============================================================================
# CODIGO - No modificar a menos que sea necesario
# =============================================================================

# Duración de segmento basada en el nombre del directorio (5seg -> 5.0)
SEGMENT_DURATION = get_script_segment_duration(Path(__file__))

# Solapamiento entre segmentos (en segundos). Default: 0 (sin solapamiento)
OVERLAP_SECONDS = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generación de splits estratificados por sesión"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Solapamiento entre segmentos en segundos (default: 0.0)",
    )
    return parser.parse_args()


def create_stratification_label(df: pd.DataFrame) -> pd.Series:
    """
    Crea etiqueta combinada para estratificación.
    Combina plate + electrode + current para mantener proporciones.
    """
    return df["Plate Thickness"] + "_" + df["Electrode"] + "_" + df["Type of Current"]


def load_all_sessions() -> pd.DataFrame:
    """Carga todas las sesiones de audio desde el directorio base."""
    print("Descubriendo sesiones de audio...")
    print(f"  Directorio base: {PROJECT_ROOT / 'audio'}")
    print(f"  Duración de segmento: {SEGMENT_DURATION}s")
    print(f"  Solapamiento: {OVERLAP_SECONDS}s")

    sessions_df = discover_sessions()

    if sessions_df.empty:
        print("ERROR: No se encontraron sesiones")
        return sessions_df

    # Contar segmentos por sesión
    print("  Contando segmentos por sesión...")
    segment_counts = []
    for _, row in sessions_df.iterrows():
        segments = get_all_segments_from_session(
            row["Session Path"],
            SEGMENT_DURATION,
            overlap_seconds=OVERLAP_SECONDS,
        )
        segment_counts.append(len(segments))

    sessions_df["Num Segments"] = segment_counts

    print(f"  Sesiones encontradas: {len(sessions_df)}")
    print(f"  Total segmentos: {sum(segment_counts)}")

    return sessions_df


def expand_sessions_to_segments(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expande el DataFrame de sesiones a segmentos individuales.

    Cada segmento hereda las etiquetas de su sesión.
    """
    print("\nExpandiendo sesiones a segmentos...")

    segments_data = []
    for _, row in sessions_df.iterrows():
        segments = get_all_segments_from_session(
            row["Session Path"],
            SEGMENT_DURATION,
            overlap_seconds=OVERLAP_SECONDS,
        )
        for audio_path, seg_idx in segments:
            # Path relativo desde PROJECT_ROOT
            rel_path = audio_path.relative_to(PROJECT_ROOT)
            segments_data.append(
                {
                    "Audio Path": str(rel_path),
                    "Segment Index": seg_idx,
                    "Plate Thickness": row["Plate Thickness"],
                    "Electrode": row["Electrode"],
                    "Type of Current": row["Type of Current"],
                    "Session": row["Session"],
                }
            )

    df = pd.DataFrame(segments_data)
    print(f"  Total segmentos: {len(df)}")

    return df


def split_by_session(
    df: pd.DataFrame,
    blind_frac: float,
    test_frac: float,
    val_frac: float,
    seed: int,
):
    """
    Divide los datos por sesion de forma estratificada.

    El orden de splitting es:
    1. Primero se separa blind (completamente independiente)
    2. Del resto, se separa test
    3. Del resto, se separa val (si aplica)
    4. Lo que queda es train

    Args:
        df: DataFrame con todos los archivos
        blind_frac: Fraccion para blind (0.0-1.0)
        test_frac: Fraccion para test (0.0-1.0)
        val_frac: Fraccion para validacion (0.0-1.0)
        seed: Semilla para reproducibilidad

    Returns:
        DataFrame con columna 'Split' agregada
    """
    # Obtener sesiones unicas con sus etiquetas
    sessions_df = df.groupby("Session").first().reset_index()
    sessions_df["Strat_Label"] = create_stratification_label(sessions_df)

    print(f"\nDistribucion de combinaciones de etiquetas (por sesion):")
    strat_counts = sessions_df["Strat_Label"].value_counts()
    for label, count in strat_counts.items():
        print(f"  {label}: {count} sesiones")

    # Manejar clases con muy pocos ejemplos
    # Para estratificacion, necesitamos al menos 3 ejemplos por clase (blind+test+train)
    min_samples = 3 if blind_frac > 0 else 2
    rare_classes = strat_counts[strat_counts < min_samples].index.tolist()

    if rare_classes:
        print(
            f"\nWarning: Clases con <{min_samples} sesiones (se asignaran a train): {len(rare_classes)} clases"
        )
        sessions_df["is_rare"] = sessions_df["Strat_Label"].isin(rare_classes)
    else:
        sessions_df["is_rare"] = False

    # Separar sesiones raras y no raras
    rare_sessions = sessions_df[sessions_df["is_rare"]]["Session"].tolist()
    normal_sessions_df = sessions_df[~sessions_df["is_rare"]]

    print(f"\nSesiones totales: {len(sessions_df)}")
    print(f"  - Normales (estratificables): {len(normal_sessions_df)}")
    print(f"  - Raras (asignadas a train): {len(rare_sessions)}")

    # Inicializar splits - sesiones raras van a train
    session_splits = {}
    for session in rare_sessions:
        session_splits[session] = "train"

    if len(normal_sessions_df) > 0:
        sessions = normal_sessions_df["Session"].values
        strat_labels = normal_sessions_df["Strat_Label"].values

        # Calcular fracciones ajustadas (considerando sesiones raras ya asignadas)
        total_sessions = len(sessions_df)
        normal_sessions = len(normal_sessions_df)

        # =====================================================================
        # PASO 1: Separar BLIND primero (completamente independiente)
        # =====================================================================
        remaining_sessions = sessions
        remaining_labels = strat_labels

        if blind_frac > 0:
            adjusted_blind_frac = min(
                blind_frac * total_sessions / normal_sessions, 0.4
            )

            try:
                remaining_sessions, blind_sessions, remaining_labels, _ = (
                    train_test_split(
                        remaining_sessions,
                        remaining_labels,
                        test_size=adjusted_blind_frac,
                        random_state=seed,  # Usar semilla base
                        stratify=remaining_labels,
                    )
                )
            except ValueError as e:
                print(
                    f"Warning: No se pudo estratificar blind, usando split aleatorio: {e}"
                )
                remaining_sessions, blind_sessions = train_test_split(
                    remaining_sessions,
                    test_size=adjusted_blind_frac,
                    random_state=seed,
                )
                # Actualizar labels restantes
                remaining_labels = normal_sessions_df[
                    normal_sessions_df["Session"].isin(remaining_sessions)
                ]["Strat_Label"].values

            for session in blind_sessions:
                session_splits[session] = "blind"

            print(f"\n  Blind: {len(blind_sessions)} sesiones separadas")

        # =====================================================================
        # PASO 2: Del resto, separar TEST
        # =====================================================================
        # Recalcular fracciones basado en lo que queda
        remaining_total = len(remaining_sessions)

        # test_frac es del total original, ajustar para el restante
        adjusted_test_frac = (
            min(test_frac * total_sessions / remaining_total, 0.5)
            if remaining_total > 0
            else 0
        )

        adjusted_val_frac = (
            min(val_frac * total_sessions / remaining_total, 0.5)
            if val_frac > 0 and remaining_total > 0
            else 0
        )

        test_val_frac = adjusted_test_frac + adjusted_val_frac

        if test_val_frac > 0 and remaining_total > 0:
            try:
                train_sessions, test_val_sessions, _, test_val_labels = (
                    train_test_split(
                        remaining_sessions,
                        remaining_labels,
                        test_size=test_val_frac,
                        random_state=seed + 1,  # Semilla diferente para este split
                        stratify=remaining_labels,
                    )
                )
            except ValueError as e:
                print(
                    f"Warning: No se pudo estratificar test, usando split aleatorio: {e}"
                )
                train_sessions, test_val_sessions = train_test_split(
                    remaining_sessions,
                    test_size=test_val_frac,
                    random_state=seed + 1,
                )
                test_val_labels = normal_sessions_df[
                    normal_sessions_df["Session"].isin(test_val_sessions)
                ]["Strat_Label"].values

            # Asignar train
            for session in train_sessions:
                session_splits[session] = "train"

            # =====================================================================
            # PASO 3: Split test vs val (si hay validacion)
            # =====================================================================
            if adjusted_val_frac > 0 and len(test_val_sessions) > 1:
                val_ratio = adjusted_val_frac / test_val_frac
                try:
                    test_sessions, val_sessions = train_test_split(
                        test_val_sessions,
                        test_size=val_ratio,
                        random_state=seed + 2,  # Semilla diferente para este split
                        stratify=test_val_labels,
                    )
                except ValueError:
                    test_sessions, val_sessions = train_test_split(
                        test_val_sessions,
                        test_size=val_ratio,
                        random_state=seed + 2,
                    )

                for session in test_sessions:
                    session_splits[session] = "test"
                for session in val_sessions:
                    session_splits[session] = "val"
            else:
                # Sin validacion, todo va a test
                for session in test_val_sessions:
                    session_splits[session] = "test"
        else:
            # Sin test ni val, todo va a train
            for session in remaining_sessions:
                session_splits[session] = "train"

    # Aplicar splits al DataFrame original
    df["Split"] = df["Session"].map(session_splits)

    return df


def save_splits(df: pd.DataFrame):
    """Guarda los CSVs de cada split."""
    splits = df["Split"].unique()

    print("\n" + "=" * 80)
    print("GUARDANDO CSVs")
    print("=" * 80)

    for split_name in sorted(splits):
        split_df = df[df["Split"] == split_name].copy()

        # Columnas a guardar: Audio Path, Segment Index, etiquetas
        save_df = split_df.drop(columns=["Session", "Split"])
        save_df = save_df.sort_values(["Audio Path", "Segment Index"]).reset_index(
            drop=True
        )

        output_path = SCRIPT_DIR / f"{split_name}.csv"
        save_df.to_csv(output_path, index=False)

        print(f"\n{split_name.upper()}: {len(save_df)} segmentos")
        print(f"  Espesores: {split_df['Plate Thickness'].value_counts().to_dict()}")
        print(f"  Electrodos: {split_df['Electrode'].value_counts().to_dict()}")
        print(f"  Corrientes: {split_df['Type of Current'].value_counts().to_dict()}")
        print(f"  Sesiones: {split_df['Session'].nunique()}")
        print(f"  Guardado: {output_path}")

    # Guardar completo.csv con todas las columnas
    print("\n" + "-" * 40)
    complete_df = (
        df.drop(columns=["Session"])
        .sort_values(["Split", "Audio Path", "Segment Index"])
        .reset_index(drop=True)
    )
    output_path = SCRIPT_DIR / "completo.csv"
    complete_df.to_csv(output_path, index=False)
    print(f"COMPLETO: {len(complete_df)} segmentos")
    print(f"  Guardado: {output_path}")


def generate_statistics(df: pd.DataFrame, sessions_df: pd.DataFrame) -> dict:
    """
    Genera estadísticas completas de sesiones y clips por etiqueta.

    Returns:
        dict con todas las estadísticas para guardar en JSON
    """
    stats = {
        "timestamp": datetime.now().isoformat(),
        "segment_duration": SEGMENT_DURATION,
        "overlap_seconds": OVERLAP_SECONDS,
        "random_seed": RANDOM_SEED,
        "config": {
            "blind_fraction": BLIND_FRACTION,
            "test_fraction": TEST_FRACTION,
            "val_fraction": VAL_FRACTION,
        },
        "totals": {
            "sessions": len(sessions_df),
            "segments": len(df),
        },
        "sessions_by_label": {},
        "segments_by_label": {},
        "splits": {},
    }

    # Conteo de sesiones por etiqueta
    for label_col in ["Plate Thickness", "Electrode", "Type of Current"]:
        stats["sessions_by_label"][label_col] = (
            sessions_df[label_col].value_counts().to_dict()
        )
        stats["segments_by_label"][label_col] = df[label_col].value_counts().to_dict()

    # Conteo por split
    for split_name in sorted(df["Split"].unique()):
        split_df = df[df["Split"] == split_name]
        split_sessions = (
            sessions_df[sessions_df["Split"] == split_name]
            if "Split" in sessions_df.columns
            else sessions_df[sessions_df["Session"].isin(split_df["Session"].unique())]
        )

        stats["splits"][split_name] = {
            "sessions": len(split_sessions),
            "segments": len(split_df),
            "sessions_by_label": {},
            "segments_by_label": {},
        }

        for label_col in ["Plate Thickness", "Electrode", "Type of Current"]:
            stats["splits"][split_name]["sessions_by_label"][label_col] = (
                split_sessions[label_col].value_counts().to_dict()
                if len(split_sessions) > 0
                else {}
            )
            stats["splits"][split_name]["segments_by_label"][label_col] = (
                split_df[label_col].value_counts().to_dict()
            )

    return stats


def generate_data_document(stats: dict) -> str:
    """
    Genera documento Markdown con estadísticas de datos.
    """
    doc = f"""# Estadísticas del Dataset - {int(SEGMENT_DURATION)}seg

**Generado:** {stats["timestamp"][:19].replace("T", " ")}

**Configuración:**
- Duración de segmento: {stats["segment_duration"]}s
- Semilla aleatoria: {stats["random_seed"]}
- Blind: {stats["config"]["blind_fraction"]:.1%}
- Test: {stats["config"]["test_fraction"]:.1%}
- Val: {stats["config"]["val_fraction"]:.1%}

---

## Resumen General

| Métrica | Valor |
|---------|-------|
| Total Sesiones | {stats["totals"]["sessions"]} |
| Total Segmentos | {stats["totals"]["segments"]} |

---

## Sesiones por Etiqueta

### Espesor de Placa (Plate Thickness)

| Etiqueta | Sesiones |
|----------|----------|
"""

    for label, count in sorted(stats["sessions_by_label"]["Plate Thickness"].items()):
        doc += f"| {label} | {count} |\n"

    doc += """
### Tipo de Electrodo (Electrode)

| Etiqueta | Sesiones |
|----------|----------|
"""

    for label, count in sorted(stats["sessions_by_label"]["Electrode"].items()):
        doc += f"| {label} | {count} |\n"

    doc += """
### Tipo de Corriente (Type of Current)

| Etiqueta | Sesiones |
|----------|----------|
"""

    for label, count in sorted(stats["sessions_by_label"]["Type of Current"].items()):
        doc += f"| {label} | {count} |\n"

    doc += """
---

## Segmentos por Etiqueta

### Espesor de Placa (Plate Thickness)

| Etiqueta | Segmentos |
|----------|----------|
"""

    for label, count in sorted(stats["segments_by_label"]["Plate Thickness"].items()):
        doc += f"| {label} | {count} |\n"

    doc += """
### Tipo de Electrodo (Electrode)

| Etiqueta | Segmentos |
|----------|----------|
"""

    for label, count in sorted(stats["segments_by_label"]["Electrode"].items()):
        doc += f"| {label} | {count} |\n"

    doc += """
### Tipo de Corriente (Type of Current)

| Etiqueta | Segmentos |
|----------|----------|
"""

    for label, count in sorted(stats["segments_by_label"]["Type of Current"].items()):
        doc += f"| {label} | {count} |\n"

    doc += """
---

## Distribución por Split

"""

    for split_name in ["train", "test", "blind", "val"]:
        if split_name not in stats["splits"]:
            continue

        split_data = stats["splits"][split_name]
        doc += f"""### {split_name.upper()}

- **Sesiones:** {split_data["sessions"]}
- **Segmentos:** {split_data["segments"]}

#### Segmentos por Etiqueta

| Etiqueta | Plate Thickness | Electrode | Current |
|----------|-----------------|-----------|--------|
"""

        # Obtener todas las etiquetas únicas
        plate_labels = sorted(
            split_data["segments_by_label"].get("Plate Thickness", {}).keys()
        )
        electrode_labels = sorted(
            split_data["segments_by_label"].get("Electrode", {}).keys()
        )
        current_labels = sorted(
            split_data["segments_by_label"].get("Type of Current", {}).keys()
        )

        # Crear tabla combinada
        max_rows = max(len(plate_labels), len(electrode_labels), len(current_labels))

        for i in range(max_rows):
            plate = plate_labels[i] if i < len(plate_labels) else ""
            plate_count = (
                split_data["segments_by_label"]
                .get("Plate Thickness", {})
                .get(plate, "")
                if plate
                else ""
            )
            electrode = electrode_labels[i] if i < len(electrode_labels) else ""
            electrode_count = (
                split_data["segments_by_label"].get("Electrode", {}).get(electrode, "")
                if electrode
                else ""
            )
            current = current_labels[i] if i < len(current_labels) else ""
            current_count = (
                split_data["segments_by_label"]
                .get("Type of Current", {})
                .get(current, "")
                if current
                else ""
            )

            plate_str = f"{plate}: {plate_count}" if plate else ""
            electrode_str = f"{electrode}: {electrode_count}" if electrode else ""
            current_str = f"{current}: {current_count}" if current else ""

            doc += f"| | {plate_str} | {electrode_str} | {current_str} |\n"

        doc += "\n"

    doc += """---

## Notas

- Las sesiones se agrupan por grabación original para evitar data leakage
- Los segmentos se generan on-the-fly desde los audios originales
- El split estratificado garantiza proporciones similares de etiquetas en cada conjunto
"""

    return doc


def save_statistics(
    df: pd.DataFrame, sessions_df: pd.DataFrame, elapsed_time: float = None
):
    """
    Genera y guarda estadísticas en JSON.
    El documento MD global se genera desde scripts/generar_datos_md.py
    """
    print("\n" + "=" * 80)
    print("GENERANDO ESTADÍSTICAS")
    print("=" * 80)

    # Generar estadísticas
    stats = generate_statistics(df, sessions_df)

    # Agregar tiempo de ejecución si está disponible
    if elapsed_time is not None:
        stats["execution_time"] = {
            "seconds": round(elapsed_time, 2),
            "minutes": round(elapsed_time / 60, 2),
        }

    # Guardar JSON
    json_path = SCRIPT_DIR / "data_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Estadísticas guardadas en: {json_path}")
    print(
        f"[INFO] Para generar DATOS.md global ejecutar: python scripts/generar_datos_md.py"
    )

    return stats


def main():
    """Genera todos los CSVs de splits."""
    global OVERLAP_SECONDS

    args = parse_args()
    OVERLAP_SECONDS = float(args.overlap)

    # Iniciar timer
    start_time = time.time()

    print("=" * 80)
    print("GENERACION DE SPLITS ESTRATIFICADOS POR SESION")
    print("=" * 80)
    print(f"\nConfiguracion:")
    print(f"  RANDOM_SEED = {RANDOM_SEED}")
    print(f"  SEGMENT_DURATION = {SEGMENT_DURATION}s")
    print(f"  OVERLAP_SECONDS = {OVERLAP_SECONDS}s")
    print(f"  BLIND_FRACTION = {BLIND_FRACTION:.1%} (validacion vida real)")
    print(f"  TEST_FRACTION = {TEST_FRACTION:.1%} (evaluacion desarrollo)")
    print(f"  VAL_FRACTION = {VAL_FRACTION:.1%}")
    print(f"  TRAIN_FRACTION = {1 - BLIND_FRACTION - TEST_FRACTION - VAL_FRACTION:.1%}")

    with timer("Carga sesiones + conteo segmentos"):
        sessions_df = load_all_sessions()

    if sessions_df.empty:
        print("ERROR: No hay sesiones para procesar")
        return

    with timer("Split estratificado por sesión"):
        print("\n" + "=" * 80)
        print("DIVIDIENDO POR SESION")
        print("=" * 80)
        sessions_df = split_by_session(
            sessions_df, BLIND_FRACTION, TEST_FRACTION, VAL_FRACTION, RANDOM_SEED
        )

    with timer("Expandir sesiones a segmentos"):
        df = expand_sessions_to_segments(sessions_df)

    # Asignar Split a cada segmento basándose en su sesión
    session_to_split = dict(zip(sessions_df["Session"], sessions_df["Split"]))
    df["Split"] = df["Session"].map(session_to_split)

    with timer("Guardar CSVs"):
        save_splits(df)

    # Calcular tiempo de ejecución
    elapsed_time = time.time() - start_time

    with timer("Guardar estadísticas"):
        save_statistics(df, sessions_df, elapsed_time=elapsed_time)

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print(f"\nDistribucion de splits:")
    for split_name, count in df["Split"].value_counts().items():
        pct = count / len(df) * 100
        sessions = df[df["Split"] == split_name]["Session"].nunique()
        print(f"  {split_name}: {count} segmentos ({pct:.1f}%), {sessions} sesiones")

    # Verificar reproducibilidad
    print(f"\n[INFO] Semilla usada: {RANDOM_SEED}")
    print(f"[INFO] Duración de segmento: {SEGMENT_DURATION}s")
    print("[INFO] Ejecutar con la misma semilla producira los mismos splits")
    print(f"\nTiempo de ejecución: {elapsed_time:.2f}s ({elapsed_time / 60:.2f}min)")

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
