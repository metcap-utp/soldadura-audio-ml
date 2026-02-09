"""
Genera CSVs de splits (train/test/val) usando split estratificado por sesion.

Este sistema:
1. Agrupa archivos por SESION (carpeta de grabacion) para evitar data leakage
2. Estratifica por combinacion de etiquetas (plate + electrode + current)
3. Usa semilla fija para reproducibilidad
4. Permite configurar fracciones de train/test/val

Estructura esperada de archivos:
    audio/Placa_Xmm/EXXXX/{AC,DC}/YYMMDD-HHMMSS_Audio/*.wav

Que es una SESION?
    Una sesion es una grabacion continua de soldadura, identificada por
    su carpeta con timestamp (ej: 240912-143741_Audio). De cada grabacion
    se extraen multiples segmentos. Es importante que todos los segmentos
    de la misma grabacion vayan al mismo split para evitar data leakage.

Genera:
    - train.csv
    - test.csv
    - val.csv (si VAL_FRACTION > 0)
    - completo.csv (todos los archivos con columna Split)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURACION - Modificar estas variables segun necesidad
# =============================================================================

# Semilla para reproducibilidad (mismo valor = mismo split)
RANDOM_SEED = 42

# Fraccion de datos para test (0.0 - 1.0)
TEST_FRACTION = 0.20

# Fraccion de datos para validacion (0.0 - 1.0, del total, no del restante)
# Si es 0.0, no se genera conjunto de validacion
VAL_FRACTION = 0.0

# =============================================================================
# CODIGO - No modificar a menos que sea necesario
# =============================================================================

# Paths relativos desde soldadura/
SCRIPT_DIR = Path("10seg")
AUDIO_DIR = SCRIPT_DIR / "audio"


def extract_labels_from_path(audio_path: Path) -> dict:
    """
    Extrae etiquetas del path del archivo.
    
    Ejemplo path relativo desde SCRIPT_DIR:
    audio/Placa_6mm/E6011/AC/240802-105935_Audio/drums_louder_50.wav
    
    parts[0] = audio
    parts[1] = Placa_Xmm
    parts[2] = EXXXX
    parts[3] = AC/DC
    parts[4] = Session (YYMMDD-HHMMSS_Audio)
    parts[5] = filename.wav
    
    Returns:
        dict con Audio Path, Plate Thickness, Electrode, Type of Current, Session
    """
    # Usar path relativo desde SCRIPT_DIR
    try:
        relative_path = audio_path.relative_to(SCRIPT_DIR)
    except ValueError:
        print(f"Warning: No se pudo obtener path relativo para {audio_path}")
        return None
    
    parts = relative_path.parts
    
    # Verificar que tiene la estructura esperada (6 partes)
    if len(parts) != 6:
        print(f"Warning: Path con estructura inesperada ({len(parts)} partes): {relative_path}")
        return None
    
    try:
        plate_thickness = parts[1]  # Placa_Xmm
        electrode = parts[2]        # EXXXX
        current_type = parts[3]     # AC/DC
        session = parts[4]          # YYMMDD-HHMMSS_Audio
        
        return {
            "Audio Path": str(relative_path),
            "Plate Thickness": plate_thickness,
            "Electrode": electrode,
            "Type of Current": current_type,
            "Session": session,
        }
    except (ValueError, IndexError) as e:
        print(f"Warning: No se pudo procesar {audio_path}: {e}")
        return None


def load_all_audio_files() -> pd.DataFrame:
    """Carga todos los archivos de audio y extrae sus etiquetas."""
    print("Escaneando archivos de audio...")
    
    # Buscar todos los archivos .wav en cualquier subdirectorio
    audio_files = list(AUDIO_DIR.rglob("*.wav"))
    print(f"  Archivos encontrados: {len(audio_files)}")
    
    # Extraer etiquetas de cada archivo
    data_list = []
    for audio_file in audio_files:
        labels = extract_labels_from_path(audio_file)
        if labels:
            data_list.append(labels)
    
    df = pd.DataFrame(data_list)
    
    if df.empty:
        print("ERROR: No se encontraron archivos validos")
        return df
    
    print(f"  Registros validos: {len(df)}")
    print(f"  Sesiones unicas: {df['Session'].nunique()}")
    
    return df


def create_stratification_label(df: pd.DataFrame) -> pd.Series:
    """
    Crea etiqueta combinada para estratificacion.
    Combina plate + electrode + current para mantener proporciones.
    """
    return df["Plate Thickness"] + "_" + df["Electrode"] + "_" + df["Type of Current"]


def split_by_session(df: pd.DataFrame, test_frac: float, val_frac: float, seed: int):
    """
    Divide los datos por sesion de forma estratificada.
    
    Args:
        df: DataFrame con todos los archivos
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
    # Para estratificacion, necesitamos al menos 2 ejemplos por clase
    rare_classes = strat_counts[strat_counts < 2].index.tolist()
    
    if rare_classes:
        print(f"\nWarning: Clases con <2 sesiones (se asignaran a train): {len(rare_classes)} clases")
        # Marcar sesiones raras para forzarlas a train
        sessions_df["is_rare"] = sessions_df["Strat_Label"].isin(rare_classes)
    else:
        sessions_df["is_rare"] = False
    
    # Separar sesiones raras y no raras
    rare_sessions = sessions_df[sessions_df["is_rare"]]["Session"].tolist()
    normal_sessions_df = sessions_df[~sessions_df["is_rare"]]
    
    print(f"\nSesiones totales: {len(sessions_df)}")
    print(f"  - Normales (estratificables): {len(normal_sessions_df)}")
    print(f"  - Raras (asignadas a train): {len(rare_sessions)}")
    
    # Inicializar splits
    session_splits = {}
    for session in rare_sessions:
        session_splits[session] = "train"
    
    if len(normal_sessions_df) > 0:
        sessions = normal_sessions_df["Session"].values
        strat_labels = normal_sessions_df["Strat_Label"].values
        
        # Calcular fracciones ajustadas (considerando sesiones raras ya asignadas)
        total_sessions = len(sessions_df)
        normal_sessions = len(normal_sessions_df)
        
        # Ajustar fracciones para el conjunto normal
        adjusted_test_frac = min(test_frac * total_sessions / normal_sessions, 0.5)
        adjusted_val_frac = min(val_frac * total_sessions / normal_sessions, 0.5) if val_frac > 0 else 0
        
        # Split train vs (test + val)
        test_val_frac = adjusted_test_frac + adjusted_val_frac
        
        if test_val_frac > 0:
            try:
                train_sessions, test_val_sessions, _, test_val_labels = train_test_split(
                    sessions,
                    strat_labels,
                    test_size=test_val_frac,
                    random_state=seed,
                    stratify=strat_labels
                )
            except ValueError as e:
                print(f"Warning: No se pudo estratificar, usando split aleatorio: {e}")
                train_sessions, test_val_sessions = train_test_split(
                    sessions,
                    test_size=test_val_frac,
                    random_state=seed
                )
                test_val_labels = normal_sessions_df[
                    normal_sessions_df["Session"].isin(test_val_sessions)
                ]["Strat_Label"].values
            
            # Asignar train
            for session in train_sessions:
                session_splits[session] = "train"
            
            # Split test vs val (si hay validacion)
            if adjusted_val_frac > 0 and len(test_val_sessions) > 1:
                val_ratio = adjusted_val_frac / test_val_frac
                try:
                    test_sessions, val_sessions = train_test_split(
                        test_val_sessions,
                        test_size=val_ratio,
                        random_state=seed,
                        stratify=test_val_labels
                    )
                except ValueError:
                    test_sessions, val_sessions = train_test_split(
                        test_val_sessions,
                        test_size=val_ratio,
                        random_state=seed
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
            for session in sessions:
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
        
        # Remover columnas auxiliares para archivos individuales
        save_df = split_df.drop(columns=["Session", "Split"])
        save_df = save_df.sort_values("Audio Path").reset_index(drop=True)
        
        output_path = SCRIPT_DIR / f"{split_name}.csv"
        save_df.to_csv(output_path, index=False)
        
        print(f"\n{split_name.upper()}: {len(save_df)} archivos")
        print(f"  Espesores: {split_df['Plate Thickness'].value_counts().to_dict()}")
        print(f"  Electrodos: {split_df['Electrode'].value_counts().to_dict()}")
        print(f"  Corrientes: {split_df['Type of Current'].value_counts().to_dict()}")
        print(f"  Sesiones: {split_df['Session'].nunique()}")
        print(f"  Guardado: {output_path}")
    
    # Guardar completo.csv con todas las columnas
    print("\n" + "-" * 40)
    complete_df = df.drop(columns=["Session"]).sort_values(["Split", "Audio Path"]).reset_index(drop=True)
    output_path = SCRIPT_DIR / "completo.csv"
    complete_df.to_csv(output_path, index=False)
    print(f"COMPLETO: {len(complete_df)} archivos")
    print(f"  Guardado: {output_path}")


def main():
    """Genera todos los CSVs de splits."""
    print("=" * 80)
    print("GENERACION DE SPLITS ESTRATIFICADOS POR SESION")
    print("=" * 80)
    print(f"\nConfiguracion:")
    print(f"  RANDOM_SEED = {RANDOM_SEED}")
    print(f"  TEST_FRACTION = {TEST_FRACTION:.1%}")
    print(f"  VAL_FRACTION = {VAL_FRACTION:.1%}")
    print(f"  TRAIN_FRACTION = {1 - TEST_FRACTION - VAL_FRACTION:.1%}")
    
    # Cargar todos los archivos
    df = load_all_audio_files()
    
    if df.empty:
        print("ERROR: No hay archivos para procesar")
        return
    
    # Realizar split por sesion
    print("\n" + "=" * 80)
    print("DIVIDIENDO POR SESION")
    print("=" * 80)
    df = split_by_session(df, TEST_FRACTION, VAL_FRACTION, RANDOM_SEED)
    
    # Guardar resultados
    save_splits(df)
    
    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print(f"\nDistribucion de splits:")
    for split_name, count in df["Split"].value_counts().items():
        pct = count / len(df) * 100
        sessions = df[df["Split"] == split_name]["Session"].nunique()
        print(f"  {split_name}: {count} archivos ({pct:.1f}%), {sessions} sesiones")
    
    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
