"""
Script de predicción usando Soft Voting.

Carga los K modelos entrenados con K-Fold y combina sus predicciones
promediando logits antes de aplicar argmax (soft voting).

Los audios se segmentan ON-THE-FLY según la duración del directorio
(5seg, 10seg, 30seg) - NO hay archivos segmentados en disco.

Uso:
    python infer.py                      # Usa 5-fold por defecto
    python infer.py --k-folds 3          # Usa modelos de 3-fold
    python infer.py --audio ruta.wav     # Predice un archivo específico
    python infer.py --evaluar            # Evalúa en conjunto blind (vida real)
    python infer.py --evaluar --k-folds 10  # Evalúa usando modelos de 10-fold
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

# Añadir carpeta raíz al path para importar modelo.py
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from modelo import SMAWXVectorModel
from utils.audio_utils import (
    PROJECT_ROOT,
    get_script_segment_duration,
    load_audio_segment,
)
from utils.timing import timer

# Definir directorios
SCRIPT_DIR = Path(__file__).parent
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"
INFER_JSON = SCRIPT_DIR / "infer.json"

# Duración de segmento basada en el nombre del directorio
DEFAULT_SEGMENT_DURATION = get_script_segment_duration(Path(__file__))


# ============= Parseo de argumentos (antes de cargar modelos) =============
def parse_args():
    parser = argparse.ArgumentParser(
        description="Predicción de soldadura SMAW usando Ensemble con Soft Voting"
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Número de folds del ensemble a usar (default: 5)",
    )
    parser.add_argument(
        "--audio", type=str, help="Ruta a un archivo de audio para predecir"
    )
    parser.add_argument(
        "--evaluar",
        action="store_true",
        help="Evaluar ensemble en conjunto blind (vida real)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Número de muestras aleatorias a mostrar (default: 10)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Solapamiento del modelo entrenado como fracción (0-1). Ej: 0.5 = 50%% (default: 0.0)",
    )
    parser.add_argument(
        "--train-seconds",
        type=int,
        default=None,
        help="Duración (seg) usada para entrenar el modelo a cargar (ej: 30). Default: el directorio actual.",
    )
    parser.add_argument(
        "--test-seconds",
        type=int,
        default=None,
        help="Duración (seg) usada para segmentar/evaluar en inferencia (ej: 1). Default: el directorio actual.",
    )
    return parser.parse_args()


# Parsear argumentos antes de cargar modelos
args = parse_args()
N_MODELS = args.k_folds

DEFAULT_SECONDS = int(DEFAULT_SEGMENT_DURATION)
TRAIN_SECONDS = (
    int(args.train_seconds) if args.train_seconds is not None else DEFAULT_SECONDS
)
TEST_SECONDS = (
    int(args.test_seconds) if args.test_seconds is not None else DEFAULT_SECONDS
)

TRAIN_DIR = ROOT_DIR / f"{TRAIN_SECONDS}seg"
TEST_DIR = ROOT_DIR / f"{TEST_SECONDS}seg"

SEGMENT_DURATION = float(TEST_SECONDS)
OVERLAP_RATIO = float(args.overlap)  # Fracción 0-1 usada en entrenamiento
OVERLAP_SECONDS = OVERLAP_RATIO * SEGMENT_DURATION  # Segundos para inferencia

OVERLAP_TAG = str(OVERLAP_RATIO)
MODELS_DIR_NEW = TRAIN_DIR / "models" / f"k{N_MODELS:02d}_overlap_{OVERLAP_TAG}"
MODELS_DIR_OLD = TRAIN_DIR / "models" / f"{N_MODELS}-fold"
MODELS_DIR = MODELS_DIR_NEW if MODELS_DIR_NEW.exists() else MODELS_DIR_OLD

if MODELS_DIR == MODELS_DIR_OLD and MODELS_DIR_NEW != MODELS_DIR_OLD:
    print(f"[INFO] Modelos (legacy): {MODELS_DIR}")
else:
    print(f"[INFO] Modelos:          {MODELS_DIR}")

# Cargar modelo VGGish de TensorFlow Hub
print(f"Cargando modelo VGGish desde TensorFlow Hub...")
vggish_model = hub.load(VGGISH_MODEL_URL)
print("Modelo VGGish cargado correctamente.")
print(f"Duración segmento (train): {TRAIN_SECONDS}s")
print(f"Duración segmento (test):  {SEGMENT_DURATION}s")
print(f"Solapamiento (test):       {OVERLAP_SECONDS}s")


# ============= Extracción de embeddings VGGish =============


def extract_vggish_embeddings_from_segment(
    audio_path: str, segment_idx: int
) -> np.ndarray:
    """Extrae embeddings VGGish de un segmento específico de audio."""
    full_path = PROJECT_ROOT / audio_path

    # Cargar el segmento específico
    segment = load_audio_segment(
        full_path,
        segment_duration=SEGMENT_DURATION,
        segment_index=segment_idx,
        sr=16000,
        overlap_seconds=OVERLAP_SECONDS,
    )

    if segment is None:
        raise ValueError(f"No se pudo cargar segmento {segment_idx} de {audio_path}")

    # VGGish espera ventanas de 1 segundo con hop de 0.5 segundos
    window_size = 16000  # 1 segundo a 16kHz
    hop_size = 8000  # 0.5 segundos
    embeddings_list = []

    for start in range(0, len(segment), hop_size):
        end = start + window_size
        if end > len(segment):
            # Padding al final
            window = np.zeros(window_size, dtype=np.float32)
            window[: len(segment) - start] = segment[start:]
        else:
            window = segment[start:end]

        embedding = vggish_model(window).numpy()
        embeddings_list.append(embedding[0])

        if end >= len(segment):
            break

    return np.stack(embeddings_list, axis=0)


def extract_vggish_embeddings(audio_path):
    """Extrae embeddings VGGish de un archivo de audio."""
    import librosa

    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    window_size = 16000
    hop_size = 8000

    embeddings_list = []

    for start in range(0, len(y), hop_size):
        end = start + window_size

        if end > len(y):
            segment = np.zeros(window_size, dtype=np.float32)
            segment[: len(y) - start] = y[start:]
        else:
            segment = y[start:end]

        embedding = vggish_model(segment).numpy()
        embeddings_list.append(embedding[0])

        if end >= len(y):
            break

    embeddings = np.stack(embeddings_list, axis=0)
    return embeddings


# ============= Cargar encoders desde train.csv =============

train_data = pd.read_csv(TRAIN_DIR / "train.csv")

plate_encoder = LabelEncoder()
electrode_encoder = LabelEncoder()
current_type_encoder = LabelEncoder()

plate_encoder.fit(train_data["Plate Thickness"])
electrode_encoder.fit(train_data["Electrode"])
current_type_encoder.fit(train_data["Type of Current"])

# ============= Cargar Ensemble de Modelos =============

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


def create_model():
    """Crea una instancia del modelo."""
    return SMAWXVectorModel(
        feat_dim=128,
        xvector_dim=512,
        emb_dim=256,
        num_classes_espesor=len(plate_encoder.classes_),
        num_classes_electrodo=len(electrode_encoder.classes_),
        num_classes_corriente=len(current_type_encoder.classes_),
    ).to(device)


def load_ensemble_models():
    """Carga los K modelos del ensemble según --k-folds."""
    if not MODELS_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta {MODELS_DIR}. "
            f"Ejecuta primero: python entrenar.py --k-folds {N_MODELS}"
        )

    models = []

    for fold in range(N_MODELS):
        model_path = MODELS_DIR / f"model_fold_{fold}.pth"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo {model_path}. "
                f"Ejecuta primero: python entrenar.py --k-folds {N_MODELS}"
            )

        model = create_model()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    print(f"Cargados {len(models)} modelos ({N_MODELS}-fold) desde {MODELS_DIR}")
    return models


# Cargar todos los modelos
ensemble_models = load_ensemble_models()


# ============= Guardar resultados de inferencia =============


def save_inference_result(result_data, elapsed_time=None):
    """
    Guarda los resultados de inferencia en infer.json.
    Conserva todas las corridas anteriores.
    """
    # Cargar resultados existentes
    if INFER_JSON.exists():
        with open(INFER_JSON, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Agregar metadatos al resultado
    result_data["timestamp"] = datetime.now().isoformat()
    result_data["config"] = {
        "train_seconds": TRAIN_SECONDS,
        "test_seconds": SEGMENT_DURATION,
        "overlap_seconds": OVERLAP_SECONDS,
        "k_folds": N_MODELS,
        "models_dir": str(MODELS_DIR),
        "train_dir": str(TRAIN_DIR),
        "test_dir": str(TEST_DIR),
    }

    # Agregar tiempo de ejecución si está disponible
    if elapsed_time is not None:
        result_data["execution_time"] = {
            "seconds": round(elapsed_time, 2),
            "minutes": round(elapsed_time / 60, 2),
            "hours": round(elapsed_time / 3600, 4),
        }

    # Agregar nuevo resultado
    all_results.append(result_data)

    # Guardar
    with open(INFER_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResultados guardados en {INFER_JSON}")


def format_confusion_matrix_markdown(cm, classes):
    """Formatea una matriz de confusión como tabla Markdown."""
    # Header
    header = "| Pred \\ Real | " + " | ".join(classes) + " |"
    separator = "|" + "|".join(["---"] * (len(classes) + 1)) + "|"

    rows = [header, separator]
    for i, cls in enumerate(classes):
        row = (
            f"| **{cls}** | "
            + " | ".join(str(cm[i][j]) for j in range(len(classes)))
            + " |"
        )
        rows.append(row)

    return "\n".join(rows)


def generate_metrics_document(results):
    """
    Genera un documento Markdown con todas las métricas y matrices de confusión.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    segment_duration = results.get("segment_duration", SEGMENT_DURATION)

    doc = f"""# Métricas de Clasificación SMAW - {int(segment_duration)}seg

**Fecha de evaluación:** {timestamp}

**Configuración:**
- Duración de segmento: {segment_duration}s
- Número de muestras (blind): {results["n_samples"]}
- Número de modelos (ensemble): {results["n_models"]}
- Método de votación: {results["voting_method"]}

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | {results["accuracy"]["plate_thickness"] * 100:.2f}% | {results["macro_f1"]["plate_thickness"]:.4f} |
| Electrode Type | {results["accuracy"]["electrode"] * 100:.2f}% | {results["macro_f1"]["electrode"]:.4f} |
| Current Type | {results["accuracy"]["current_type"] * 100:.2f}% | {results["macro_f1"]["current_type"]:.4f} |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** {results["accuracy"]["plate_thickness"] * 100:.2f}%
- **Macro F1-Score:** {results["macro_f1"]["plate_thickness"]:.4f}

### Confusion Matrix

{format_confusion_matrix_markdown(results["confusion_matrices"]["plate_thickness"], results["classes"]["plate_thickness"])}

### Classification Report
"""
    # Agregar métricas por clase para plate
    cr_plate = results["classification_reports"]["plate_thickness"]
    doc += "\n| Clase | Precision | Recall | F1-Score | Support |\n"
    doc += "|-------|-----------|--------|----------|--------|\n"
    for cls in results["classes"]["plate_thickness"]:
        metrics = cr_plate.get(cls, {})
        doc += f"| {cls} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('f1-score', 0):.4f} | {int(metrics.get('support', 0))} |\n"

    doc += f"""
---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** {results["accuracy"]["electrode"] * 100:.2f}%
- **Macro F1-Score:** {results["macro_f1"]["electrode"]:.4f}

### Confusion Matrix

{format_confusion_matrix_markdown(results["confusion_matrices"]["electrode"], results["classes"]["electrode"])}

### Classification Report
"""
    # Agregar métricas por clase para electrode
    cr_electrode = results["classification_reports"]["electrode"]
    doc += "\n| Clase | Precision | Recall | F1-Score | Support |\n"
    doc += "|-------|-----------|--------|----------|--------|\n"
    for cls in results["classes"]["electrode"]:
        metrics = cr_electrode.get(cls, {})
        doc += f"| {cls} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('f1-score', 0):.4f} | {int(metrics.get('support', 0))} |\n"

    doc += f"""
---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** {results["accuracy"]["current_type"] * 100:.2f}%
- **Macro F1-Score:** {results["macro_f1"]["current_type"]:.4f}

### Confusion Matrix

{format_confusion_matrix_markdown(results["confusion_matrices"]["current_type"], results["classes"]["current_type"])}

### Classification Report
"""
    # Agregar métricas por clase para current
    cr_current = results["classification_reports"]["current_type"]
    doc += "\n| Clase | Precision | Recall | F1-Score | Support |\n"
    doc += "|-------|-----------|--------|----------|--------|\n"
    for cls in results["classes"]["current_type"]:
        metrics = cr_current.get(cls, {})
        doc += f"| {cls} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('f1-score', 0):.4f} | {int(metrics.get('support', 0))} |\n"

    doc += """
---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
"""

    # Guardar documento
    metrics_dir = SCRIPT_DIR / "metricas"
    metrics_dir.mkdir(exist_ok=True)
    metrics_file = metrics_dir / "METRICAS.md"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"Documento de métricas guardado en {metrics_file}")


# ============= Predicción con Ensemble (Soft Voting) =============


def predict_ensemble(embeddings_tensor):
    """
    Realiza predicción con ensemble usando soft voting.

    Proceso:
    1. Obtiene logits de cada modelo
    2. Promedia los logits de los 5 modelos
    3. Aplica argmax para obtener la clase predicha

    Args:
        embeddings_tensor: Tensor de embeddings VGGish [1, T, 128]

    Returns:
        dict con predicciones y probabilidades
    """
    # Acumular logits de todos los modelos
    logits_espesor_list = []
    logits_electrodo_list = []
    logits_corriente_list = []

    with torch.no_grad():
        for model in ensemble_models:
            outputs = model(embeddings_tensor)
            logits_espesor_list.append(outputs["logits_espesor"])
            logits_electrodo_list.append(outputs["logits_electrodo"])
            logits_corriente_list.append(outputs["logits_corriente"])

    # Promediar logits (soft voting)
    avg_logits_espesor = torch.stack(logits_espesor_list).mean(dim=0)
    avg_logits_electrodo = torch.stack(logits_electrodo_list).mean(dim=0)
    avg_logits_corriente = torch.stack(logits_corriente_list).mean(dim=0)

    # Obtener clases predichas
    pred_plate_idx = avg_logits_espesor.argmax(dim=1).item()
    pred_electrode_idx = avg_logits_electrodo.argmax(dim=1).item()
    pred_current_idx = avg_logits_corriente.argmax(dim=1).item()

    # Decodificar etiquetas
    pred_plate = plate_encoder.classes_[pred_plate_idx]
    pred_electrode = electrode_encoder.classes_[pred_electrode_idx]
    pred_current = current_type_encoder.classes_[pred_current_idx]

    # Obtener probabilidades (softmax sobre logits promediados)
    probs_plate = torch.softmax(avg_logits_espesor, dim=1)[0].cpu().numpy()
    probs_electrode = torch.softmax(avg_logits_electrodo, dim=1)[0].cpu().numpy()
    probs_current = torch.softmax(avg_logits_corriente, dim=1)[0].cpu().numpy()

    return {
        "plate": pred_plate,
        "electrode": pred_electrode,
        "current": pred_current,
        "probs_plate": probs_plate,
        "probs_electrode": probs_electrode,
        "probs_current": probs_current,
        "plate_idx": pred_plate_idx,
        "electrode_idx": pred_electrode_idx,
        "current_idx": pred_current_idx,
    }


def predict_segment(audio_path: str, segment_idx: int):
    """Predice las tres etiquetas de un segmento de audio usando ensemble."""
    embeddings = extract_vggish_embeddings_from_segment(audio_path, segment_idx)
    embeddings_tensor = (
        torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    )
    return predict_ensemble(embeddings_tensor)


def predict_audio(audio_path):
    """Predice las tres etiquetas de un archivo de audio completo usando ensemble."""
    embeddings = extract_vggish_embeddings(audio_path)
    embeddings_tensor = (
        torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    )
    return predict_ensemble(embeddings_tensor)


# ============= Funciones de Evaluación =============


def evaluate_blind_set():
    """Evalúa el ensemble en el conjunto blind (validación vida real)."""
    start_time = time.time()

    blind_csv = TEST_DIR / "blind.csv"
    if not blind_csv.exists():
        print(
            f"No se encontró {blind_csv}. Ejecuta generar_splits.py en {TEST_DIR.name} primero."
        )
        return None

    with timer("Cargar blind.csv"):
        blind_df = pd.read_csv(blind_csv)
    print(f"\nEvaluando ensemble en {len(blind_df)} segmentos de BLIND (vida real)...")
    print(f"Duración de segmento (test): {SEGMENT_DURATION}s")
    print(f"Solapamiento (test): {OVERLAP_SECONDS}s")

    # Listas para almacenar predicciones y etiquetas reales
    y_true_plate, y_pred_plate = [], []
    y_true_electrode, y_pred_electrode = [], []
    y_true_current, y_pred_current = [], []

    with timer("Inferencia BLIND (segmentos)"):
        for idx, row in blind_df.iterrows():
            if idx % 100 == 0:
                print(f"  Procesando {idx}/{len(blind_df)}...")

            audio_path = row["Audio Path"]
            segment_idx = int(row["Segment Index"])

            # Etiquetas reales
            y_true_plate.append(row["Plate Thickness"])
            y_true_electrode.append(row["Electrode"])
            y_true_current.append(row["Type of Current"])

            # Predicciones usando segmento on-the-fly
            result = predict_segment(audio_path, segment_idx)
            y_pred_plate.append(result["plate"])
            y_pred_electrode.append(result["electrode"])
            y_pred_current.append(result["current"])

    # Calcular métricas
    print("\n" + "=" * 70)
    print("RESULTADOS DEL ENSEMBLE EN CONJUNTO BLIND (VIDA REAL)")
    print("=" * 70)

    acc_plate = accuracy_score(y_true_plate, y_pred_plate)
    acc_electrode = accuracy_score(y_true_electrode, y_pred_electrode)
    acc_current = accuracy_score(y_true_current, y_pred_current)

    # Métricas globales multi-tarea
    n_samples = len(y_true_plate)
    exact_matches = sum(
        1
        for i in range(n_samples)
        if (
            y_pred_plate[i] == y_true_plate[i]
            and y_pred_electrode[i] == y_true_electrode[i]
            and y_pred_current[i] == y_true_current[i]
        )
    )
    exact_match_accuracy = exact_matches / n_samples
    hamming_accuracy = (acc_plate + acc_electrode + acc_current) / 3

    print(f"\nMétricas Globales (Multi-tarea):")
    print(f"  Exact Match Accuracy: {exact_match_accuracy * 100:.2f}%")
    print(f"  Hamming Accuracy:     {hamming_accuracy * 100:.2f}%")

    print(f"\nAccuracy por Tarea:")
    print(f"  Plate Thickness:  {acc_plate * 100:.2f}%")
    print(f"  Electrode:        {acc_electrode * 100:.2f}%")
    print(f"  Type of Current:  {acc_current * 100:.2f}%")

    # Calcular Macro F1
    f1_plate = f1_score(y_true_plate, y_pred_plate, average="macro")
    f1_electrode = f1_score(y_true_electrode, y_pred_electrode, average="macro")
    f1_current = f1_score(y_true_current, y_pred_current, average="macro")

    print(f"\nMacro F1-Score:")
    print(f"  Plate Thickness:  {f1_plate:.4f}")
    print(f"  Electrode:        {f1_electrode:.4f}")
    print(f"  Type of Current:  {f1_current:.4f}")

    # Reportes de clasificación
    print("\n--- Plate Thickness ---")
    print(classification_report(y_true_plate, y_pred_plate))

    print("\n--- Electrode Type ---")
    print(classification_report(y_true_electrode, y_pred_electrode))

    print("\n--- Type of Current ---")
    print(classification_report(y_true_current, y_pred_current))

    # Matrices de confusión
    print("\nMatrices de Confusión:")
    print("\nPlate Thickness:")
    cm_plate = confusion_matrix(
        y_true_plate, y_pred_plate, labels=plate_encoder.classes_
    )
    print(cm_plate)
    print(f"Clases: {plate_encoder.classes_}")

    print("\nElectrode Type:")
    cm_electrode = confusion_matrix(
        y_true_electrode, y_pred_electrode, labels=electrode_encoder.classes_
    )
    print(cm_electrode)
    print(f"Clases: {electrode_encoder.classes_}")

    print("\nType of Current:")
    cm_current = confusion_matrix(
        y_true_current, y_pred_current, labels=current_type_encoder.classes_
    )
    print(cm_current)
    print(f"Clases: {current_type_encoder.classes_}")

    # Crear diccionario de resultados
    results = {
        "mode": "blind_evaluation",
        "segment_duration": SEGMENT_DURATION,
        "n_samples": len(blind_df),
        "n_models": N_MODELS,
        "voting_method": "soft",
        "global_metrics": {
            "exact_match_accuracy": float(exact_match_accuracy),
            "hamming_accuracy": float(hamming_accuracy),
        },
        "accuracy": {
            "plate_thickness": float(acc_plate),
            "electrode": float(acc_electrode),
            "current_type": float(acc_current),
        },
        "macro_f1": {
            "plate_thickness": float(f1_plate),
            "electrode": float(f1_electrode),
            "current_type": float(f1_current),
        },
        "classification_reports": {
            "plate_thickness": classification_report(
                y_true_plate, y_pred_plate, output_dict=True
            ),
            "electrode": classification_report(
                y_true_electrode, y_pred_electrode, output_dict=True
            ),
            "current_type": classification_report(
                y_true_current, y_pred_current, output_dict=True
            ),
        },
        "confusion_matrices": {
            "plate_thickness": cm_plate.tolist(),
            "electrode": cm_electrode.tolist(),
            "current_type": cm_current.tolist(),
        },
        "classes": {
            "plate_thickness": plate_encoder.classes_.tolist(),
            "electrode": electrode_encoder.classes_.tolist(),
            "current_type": current_type_encoder.classes_.tolist(),
        },
    }

    # Calcular tiempo de ejecución
    elapsed_time = time.time() - start_time
    print(f"\nTiempo de ejecución: {elapsed_time:.2f}s ({elapsed_time / 60:.2f}min)")

    # Guardar resultados en infer.json
    save_inference_result(results, elapsed_time=elapsed_time)

    # Generar documento de métricas
    generate_metrics_document(results)

    return results


def show_random_predictions(n_samples=10):
    """Muestra predicciones aleatorias del conjunto blind."""
    start_time = time.time()

    blind_csv = TEST_DIR / "blind.csv"
    if not blind_csv.exists():
        print(
            f"No se encontró {blind_csv}. Ejecuta generar_splits.py en {TEST_DIR.name} primero."
        )
        return

    with timer("Cargar blind.csv"):
        blind_df = pd.read_csv(blind_csv)
    num_samples = min(n_samples, len(blind_df))
    samples = blind_df.sample(n=num_samples, random_state=None)

    print(f"\n{'=' * 80}")
    print(f"  PREDICCIONES ALEATORIAS ({num_samples} muestras)")
    print(f"{'=' * 80}")
    print(f"\n  {'Archivo':<25} {'Plate':<12} {'Electrode':<10} {'Current':<8}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 10} {'-' * 8}")

    correctas_plate = 0
    correctas_electrode = 0
    correctas_current = 0
    correctas_todas = 0

    with timer("Inferencia muestras aleatorias"):
        for _, row in samples.iterrows():
            audio_path = PROJECT_ROOT / row["Audio Path"]
            real_plate = row["Plate Thickness"]
            real_electrode = row["Electrode"]
            real_current = row["Type of Current"]

            result = predict_audio(str(audio_path))

        correct_plate = result["plate"] == real_plate
        correct_electrode = result["electrode"] == real_electrode
        correct_current = result["current"] == real_current
        correct_all = correct_plate and correct_electrode and correct_current

        if correct_plate:
            correctas_plate += 1
        if correct_electrode:
            correctas_electrode += 1
        if correct_current:
            correctas_current += 1
        if correct_all:
            correctas_todas += 1

        # Símbolos para resultado
        sym_plate = "✓" if correct_plate else "✗"
        sym_electrode = "✓" if correct_electrode else "✗"
        sym_current = "✓" if correct_current else "✗"

        print(f"\n  {audio_path.name:<25}")
        print(f"    Real:      {real_plate:<12} {real_electrode:<10} {real_current:<8}")
        print(
            f"    Pred:      {result['plate']:<12} {result['electrode']:<10} {result['current']:<8}"
        )
        print(f"    Status:    {sym_plate:<12} {sym_electrode:<10} {sym_current:<8}")
        print(
            f"    Conf:      {result['probs_plate'].max():.2f}         {result['probs_electrode'].max():.2f}       {result['probs_current'].max():.2f}"
        )

    print(f"\n{'=' * 80}")
    print(f"  RESUMEN ({N_MODELS} modelos con Soft Voting)")
    print(f"{'=' * 80}")
    print(
        f"\n  Plate Thickness:  {correctas_plate:>2}/{num_samples} = {correctas_plate / num_samples * 100:>6.2f}%"
    )
    print(
        f"  Electrode:        {correctas_electrode:>2}/{num_samples} = {correctas_electrode / num_samples * 100:>6.2f}%"
    )
    print(
        f"  Type of Current:  {correctas_current:>2}/{num_samples} = {correctas_current / num_samples * 100:>6.2f}%"
    )
    print(
        f"  Todas correctas:  {correctas_todas:>2}/{num_samples} = {correctas_todas / num_samples * 100:>6.2f}%"
    )
    print()

    # Calcular tiempo de ejecución
    elapsed_time = time.time() - start_time
    print(f"Tiempo de ejecución: {elapsed_time:.2f}s ({elapsed_time / 60:.2f}min)")

    # Guardar resultados en infer.json
    inference_result = {
        "mode": "random_predictions",
        "n_samples": num_samples,
        "n_models": N_MODELS,
        "voting_method": "soft",
        "accuracy": {
            "plate_thickness": correctas_plate / num_samples,
            "electrode": correctas_electrode / num_samples,
            "current_type": correctas_current / num_samples,
            "all_correct": correctas_todas / num_samples,
        },
    }
    save_inference_result(inference_result, elapsed_time=elapsed_time)


def predict_single_audio(audio_path):
    """Predice un archivo de audio específico."""
    start_time = time.time()

    audio_path = Path(audio_path)

    if not audio_path.exists():
        print(f"Error: No se encontró el archivo {audio_path}")
        return

    print(f"\nPrediciendo: {audio_path}")
    print("=" * 70)

    with timer("Inferencia audio"):
        result = predict_audio(str(audio_path))

    print(f"\nResultados (Ensemble de {N_MODELS} modelos con Soft Voting):")
    print(f"\n  Plate Thickness: {result['plate']}")
    print(
        f"    Probabilidades: {dict(zip(plate_encoder.classes_, result['probs_plate'].round(3)))}"
    )

    print(f"\n  Electrode: {result['electrode']}")
    print(
        f"    Probabilidades: {dict(zip(electrode_encoder.classes_, result['probs_electrode'].round(3)))}"
    )

    print(f"\n  Type of Current: {result['current']}")
    print(
        f"    Probabilidades: {dict(zip(current_type_encoder.classes_, result['probs_current'].round(3)))}"
    )

    # Guardar resultados en infer.json
    inference_result = {
        "mode": "single_audio",
        "audio_path": str(audio_path),
        "n_models": N_MODELS,
        "voting_method": "soft",
        "predictions": {
            "plate_thickness": result["plate"],
            "electrode": result["electrode"],
            "current_type": result["current"],
        },
        "probabilities": {
            "plate_thickness": dict(
                zip(
                    plate_encoder.classes_.tolist(),
                    result["probs_plate"].round(4).tolist(),
                )
            ),
            "electrode": dict(
                zip(
                    electrode_encoder.classes_.tolist(),
                    result["probs_electrode"].round(4).tolist(),
                )
            ),
            "current_type": dict(
                zip(
                    current_type_encoder.classes_.tolist(),
                    result["probs_current"].round(4).tolist(),
                )
            ),
        },
    }

    # Calcular tiempo de ejecución
    elapsed_time = time.time() - start_time
    print(f"\nTiempo de ejecución: {elapsed_time:.2f}s")

    save_inference_result(inference_result, elapsed_time=elapsed_time)


# ============= Main =============

if __name__ == "__main__":
    # args ya fue parseado al inicio del script
    if args.audio:
        predict_single_audio(args.audio)
    elif args.evaluar:
        evaluate_blind_set()
    else:
        show_random_predictions(n_samples=args.n)
