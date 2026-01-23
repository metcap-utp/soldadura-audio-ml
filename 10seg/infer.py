"""
Script de predicción usando Soft Voting.

Carga los 5 modelos entrenados con K-Fold y combina sus predicciones
promediando logits antes de aplicar argmax (soft voting).

Los audios se segmentan ON-THE-FLY según la duración del directorio
(5seg, 10seg, 30seg) - NO hay archivos segmentados en disco.

Uso:
    python infer.py                    # Muestra 10 predicciones aleatorias
    python infer.py --audio ruta.wav   # Predice un archivo específico
    python infer.py --evaluar          # Evalúa en conjunto holdout (vida real)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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

# Definir directorios
SCRIPT_DIR = Path(__file__).parent
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"
MODELS_DIR = SCRIPT_DIR / "models"
INFER_JSON = SCRIPT_DIR / "infer.json"

# Número de modelos
N_MODELS = 5

# Duración de segmento basada en el nombre del directorio
SEGMENT_DURATION = get_script_segment_duration(Path(__file__))

# Cargar modelo VGGish de TensorFlow Hub
print(f"Cargando modelo VGGish desde TensorFlow Hub...")
vggish_model = hub.load(VGGISH_MODEL_URL)
print("Modelo VGGish cargado correctamente.")
print(f"Duración de segmento: {SEGMENT_DURATION}s")


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
        hop_ratio=0.5,
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

train_data = pd.read_csv(SCRIPT_DIR / "train.csv")

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
    """Carga los 5 modelos del ensemble."""
    models = []

    for fold in range(N_MODELS):
        model_path = MODELS_DIR / f"model_fold_{fold}.pth"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo {model_path}. "
                "Ejecuta entrenar_ensemble.py primero."
            )

        model = create_model()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    print(f"Cargados {len(models)} modelos desde {MODELS_DIR}")
    return models


# Cargar todos los modelos
ensemble_models = load_ensemble_models()


# ============= Guardar resultados de inferencia =============


def save_inference_result(result_data):
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

    # Agregar timestamp al resultado
    result_data["timestamp"] = datetime.now().isoformat()

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
        row = f"| **{cls}** | " + " | ".join(str(cm[i][j]) for j in range(len(classes))) + " |"
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
- Número de muestras (holdout): {results['n_samples']}
- Número de modelos (ensemble): {results['n_models']}
- Método de votación: {results['voting_method']}

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | {results['accuracy']['plate_thickness']*100:.2f}% | {results['macro_f1']['plate_thickness']:.4f} |
| Electrode Type | {results['accuracy']['electrode']*100:.2f}% | {results['macro_f1']['electrode']:.4f} |
| Current Type | {results['accuracy']['current_type']*100:.2f}% | {results['macro_f1']['current_type']:.4f} |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** {results['accuracy']['plate_thickness']*100:.2f}%
- **Macro F1-Score:** {results['macro_f1']['plate_thickness']:.4f}

### Confusion Matrix

{format_confusion_matrix_markdown(results['confusion_matrices']['plate_thickness'], results['classes']['plate_thickness'])}

### Classification Report
"""
    # Agregar métricas por clase para plate
    cr_plate = results['classification_reports']['plate_thickness']
    doc += "\n| Clase | Precision | Recall | F1-Score | Support |\n"
    doc += "|-------|-----------|--------|----------|--------|\n"
    for cls in results['classes']['plate_thickness']:
        metrics = cr_plate.get(cls, {})
        doc += f"| {cls} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('f1-score', 0):.4f} | {int(metrics.get('support', 0))} |\n"

    doc += f"""
---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** {results['accuracy']['electrode']*100:.2f}%
- **Macro F1-Score:** {results['macro_f1']['electrode']:.4f}

### Confusion Matrix

{format_confusion_matrix_markdown(results['confusion_matrices']['electrode'], results['classes']['electrode'])}

### Classification Report
"""
    # Agregar métricas por clase para electrode
    cr_electrode = results['classification_reports']['electrode']
    doc += "\n| Clase | Precision | Recall | F1-Score | Support |\n"
    doc += "|-------|-----------|--------|----------|--------|\n"
    for cls in results['classes']['electrode']:
        metrics = cr_electrode.get(cls, {})
        doc += f"| {cls} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('f1-score', 0):.4f} | {int(metrics.get('support', 0))} |\n"

    doc += f"""
---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** {results['accuracy']['current_type']*100:.2f}%
- **Macro F1-Score:** {results['macro_f1']['current_type']:.4f}

### Confusion Matrix

{format_confusion_matrix_markdown(results['confusion_matrices']['current_type'], results['classes']['current_type'])}

### Classification Report
"""
    # Agregar métricas por clase para current
    cr_current = results['classification_reports']['current_type']
    doc += "\n| Clase | Precision | Recall | F1-Score | Support |\n"
    doc += "|-------|-----------|--------|----------|--------|\n"
    for cls in results['classes']['current_type']:
        metrics = cr_current.get(cls, {})
        doc += f"| {cls} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('f1-score', 0):.4f} | {int(metrics.get('support', 0))} |\n"

    doc += """
---

## Notas

- Las métricas se calcularon sobre el conjunto **holdout** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
"""

    # Guardar documento
    metrics_file = SCRIPT_DIR / "METRICAS.md"
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


def evaluate_holdout_set():
    """Evalúa el ensemble en el conjunto holdout (validación vida real)."""
    holdout_csv = SCRIPT_DIR / "holdout.csv"
    if not holdout_csv.exists():
        print("No se encontró holdout.csv. Ejecuta generar_splits.py primero.")
        return None

    holdout_df = pd.read_csv(holdout_csv)
    print(
        f"\nEvaluando ensemble en {len(holdout_df)} segmentos de HOLDOUT (vida real)..."
    )
    print(f"Duración de segmento: {SEGMENT_DURATION}s")

    # Listas para almacenar predicciones y etiquetas reales
    y_true_plate, y_pred_plate = [], []
    y_true_electrode, y_pred_electrode = [], []
    y_true_current, y_pred_current = [], []

    for idx, row in holdout_df.iterrows():
        if idx % 100 == 0:
            print(f"  Procesando {idx}/{len(holdout_df)}...")

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
    print("RESULTADOS DEL ENSEMBLE EN CONJUNTO HOLDOUT (VIDA REAL)")
    print("=" * 70)

    acc_plate = accuracy_score(y_true_plate, y_pred_plate)
    acc_electrode = accuracy_score(y_true_electrode, y_pred_electrode)
    acc_current = accuracy_score(y_true_current, y_pred_current)

    print(f"\nAccuracy:")
    print(f"  Plate Thickness:  {acc_plate * 100:.2f}%")
    print(f"  Electrode:        {acc_electrode * 100:.2f}%")
    print(f"  Type of Current:  {acc_current * 100:.2f}%")

    # Calcular Macro F1
    f1_plate = f1_score(y_true_plate, y_pred_plate, average='macro')
    f1_electrode = f1_score(y_true_electrode, y_pred_electrode, average='macro')
    f1_current = f1_score(y_true_current, y_pred_current, average='macro')

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
    cm_plate = confusion_matrix(y_true_plate, y_pred_plate, labels=plate_encoder.classes_)
    print(cm_plate)
    print(f"Clases: {plate_encoder.classes_}")

    print("\nElectrode Type:")
    cm_electrode = confusion_matrix(y_true_electrode, y_pred_electrode, labels=electrode_encoder.classes_)
    print(cm_electrode)
    print(f"Clases: {electrode_encoder.classes_}")

    print("\nType of Current:")
    cm_current = confusion_matrix(y_true_current, y_pred_current, labels=current_type_encoder.classes_)
    print(cm_current)
    print(f"Clases: {current_type_encoder.classes_}")

    # Crear diccionario de resultados
    results = {
        "mode": "holdout_evaluation",
        "segment_duration": SEGMENT_DURATION,
        "n_samples": len(holdout_df),
        "n_models": N_MODELS,
        "voting_method": "soft",
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

    # Guardar resultados en infer.json
    save_inference_result(results)

    # Generar documento de métricas
    generate_metrics_document(results)

    return results


def show_random_predictions(n_samples=10):
    """Muestra predicciones aleatorias del conjunto holdout."""
    holdout_csv = SCRIPT_DIR / "holdout.csv"
    if not holdout_csv.exists():
        print("No se encontró holdout.csv. Ejecuta generar_splits.py primero.")
        return

    holdout_df = pd.read_csv(holdout_csv)
    num_samples = min(n_samples, len(holdout_df))
    samples = holdout_df.sample(n=num_samples, random_state=None)

    print(f"\n{'=' * 80}")
    print(f"  PREDICCIONES ALEATORIAS ({num_samples} muestras)")
    print(f"{'=' * 80}")
    print(f"\n  {'Archivo':<25} {'Plate':<12} {'Electrode':<10} {'Current':<8}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 10} {'-' * 8}")

    correctas_plate = 0
    correctas_electrode = 0
    correctas_current = 0
    correctas_todas = 0

    for _, row in samples.iterrows():
        audio_path = SCRIPT_DIR / row["Audio Path"]
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
    save_inference_result(inference_result)


def predict_single_audio(audio_path):
    """Predice un archivo de audio específico."""
    audio_path = Path(audio_path)

    if not audio_path.exists():
        print(f"Error: No se encontró el archivo {audio_path}")
        return

    print(f"\nPrediciendo: {audio_path}")
    print("=" * 70)

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
    save_inference_result(inference_result)


# ============= Main =============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicción de soldadura SMAW usando Ensemble con Soft Voting"
    )
    parser.add_argument(
        "--audio", type=str, help="Ruta a un archivo de audio para predecir"
    )
    parser.add_argument(
        "--evaluar",
        action="store_true",
        help="Evaluar ensemble en conjunto holdout (vida real)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Número de muestras aleatorias a mostrar (default: 10)",
    )

    args = parser.parse_args()

    if args.audio:
        predict_single_audio(args.audio)
    elif args.evaluar:
        evaluate_holdout_set()
    else:
        show_random_predictions(n_samples=args.n)
