"""
Script de predicción usando Soft Voting.

Carga los 5 modelos entrenados con K-Fold y combina sus predicciones
promediando logits antes de aplicar argmax (soft voting).

Uso:
    python 10seg/predecir.py                    # Muestra 10 predicciones aleatorias
    python 10seg/predecir.py --audio ruta.wav   # Predice un archivo específico
    python 10seg/predecir.py --evaluar          # Evalúa en todo el conjunto de test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Añadir carpeta raíz al path para importar modelo.py
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from modelo import SMAWXVectorModel

# Definir directorios
SCRIPT_DIR = Path("10seg")
VGGISH_MODEL_DIR = Path("vggish")
MODELS_DIR = SCRIPT_DIR / "models"

# Número de modelos
N_MODELS = 5

# Cargar modelo VGGish de TensorFlow Hub
print(f"Cargando modelo VGGish desde {VGGISH_MODEL_DIR}...")
vggish_model = hub.load(str(VGGISH_MODEL_DIR))
print("Modelo VGGish cargado correctamente.")


# ============= Extracción de embeddings VGGish =============


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


def predict_audio(audio_path):
    """Predice las tres etiquetas de un archivo de audio usando ensemble."""
    embeddings = extract_vggish_embeddings(audio_path)
    embeddings_tensor = (
        torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    )
    return predict_ensemble(embeddings_tensor)


# ============= Funciones de Evaluación =============


def evaluate_test_set():
    """Evalúa el ensemble en todo el conjunto de test."""
    test_csv = SCRIPT_DIR / "test.csv"
    if not test_csv.exists():
        print("No se encontró test.csv. Ejecuta generar_splits.py primero.")
        return

    test_df = pd.read_csv(test_csv)
    print(f"\nEvaluando ensemble en {len(test_df)} muestras de test...")

    # Listas para almacenar predicciones y etiquetas reales
    y_true_plate, y_pred_plate = [], []
    y_true_electrode, y_pred_electrode = [], []
    y_true_current, y_pred_current = [], []

    for idx, row in test_df.iterrows():
        if idx % 100 == 0:
            print(f"  Procesando {idx}/{len(test_df)}...")

        audio_path = SCRIPT_DIR / row["Audio Path"]

        # Etiquetas reales
        y_true_plate.append(row["Plate Thickness"])
        y_true_electrode.append(row["Electrode"])
        y_true_current.append(row["Type of Current"])

        # Predicciones
        result = predict_audio(str(audio_path))
        y_pred_plate.append(result["plate"])
        y_pred_electrode.append(result["electrode"])
        y_pred_current.append(result["current"])

    # Calcular métricas
    print("\n" + "=" * 70)
    print("RESULTADOS DEL ENSEMBLE EN CONJUNTO DE TEST")
    print("=" * 70)

    acc_plate = accuracy_score(y_true_plate, y_pred_plate)
    acc_electrode = accuracy_score(y_true_electrode, y_pred_electrode)
    acc_current = accuracy_score(y_true_current, y_pred_current)

    print(f"\nAccuracy:")
    print(f"  Plate Thickness:  {acc_plate * 100:.2f}%")
    print(f"  Electrode:        {acc_electrode * 100:.2f}%")
    print(f"  Type of Current:  {acc_current * 100:.2f}%")

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
    print(confusion_matrix(y_true_plate, y_pred_plate))
    print(f"Clases: {plate_encoder.classes_}")

    print("\nElectrode Type:")
    print(confusion_matrix(y_true_electrode, y_pred_electrode))
    print(f"Clases: {electrode_encoder.classes_}")

    print("\nType of Current:")
    print(confusion_matrix(y_true_current, y_pred_current))
    print(f"Clases: {current_type_encoder.classes_}")


def show_random_predictions(n_samples=10):
    """Muestra predicciones aleatorias del conjunto de test."""
    test_csv = SCRIPT_DIR / "test.csv"
    if not test_csv.exists():
        print("No se encontró test.csv. Ejecuta generar_splits.py primero.")
        return

    test_df = pd.read_csv(test_csv)
    num_samples = min(n_samples, len(test_df))
    test_samples = test_df.sample(n=num_samples, random_state=None)

    print(f"\n{'='*80}")
    print(f"  PREDICCIONES ALEATORIAS ({num_samples} muestras)")
    print(f"{'='*80}")
    print(f"\n  {'Archivo':<25} {'Plate':<12} {'Electrode':<10} {'Current':<8}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*8}")

    correctas_plate = 0
    correctas_electrode = 0
    correctas_current = 0
    correctas_todas = 0

    for _, row in test_samples.iterrows():
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
        print(f"    Pred:      {result['plate']:<12} {result['electrode']:<10} {result['current']:<8}")
        print(f"    Status:    {sym_plate:<12} {sym_electrode:<10} {sym_current:<8}")
        print(f"    Conf:      {result['probs_plate'].max():.2f}         {result['probs_electrode'].max():.2f}       {result['probs_current'].max():.2f}")

    print(f"\n{'='*80}")
    print(f"  RESUMEN ({N_MODELS} modelos con Soft Voting)")
    print(f"{'='*80}")
    print(f"\n  Plate Thickness:  {correctas_plate:>2}/{num_samples} = {correctas_plate / num_samples * 100:>6.2f}%")
    print(f"  Electrode:        {correctas_electrode:>2}/{num_samples} = {correctas_electrode / num_samples * 100:>6.2f}%")
    print(f"  Type of Current:  {correctas_current:>2}/{num_samples} = {correctas_current / num_samples * 100:>6.2f}%")
    print(f"  Todas correctas:  {correctas_todas:>2}/{num_samples} = {correctas_todas / num_samples * 100:>6.2f}%")
    print()


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
        help="Evaluar ensemble en todo el conjunto de test",
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
        evaluate_test_set()
    else:
        show_random_predictions(n_samples=args.n)
