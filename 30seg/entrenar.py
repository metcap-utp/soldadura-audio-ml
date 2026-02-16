"""
Entrenamiento de modelos para clasificación SMAW.

Entrena K modelos usando K-Fold CV y los guarda para hacer voting.
Cada modelo ve diferentes datos de validación, lo que aumenta la diversidad.

Fuente: "Ensemble Methods" (Dietterich, 2000)
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, Dataset

# Añadir carpeta raíz al path para importar modelo.py
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from modelo import SMAWXVectorModel

warnings.filterwarnings("ignore")

# ============= Configuración =============
N_FOLDS = 5
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SWA_START = 5

# Directorios
SCRIPT_DIR = Path("30seg")
VGGISH_MODEL_DIR = Path("vggish")
MODELS_DIR = SCRIPT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Cargar modelo VGGish
print(f"Cargando modelo VGGish desde {VGGISH_MODEL_DIR}...")
vggish_model = hub.load(str(VGGISH_MODEL_DIR))
print("Modelo VGGish cargado correctamente.")


# ============= Funciones auxiliares =============


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

    return np.stack(embeddings_list, axis=0)


def collate_fn_pad(batch):
    """Padding de secuencias a longitud máxima del batch."""
    embeddings, labels_plate, labels_electrode, labels_current = zip(*batch)
    max_len = max(emb.shape[0] for emb in embeddings)

    padded_embeddings = []
    for emb in embeddings:
        if emb.shape[0] < max_len:
            pad = torch.zeros(max_len - emb.shape[0], emb.shape[1])
            emb = torch.cat([emb, pad], dim=0)
        padded_embeddings.append(emb)

    return (
        torch.stack(padded_embeddings),
        torch.stack(list(labels_plate)),
        torch.stack(list(labels_electrode)),
        torch.stack(list(labels_current)),
    )


class AudioDataset(Dataset):
    """Dataset con embeddings pre-extraídos."""

    def __init__(self, embeddings_list, labels_plate, labels_electrode, labels_current):
        self.embeddings_list = embeddings_list
        self.labels_plate = labels_plate
        self.labels_electrode = labels_electrode
        self.labels_current = labels_current

    def __len__(self):
        return len(self.embeddings_list)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings_list[idx], dtype=torch.float32),
            torch.tensor(self.labels_plate[idx], dtype=torch.long),
            torch.tensor(self.labels_electrode[idx], dtype=torch.long),
            torch.tensor(self.labels_current[idx], dtype=torch.long),
        )


def train_one_fold(
    fold_idx,
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    class_weights,
    encoders,
    device,
):
    """Entrena un fold y guarda el mejor modelo."""

    plate_encoder, electrode_encoder, current_type_encoder = encoders

    # Crear datasets
    train_dataset = AudioDataset(
        train_embeddings,
        train_labels["plate"],
        train_labels["electrode"],
        train_labels["current"],
    )
    val_dataset = AudioDataset(
        val_embeddings,
        val_labels["plate"],
        val_labels["electrode"],
        val_labels["current"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad
    )

    # Crear modelo
    model = SMAWXVectorModel(
        feat_dim=128,
        xvector_dim=512,
        emb_dim=256,
        num_classes_espesor=len(plate_encoder.classes_),
        num_classes_electrodo=len(electrode_encoder.classes_),
        num_classes_corriente=len(current_type_encoder.classes_),
    ).to(device)

    # Criterios con class weights
    criterion_plate = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights["plate"]).to(device),
        label_smoothing=LABEL_SMOOTHING,
    )
    criterion_electrode = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights["electrode"]).to(device),
        label_smoothing=LABEL_SMOOTHING,
    )
    criterion_current = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights["current"]).to(device),
        label_smoothing=LABEL_SMOOTHING,
    )

    # Optimizador
    log_vars = nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [log_vars],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Schedulers
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = {}
    best_state_dict = None

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0

        for embeddings, labels_p, labels_e, labels_c in train_loader:
            embeddings = embeddings.to(device)
            labels_p = labels_p.to(device)
            labels_e = labels_e.to(device)
            labels_c = labels_c.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)

            # Multi-task loss con incertidumbre
            loss_p = criterion_plate(outputs["logits_espesor"], labels_p)
            loss_e = criterion_electrode(outputs["logits_electrodo"], labels_e)
            loss_c = criterion_current(outputs["logits_corriente"], labels_c)

            precision = torch.exp(-log_vars)
            loss = (
                precision[0] * loss_p
                + precision[1] * loss_e
                + precision[2] * loss_c
                + log_vars.sum()
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = {"plate": [], "electrode": [], "current": []}
        all_labels = {"plate": [], "electrode": [], "current": []}

        with torch.no_grad():
            for embeddings, labels_p, labels_e, labels_c in val_loader:
                embeddings = embeddings.to(device)
                labels_p = labels_p.to(device)
                labels_e = labels_e.to(device)
                labels_c = labels_c.to(device)

                outputs = model(embeddings)

                loss_p = criterion_plate(outputs["logits_espesor"], labels_p)
                loss_e = criterion_electrode(outputs["logits_electrodo"], labels_e)
                loss_c = criterion_current(outputs["logits_corriente"], labels_c)
                val_loss += (loss_p + loss_e + loss_c).item()

                _, pred_p = outputs["logits_espesor"].max(1)
                _, pred_e = outputs["logits_electrodo"].max(1)
                _, pred_c = outputs["logits_corriente"].max(1)

                all_preds["plate"].extend(pred_p.cpu().numpy())
                all_preds["electrode"].extend(pred_e.cpu().numpy())
                all_preds["current"].extend(pred_c.cpu().numpy())
                all_labels["plate"].extend(labels_p.cpu().numpy())
                all_labels["electrode"].extend(labels_e.cpu().numpy())
                all_labels["current"].extend(labels_c.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Calcular métricas
        acc_p = np.mean(np.array(all_preds["plate"]) == np.array(all_labels["plate"]))
        acc_e = np.mean(
            np.array(all_preds["electrode"]) == np.array(all_labels["electrode"])
        )
        acc_c = np.mean(
            np.array(all_preds["current"]) == np.array(all_labels["current"])
        )

        f1_p = f1_score(all_labels["plate"], all_preds["plate"], average="weighted")
        f1_e = f1_score(
            all_labels["electrode"], all_preds["electrode"], average="weighted"
        )
        f1_c = f1_score(all_labels["current"], all_preds["current"], average="weighted")

        # SWA update
        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Early stopping y guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_metrics = {
                "acc_plate": acc_p,
                "acc_electrode": acc_e,
                "acc_current": acc_c,
                "f1_plate": f1_p,
                "f1_electrode": f1_e,
                "f1_current": f1_c,
            }
            # Guardar state dict del mejor modelo
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

    # Guardar el mejor modelo de este fold
    model_path = MODELS_DIR / f"model_fold_{fold_idx}.pth"
    torch.save(best_state_dict, model_path)

    print(
        f"  Fold {fold_idx + 1}: Plate={best_metrics['acc_plate']:.4f} | "
        f"Electrode={best_metrics['acc_electrode']:.4f} | "
        f"Current={best_metrics['acc_current']:.4f} | "
        f"Guardado: {model_path.name}"
    )

    return best_metrics


def ensemble_predict(models, embeddings, device):
    """Realiza predicciones usando voting de múltiples modelos."""
    all_logits_plate = []
    all_logits_electrode = []
    all_logits_current = []

    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(embeddings.to(device))
            all_logits_plate.append(outputs["logits_espesor"])
            all_logits_electrode.append(outputs["logits_electrodo"])
            all_logits_current.append(outputs["logits_corriente"])

    # Soft voting: promediar logits (antes de softmax)
    avg_logits_plate = torch.stack(all_logits_plate).mean(dim=0)
    avg_logits_electrode = torch.stack(all_logits_electrode).mean(dim=0)
    avg_logits_current = torch.stack(all_logits_current).mean(dim=0)

    # Predicciones finales
    pred_plate = avg_logits_plate.argmax(dim=1)
    pred_electrode = avg_logits_electrode.argmax(dim=1)
    pred_current = avg_logits_current.argmax(dim=1)

    return pred_plate, pred_electrode, pred_current


# ============= Main =============

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar todos los datos (train + test)
    train_data = pd.read_csv(SCRIPT_DIR / "train.csv")
    test_data = pd.read_csv(SCRIPT_DIR / "test.csv")
    all_data = pd.concat([train_data, test_data], ignore_index=True)

    print(f"Total de muestras: {len(all_data)}")

    # Preparar paths
    all_data["Audio Path"] = all_data["Audio Path"].apply(lambda x: str(SCRIPT_DIR / x))

    # Encoders
    plate_encoder = LabelEncoder()
    electrode_encoder = LabelEncoder()
    current_type_encoder = LabelEncoder()

    plate_encoder.fit(all_data["Plate Thickness"])
    electrode_encoder.fit(all_data["Electrode"])
    current_type_encoder.fit(all_data["Type of Current"])

    all_data["Plate Encoded"] = plate_encoder.transform(all_data["Plate Thickness"])
    all_data["Electrode Encoded"] = electrode_encoder.transform(all_data["Electrode"])
    all_data["Current Encoded"] = current_type_encoder.transform(
        all_data["Type of Current"]
    )

    # Extraer todos los embeddings una sola vez
    print("\nExtrayendo embeddings VGGish de todas las muestras...")
    all_embeddings = []
    paths = all_data["Audio Path"].values
    for i, path in enumerate(paths):
        if i % 100 == 0:
            print(f"  Procesando {i}/{len(paths)}...")
        emb = extract_vggish_embeddings(path)
        all_embeddings.append(emb)

    print(f"Embeddings extraídos: {len(all_embeddings)}")

    # Preparar arrays
    y_plate = all_data["Plate Encoded"].values
    y_electrode = all_data["Electrode Encoded"].values
    y_current = all_data["Current Encoded"].values

    # Crear etiqueta combinada para stratification
    y_stratify = y_electrode

    # ============= FASE 1: Entrenar K modelos =============
    print(f"\n{'=' * 70}")
    print(f"FASE 1: ENTRENAMIENTO DE {N_FOLDS} MODELOS (K-Fold)")
    print(f"{'=' * 70}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(all_embeddings, y_stratify)
    ):
        print(f"\nFold {fold_idx + 1}/{N_FOLDS}")
        print(f"  Train: {len(train_idx)} muestras | Val: {len(val_idx)} muestras")

        # Separar datos
        train_embeddings = [all_embeddings[i] for i in train_idx]
        val_embeddings = [all_embeddings[i] for i in val_idx]

        train_labels = {
            "plate": y_plate[train_idx],
            "electrode": y_electrode[train_idx],
            "current": y_current[train_idx],
        }
        val_labels = {
            "plate": y_plate[val_idx],
            "electrode": y_electrode[val_idx],
            "current": y_current[val_idx],
        }

        # Class weights del fold de entrenamiento
        class_weights = {
            "plate": compute_class_weight(
                "balanced",
                classes=np.unique(train_labels["plate"]),
                y=train_labels["plate"],
            ),
            "electrode": compute_class_weight(
                "balanced",
                classes=np.unique(train_labels["electrode"]),
                y=train_labels["electrode"],
            ),
            "current": compute_class_weight(
                "balanced",
                classes=np.unique(train_labels["current"]),
                y=train_labels["current"],
            ),
        }

        # Entrenar fold
        metrics = train_one_fold(
            fold_idx,
            train_embeddings,
            train_labels,
            val_embeddings,
            val_labels,
            class_weights,
            (plate_encoder, electrode_encoder, current_type_encoder),
            device,
        )
        fold_metrics.append(metrics)

    # ============= FASE 2: Evaluar Ensemble =============
    print(f"\n{'=' * 70}")
    print("FASE 2: EVALUACIÓN DEL ENSEMBLE (Soft Voting)")
    print(f"{'=' * 70}")

    # Cargar todos los modelos
    models = []
    for fold_idx in range(N_FOLDS):
        model = SMAWXVectorModel(
            feat_dim=128,
            xvector_dim=512,
            emb_dim=256,
            num_classes_espesor=len(plate_encoder.classes_),
            num_classes_electrodo=len(electrode_encoder.classes_),
            num_classes_corriente=len(current_type_encoder.classes_),
        ).to(device)
        model.load_state_dict(torch.load(MODELS_DIR / f"model_fold_{fold_idx}.pth"))
        model.eval()
        models.append(model)

    print(f"Cargados {len(models)} modelos del ensemble")

    # Evaluar en todo el dataset
    all_preds = {"plate": [], "electrode": [], "current": []}
    all_labels = {"plate": [], "electrode": [], "current": []}

    # Crear dataset completo
    full_dataset = AudioDataset(all_embeddings, y_plate, y_electrode, y_current)
    full_loader = DataLoader(
        full_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad
    )

    print("Evaluando ensemble en todo el dataset...")
    for embeddings, labels_p, labels_e, labels_c in full_loader:
        pred_p, pred_e, pred_c = ensemble_predict(models, embeddings, device)

        all_preds["plate"].extend(pred_p.cpu().numpy())
        all_preds["electrode"].extend(pred_e.cpu().numpy())
        all_preds["current"].extend(pred_c.cpu().numpy())
        all_labels["plate"].extend(labels_p.numpy())
        all_labels["electrode"].extend(labels_e.numpy())
        all_labels["current"].extend(labels_c.numpy())

    # Calcular métricas del ensemble
    acc_p = np.mean(np.array(all_preds["plate"]) == np.array(all_labels["plate"]))
    acc_e = np.mean(
        np.array(all_preds["electrode"]) == np.array(all_labels["electrode"])
    )
    acc_c = np.mean(np.array(all_preds["current"]) == np.array(all_labels["current"]))

    f1_p = f1_score(all_labels["plate"], all_preds["plate"], average="weighted")
    f1_e = f1_score(all_labels["electrode"], all_preds["electrode"], average="weighted")
    f1_c = f1_score(all_labels["current"], all_preds["current"], average="weighted")

    prec_p = precision_score(
        all_labels["plate"], all_preds["plate"], average="weighted"
    )
    prec_e = precision_score(
        all_labels["electrode"], all_preds["electrode"], average="weighted"
    )
    prec_c = precision_score(
        all_labels["current"], all_preds["current"], average="weighted"
    )

    rec_p = recall_score(all_labels["plate"], all_preds["plate"], average="weighted")
    rec_e = recall_score(
        all_labels["electrode"], all_preds["electrode"], average="weighted"
    )
    rec_c = recall_score(
        all_labels["current"], all_preds["current"], average="weighted"
    )

    # Promedios K-Fold individuales
    avg_acc_p = np.mean([m["acc_plate"] for m in fold_metrics])
    avg_acc_e = np.mean([m["acc_electrode"] for m in fold_metrics])
    avg_acc_c = np.mean([m["acc_current"] for m in fold_metrics])

    print(f"\n{'=' * 70}")
    print("RESULTADOS FINALES")
    print(f"{'=' * 70}")

    print("\nMétricas individuales por fold (promedio):")
    print(f"  Plate:     {avg_acc_p:.4f}")
    print(f"  Electrode: {avg_acc_e:.4f}")
    print(f"  Current:   {avg_acc_c:.4f}")

    print(f"\nMétricas del ENSEMBLE (Soft Voting, {N_FOLDS} modelos):")
    print(
        f"  Plate:     Acc={acc_p:.4f} | F1={f1_p:.4f} | Prec={prec_p:.4f} | Rec={rec_p:.4f}"
    )
    print(
        f"  Electrode: Acc={acc_e:.4f} | F1={f1_e:.4f} | Prec={prec_e:.4f} | Rec={rec_e:.4f}"
    )
    print(
        f"  Current:   Acc={acc_c:.4f} | F1={f1_c:.4f} | Prec={prec_c:.4f} | Rec={rec_c:.4f}"
    )

    print(f"\nMejora del Ensemble vs Promedio Individual:")
    print(f"  Plate:     {acc_p - avg_acc_p:+.4f}")
    print(f"  Electrode: {acc_e - avg_acc_e:+.4f}")
    print(f"  Current:   {acc_c - avg_acc_c:+.4f}")

    print(f"\n{'=' * 70}")
    print("REPORTES DE CLASIFICACIÓN")
    print(f"{'=' * 70}")

    print("\n--- Plate Thickness ---")
    print(
        classification_report(
            all_labels["plate"],
            all_preds["plate"],
            target_names=plate_encoder.classes_,
            zero_division=0,
        )
    )

    print("\n--- Electrode Type ---")
    print(
        classification_report(
            all_labels["electrode"],
            all_preds["electrode"],
            target_names=electrode_encoder.classes_,
            zero_division=0,
        )
    )

    print("\n--- Type of Current ---")
    print(
        classification_report(
            all_labels["current"],
            all_preds["current"],
            target_names=current_type_encoder.classes_,
            zero_division=0,
        )
    )

    print(f"\n{'=' * 70}")
    print("MATRICES DE CONFUSIÓN")
    print(f"{'=' * 70}")

    print("\nPlate Thickness:")
    print(confusion_matrix(all_labels["plate"], all_preds["plate"]))
    print(f"Clases: {plate_encoder.classes_}")

    print("\nElectrode Type:")
    print(confusion_matrix(all_labels["electrode"], all_preds["electrode"]))
    print(f"Clases: {electrode_encoder.classes_}")

    print("\nType of Current:")
    print(confusion_matrix(all_labels["current"], all_preds["current"]))
    print(f"Clases: {current_type_encoder.classes_}")

    # Guardar resultados (acumulativo)
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_folds": N_FOLDS,
            "random_seed": RANDOM_SEED,
            "voting_method": "soft",
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
        },
        "fold_results": fold_metrics,
        "results": {
            "plate": {
                "accuracy": round(acc_p, 4),
                "f1": round(f1_p, 4),
                "precision": round(prec_p, 4),
                "recall": round(rec_p, 4),
            },
            "electrode": {
                "accuracy": round(acc_e, 4),
                "f1": round(f1_e, 4),
                "precision": round(prec_e, 4),
                "recall": round(rec_e, 4),
            },
            "current": {
                "accuracy": round(acc_c, 4),
                "f1": round(f1_c, 4),
                "precision": round(prec_c, 4),
                "recall": round(rec_c, 4),
            },
        },
        "improvement_vs_individual": {
            "plate": round(acc_p - avg_acc_p, 4),
            "electrode": round(acc_e - avg_acc_e, 4),
            "current": round(acc_c - avg_acc_c, 4),
        },
    }

    # Cargar historial existente o crear nuevo
    results_path = SCRIPT_DIR / "resultados.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = [history]  # Convertir formato antiguo a lista
    else:
        history = []
    
    history.append(new_entry)
    
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResultados guardados en: {results_path} (entrada #{len(history)})")
    print(f"Modelos guardados en: {MODELS_DIR}/")
