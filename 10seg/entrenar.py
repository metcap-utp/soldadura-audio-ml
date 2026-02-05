"""
Entrenamiento de modelos para clasificación SMAW.

Entrena K modelos usando K-Fold CV estratificado por GRUPOS (sesiones)
y los guarda para hacer voting. Cada modelo ve diferentes datos de
validación, lo que aumenta la diversidad.

IMPORTANTE: Se usa StratifiedGroupKFold para garantizar que:
- Todos los segmentos de una misma sesión van al mismo fold
- Esto evita data leakage por grabaciones similares

Los audios se segmentan ON-THE-FLY según la duración del directorio
(5seg, 10seg, 30seg) - NO hay archivos segmentados en disco.

Fuente: "Ensemble Methods" (Dietterich, 2000)
"""

import argparse
import hashlib
import json
import pickle
import sys
import time
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, Dataset

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

warnings.filterwarnings("ignore")


# ============= Parseo de argumentos =============
def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos SMAW con K-Fold CV"
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        choices=[3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
        help="Número de folds para cross-validation (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Solapamiento entre segmentos como fracción (0-1). Ej: 0.5 = 50% (default: 0.0)",
    )
    return parser.parse_args()


# ============= Configuración =============
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SWA_START = 5

# Duración de segmento basada en el nombre del directorio (5seg -> 5.0)
SEGMENT_DURATION = get_script_segment_duration(Path(__file__))

# Solapamiento entre segmentos como fracción (0-1). Default: 0 (sin solapamiento)
OVERLAP_RATIO = 0.0
# Solapamiento entre segmentos en segundos (derivado de OVERLAP_RATIO)
OVERLAP_SECONDS = 0.0

# Directorios
SCRIPT_DIR = Path(__file__).parent
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"

# Cargar modelo VGGish
print(f"Cargando modelo VGGish desde TensorFlow Hub...")
vggish_model = hub.load(VGGISH_MODEL_URL)
print("Modelo VGGish cargado correctamente.")


# ============= Funciones auxiliares =============


def extract_session_from_path(audio_path: str) -> str:
    """Extrae el identificador de sesión del path del audio.

    El path tiene estructura: audio/Placa_Xmm/EXXXX/AC|DC/YYMMDD-HHMMSS_Audio/file.wav
    La sesión es la carpeta con formato YYMMDD-HHMMSS_Audio
    """
    parts = Path(audio_path).parts
    for part in parts:
        if part.endswith("_Audio"):
            return part
    # Fallback: usar el directorio padre
    return Path(audio_path).parent.name


# ============= Cache de Embeddings =============


def _overlap_tag() -> str:
    return str(OVERLAP_RATIO)


def get_embeddings_cache_dir() -> Path:
    """Directorio de cache de embeddings (siempre `embeddings_cache/`)."""
    cache_dir = SCRIPT_DIR / "embeddings_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_legacy_embeddings_cache_paths() -> list[Path]:
    """Rutas legacy posibles para reutilizar caches viejos sin borrarlos."""
    cache_dir = SCRIPT_DIR / "embeddings_cache"
    cache_dir.mkdir(exist_ok=True)
    tag = _overlap_tag()
    dur = float(SEGMENT_DURATION)
    return [
        # Nuevo/previo naming en raíz
        cache_dir / f"vggish_embeddings_{dur}s_overlap{tag}s.pkl",
        cache_dir / f"vggish_embeddings_{dur}s_overlap_{tag}.pkl",
        # Legacy sin overlap en nombre
        cache_dir / f"vggish_embeddings_{dur}s.pkl",
        # Legacy que quedó en subcarpeta overlap_* (de migraciones anteriores)
        cache_dir / f"overlap_{tag}s" / f"vggish_embeddings_{dur}s.pkl",
        cache_dir / f"overlap_{tag}s" / f"vggish_embeddings_{dur}s_overlap{tag}s.pkl",
    ]


def get_embeddings_cache_path() -> Path:
    """Obtiene la ruta del archivo de cache de embeddings."""
    cache_dir = get_embeddings_cache_dir()
    return (
        cache_dir
        / f"vggish_embeddings_{float(SEGMENT_DURATION)}s_overlap_{_overlap_tag()}.pkl"
    )


def compute_dataset_hash(paths: list, segment_indices: list) -> str:
    """Calcula un hash del dataset para detectar cambios."""
    data_str = f"dur={SEGMENT_DURATION}|overlap_ratio={OVERLAP_RATIO}|" + "".join(
        [f"{p}:{s}" for p, s in zip(paths, segment_indices)]
    )
    return hashlib.md5(data_str.encode()).hexdigest()


def _get_cache_overlap_ratio(cache_data: dict) -> float | None:
    """Normaliza overlap a fracción (0-1) desde distintas versiones de cache."""
    if cache_data is None:
        return None

    for k in ("overlap_ratio", "overlap"):
        v = cache_data.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None

    v = cache_data.get("overlap_seconds")
    if v is None:
        return None

    try:
        v = float(v)
    except Exception:
        return None

    # En este proyecto el overlap se especifica como fracción; caches antiguos
    # pueden tenerlo guardado en `overlap_seconds` por compatibilidad.
    if 0.0 <= v <= 1.0:
        return v

    dur = cache_data.get("segment_duration", SEGMENT_DURATION)
    try:
        dur = float(dur)
    except Exception:
        return None
    if dur <= 0:
        return None
    return v / dur


def load_embeddings_cache(paths: list, segment_indices: list) -> tuple:
    """Carga embeddings del cache si existe y es válido.

    Returns:
        tuple: (embeddings_list, success) donde success indica si se cargó del cache
    """
    new_cache_path = get_embeddings_cache_path()
    cache_paths = [new_cache_path, *get_legacy_embeddings_cache_paths()]
    cache_path = next((p for p in cache_paths if p.exists()), None)

    if cache_path is None:
        return None, False

    try:
        with timer("Cargar cache embeddings VGGish"):
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Verificaciones rápidas de configuración
            if cache_data.get("segment_duration") != SEGMENT_DURATION:
                print("  [CACHE] Duración no coincide, regenerando embeddings...")
                return None, False

            cache_overlap_ratio = _get_cache_overlap_ratio(cache_data)
            if cache_overlap_ratio is None or cache_overlap_ratio != OVERLAP_RATIO:
                print("  [CACHE] Overlap no coincide, regenerando embeddings...")
                return None, False

            # Verificar hash
            current_hash = compute_dataset_hash(paths, segment_indices)
            if cache_data.get("hash") != current_hash:
                print("  [CACHE] Hash no coincide, regenerando embeddings...")
                return None, False

        print(
            f"  [CACHE] Cargando {len(cache_data['embeddings'])} embeddings desde cache ({cache_path})"
        )

        # Si es legacy, migrar al nuevo naming sin borrar el original
        if cache_path != new_cache_path:
            try:
                with timer("Migrar cache legacy -> nuevo naming"):
                    with open(new_cache_path, "wb") as f:
                        pickle.dump(cache_data, f)
                print(f"  [CACHE] Migrado a: {new_cache_path}")
            except Exception as e:
                print(f"  [CACHE] No se pudo migrar cache legacy: {e}")
        return cache_data["embeddings"], True

    except Exception as e:
        print(f"  [CACHE] Error leyendo cache: {e}")
        return None, False


def save_embeddings_cache(embeddings: list, paths: list, segment_indices: list):
    """Guarda embeddings en cache."""
    cache_path = get_embeddings_cache_path()

    cache_data = {
        "hash": compute_dataset_hash(paths, segment_indices),
        "embeddings": embeddings,
        "segment_duration": SEGMENT_DURATION,
        "overlap_ratio": OVERLAP_RATIO,
        "overlap_seconds": OVERLAP_SECONDS,
        "created_at": datetime.now().isoformat(),
        "num_embeddings": len(embeddings),
    }

    with timer("Guardar cache embeddings VGGish"):
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

    print(f"  [CACHE] Guardados {len(embeddings)} embeddings en cache ({cache_path})")


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
    """Extrae embeddings VGGish de un archivo de audio completo.

    Esta función se mantiene para compatibilidad con inferencia.
    """
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
    models_dir,
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
    best_metrics = {
        "acc_plate": 0.0,
        "acc_electrode": 0.0,
        "acc_current": 0.0,
        "f1_plate": 0.0,
        "f1_electrode": 0.0,
        "f1_current": 0.0,
    }
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
    model_path = models_dir / f"model_fold_{fold_idx}.pth"
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
    # Iniciar timer
    start_time = time.time()

    # Parsear argumentos
    args = parse_args()
    N_FOLDS = args.k_folds
    RANDOM_SEED = args.seed
    OVERLAP_RATIO = float(args.overlap)
    OVERLAP_SECONDS = float(OVERLAP_RATIO) * float(SEGMENT_DURATION)

    # Crear directorio de modelos basado en k-folds (models/k-fold/)
    MODELS_BASE_DIR = SCRIPT_DIR / "models"
    MODELS_BASE_DIR.mkdir(exist_ok=True)
    MODELS_DIR = MODELS_BASE_DIR / f"k{N_FOLDS:02d}_overlap_{_overlap_tag()}"
    MODELS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}")
    print(f"CONFIGURACIÓN")
    print(f"{'=' * 70}")
    print(f"Dispositivo: {device}")
    print(f"Duración de segmento: {SEGMENT_DURATION}s")
    print(f"Solapamiento (fracción): {OVERLAP_RATIO}")
    print(f"K-Folds: {N_FOLDS}")
    print(f"Semilla: {RANDOM_SEED}")
    print(f"Modelos se guardarán en: {MODELS_DIR}/")

    with timer("Cargar CSVs (train/test)"):
        train_data = pd.read_csv(SCRIPT_DIR / "train.csv")
        test_data = pd.read_csv(SCRIPT_DIR / "test.csv")
        all_data = pd.concat([train_data, test_data], ignore_index=True)

    print(f"\nTotal de segmentos: {len(all_data)}")

    # Extraer sesión de cada path para agrupar en K-Fold
    all_data["Session"] = all_data["Audio Path"].apply(extract_session_from_path)
    print(f"Sesiones únicas: {all_data['Session'].nunique()}")

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

    with timer("Embeddings VGGish (cache/compute)"):
        print("\nExtrayendo embeddings VGGish de todos los segmentos...")
        paths = all_data["Audio Path"].values
        segment_indices = all_data["Segment Index"].values

        all_embeddings, loaded_from_cache = load_embeddings_cache(
            paths.tolist(), segment_indices.tolist()
        )

        if not loaded_from_cache:
            with timer("Calcular embeddings VGGish (compute)"):
                all_embeddings = []
                for i, (path, seg_idx) in enumerate(zip(paths, segment_indices)):
                    if i % 100 == 0:
                        print(f"  Procesando {i}/{len(paths)}...")
                    emb = extract_vggish_embeddings_from_segment(path, int(seg_idx))
                    all_embeddings.append(emb)

            save_embeddings_cache(
                all_embeddings, paths.tolist(), segment_indices.tolist()
            )

    print(f"Embeddings extraídos: {len(all_embeddings)}")

    # Preparar arrays
    y_plate = all_data["Plate Encoded"].values
    y_electrode = all_data["Electrode Encoded"].values
    y_current = all_data["Current Encoded"].values
    sessions = all_data["Session"].values

    # Crear etiqueta combinada para stratification
    y_stratify = y_electrode

    # ============= FASE 1: Entrenar K modelos =============
    print(f"\n{'=' * 70}")
    print(f"FASE 1: ENTRENAMIENTO DE {N_FOLDS} MODELOS (StratifiedGroupKFold)")
    print(f"{'=' * 70}")
    print("[INFO] Usando StratifiedGroupKFold")

    sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(all_embeddings, y_stratify, groups=sessions)
    ):
        # Verificar que sesiones no se mezclan
        train_sessions = set(sessions[train_idx])
        val_sessions = set(sessions[val_idx])
        assert len(train_sessions & val_sessions) == 0, "ERROR: Sesiones mezcladas!"

        print(f"\nFold {fold_idx + 1}/{N_FOLDS}")
        print(f"  Train: {len(train_idx)} segmentos ({len(train_sessions)} sesiones)")
        print(f"  Val: {len(val_idx)} segmentos ({len(val_sessions)} sesiones)")

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

        with timer(f"Entrenar fold {fold_idx + 1}/{N_FOLDS}"):
            metrics = train_one_fold(
                fold_idx,
                train_embeddings,
                train_labels,
                val_embeddings,
                val_labels,
                class_weights,
                (plate_encoder, electrode_encoder, current_type_encoder),
                device,
                MODELS_DIR,
            )
        fold_metrics.append(metrics)

    # ============= FASE 2: Evaluar Ensemble =============
    print(f"\n{'=' * 70}")
    print("FASE 2: EVALUACIÓN DEL ENSEMBLE (Soft Voting)")
    print(f"{'=' * 70}")

    with timer("Cargar modelos del ensemble"):
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

    with timer("Evaluación ensemble (Soft Voting)"):
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
    # Calcular tiempo de ejecución
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    elapsed_hours = elapsed_time / 3600

    print(
        f"\nTiempo de ejecución: {elapsed_time:.2f}s ({elapsed_minutes:.2f}min / {elapsed_hours:.2f}h)"
    )

    new_entry = {
        "id": f"{int(SEGMENT_DURATION)}seg_{N_FOLDS}fold_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "execution_time": {
            "seconds": round(elapsed_time, 2),
            "minutes": round(elapsed_minutes, 2),
            "hours": round(elapsed_hours, 4),
        },
        "config": {
            "segment_duration": SEGMENT_DURATION,
            "overlap_seconds": OVERLAP_SECONDS,
            "n_folds": N_FOLDS,
            "models_dir": str(MODELS_DIR.name),
            "random_seed": RANDOM_SEED,
            "voting_method": "soft",
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        },
        "data": {
            "total_segments": len(all_data),
            "unique_sessions": all_data["Session"].nunique(),
            "classes": {
                "plate": list(plate_encoder.classes_),
                "electrode": list(electrode_encoder.classes_),
                "current": list(current_type_encoder.classes_),
            },
        },
        "fold_results": fold_metrics,
        "ensemble_results": {
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
        "individual_avg": {
            "plate": round(avg_acc_p, 4),
            "electrode": round(avg_acc_e, 4),
            "current": round(avg_acc_c, 4),
        },
        "improvement_vs_individual": {
            "plate": round(acc_p - avg_acc_p, 4),
            "electrode": round(acc_e - avg_acc_e, 4),
            "current": round(acc_c - avg_acc_c, 4),
        },
    }

    # Cargar historial existente o crear nuevo
    results_path = SCRIPT_DIR / "results.json"
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
