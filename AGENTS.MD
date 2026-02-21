# AGENTS.md - Guía para Agentes de Código

## Información del Proyecto

**Proyecto**: Clasificación de Audio SMAW (Soldadura por Arco Eléctrico)  
**Lenguaje**: Python 3.10+  
**Framework**: PyTorch + Librosa + Scikit-learn  
**Idioma**: Español (código), documentación en español  

## Estructura del Proyecto

```
/home/luis/projects/vggish-backbone/
├── entrenar.py              # Script principal de entrenamiento (X-Vector)
├── entrenar_ecapa.py        # Script de entrenamiento ECAPA-TDNN
├── entrenar_feedforward.py  # Script de entrenamiento FeedForward
├── inferir.py               # Script de inferencia/evaluación
├── generar_splits.py        # Generación de splits de datos
├── modelo_xvector.py        # Arquitectura X-Vector (PyTorch)
├── modelo_ecapa.py          # Arquitectura ECAPA-TDNN
├── modelo_feedforward.py    # Arquitectura FeedForward
├── modelo_multitask.py      # Wrappers multi-task
├── scripts/                 # Scripts de análisis y visualización
├── utils/                   # Utilidades compartidas
└── {N}seg/                  # Datos por duración (01seg, 05seg, 10seg, etc.)
```

## Comandos de Desarrollo

### Scripts Principales

```bash
# Generar splits de datos
python generar_splits.py --duration 5 --overlap 0.5

# Entrenar modelo X-Vector
python entrenar.py --duration 5 --overlap 0.5 --k-folds 10

# Entrenar modelo ECAPA-TDNN
python entrenar_ecapa.py --duration 5 --overlap 0.5 --k-folds 10

# Entrenar modelo FeedForward
python entrenar_feedforward.py --duration 5 --overlap 0.5 --k-folds 10

# Inferencia/evaluación
python inferir.py --duration 5 --overlap 0.5 --evaluar

# Scripts de visualización
python scripts/graficar_folds.py 05seg
python scripts/graficar_duraciones.py
python scripts/graficar_overlap.py --save
```

### Testing

No hay suite de tests configurada. Se recomienda usar pytest:

```bash
# Instalar pytest
pip install pytest pytest-cov

# Ejecutar todos los tests
pytest

# Ejecutar un test específico
pytest test_modelo.py::test_funcion -v

# Ejecutar tests con cobertura
pytest --cov=. --cov-report=html
```

### Linting y Formato

```bash
# Instalar herramientas
pip install black ruff mypy

# Formatear código (linea guía: 100 caracteres)
black --line-length 100 *.py scripts/ utils/

# Linting
ruff check *.py scripts/ utils/

# Type checking
mypy modelo_xvector.py scripts/ utils/
```

## Guías de Estilo de Código

### Formato General

- **Longitud de línea**: 100 caracteres máximo
- **Indentación**: 4 espacios (PEP 8)
- **Comillas**: Dobles `""` para strings, docstrings con triple comilla doble
- **Encoding**: UTF-8
- **Fin de línea**: Unix (LF)

### Imports (Orden estricto)

```python
# 1. Librerías estándar
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 2. Librerías de terceros
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 3. Imports internos (siempre usar sys.path.insert)
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
from modelo_xvector import SMAWXVectorModel
from utils.audio_utils import PROJECT_ROOT
```

### Convenciones de Nombres

| Elemento | Convención | Ejemplo |
|----------|-----------|---------|
| Clases | PascalCase | `SMAWXVectorModel` |
| Funciones | snake_case | `extract_vggish_embeddings()` |
| Variables | snake_case | `segment_duration`, `num_epochs` |
| Constantes | UPPER_SNAKE_CASE | `RANDOM_SEED = 42` |
| Parámetros | snake_case | `learning_rate` |
| Atributos privados | _snake_case | `_internal_state` |

### Type Hints (Obligatorios en funciones públicas)

```python
def load_audio_segment(
    audio_path: Path,
    segment_duration: float,
    segment_index: int,
    sr: int = 16000,
) -> Optional[np.ndarray]:
    """Carga un segmento específico de audio."""
    ...
```

### Docstrings (Google Style)

```python
def extract_labels_from_session_path(session_path: Path) -> Optional[Dict]:
    """
    Extrae etiquetas del path de una carpeta de sesión.

    Args:
        session_path: Path de la sesión de audio.

    Returns:
        dict con Plate Thickness, Electrode, Type of Current, Session.
        None si el path no tiene la estructura esperada.
    """
```

### Manejo de Errores

```python
try:
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
except Exception as e:
    print(f"Error loading {audio_path}: {e}")
    return None

# Validaciones con mensajes claros
if not audio_path.exists():
    raise ValueError(f"Audio file not found: {audio_path}")
```

### Estructura de Archivos Python

```python
"""
Descripción corta del módulo.

Descripción más larga si es necesaria.
"""

# === Imports ===
# (separados por categoría)

# === Constantes ===
RANDOM_SEED = 42
BATCH_SIZE = 32

# === Clases ===
class MiClase:
    """Docstring de la clase."""
    pass

# === Funciones ===
def funcion_principal():
    """Docstring de la función."""
    pass

# === Entry Point ===
if __name__ == "__main__":
    main()
```

## Reglas Específicas del Proyecto

### 1. Path Handling (Siempre usar Pathlib)

```python
from pathlib import Path

# Correcto
ROOT_DIR = Path(__file__).parent
audio_path = ROOT_DIR / "audio" / "file.wav"

# Incorrecto
import os
audio_path = os.path.join("audio", "file.wav")
```

### 2. Imports de Proyecto

Siempre usar `sys.path.insert` para imports internos:

```python
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
from modelo_xvector import SMAWXVectorModel
from utils.audio_utils import load_audio_segment
```

### 3. Warning Filters

Suprimir warnings en scripts de producción:

```python
import warnings
warnings.filterwarnings("ignore")
```

### 4. Seed para Reproducibilidad

```python
import numpy as np
import torch

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
```

## Dependencias Principales

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install librosa tensorflow tensorflow-hub
pip install matplotlib seaborn
pip install black ruff mypy pytest
```

## Notas Importantes

1. **Idioma**: Código y comentarios en español, excepto términos técnicos
2. **Modelos**: No versionar archivos `.pth`, `.keras`, `.pkl` (están en .gitignore)
3. **Audio**: No versionar archivos `.wav`, `.mp3` (están en .gitignore)
4. **VGGish**: Modelo pre-entrenado de Google (~275MB) se descarga automáticamente
5. **JSON unificado**: Los 3 modelos (xvector, ecapa, feedforward) guardan resultados en el mismo formato en `resultados.json`
