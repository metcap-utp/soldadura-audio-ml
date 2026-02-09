# Extract and organize audio from videos_soldadura

## Requisitos

- ffmpeg en PATH
- Python 3.8+

## Uso básico

Ejecutar extracción (por defecto coloca en `./05seg/audio/...`, pero puedes usar `--output-dir audio`):

```bash
python3 scripts/extract_and_organize_audio.py --videos-dir videos_soldadura --target 05seg
# o para mover ahora todo a ./audio:
python3 scripts/extract_and_organize_audio.py --videos-dir videos_soldadura --output-dir audio
```

## Opciones relevantes

- `--videos-dir`: ruta a la carpeta `videos_soldadura` (por defecto `videos_soldadura`).
- `--target`: carpeta destino donde crear `audio/` (por ejemplo `05seg`, `10seg`, `30seg`).
- `--target-root`: raíz donde está la carpeta `target` (por defecto `.`).
- `--samplerate`: sample rate para WAV (por defecto `16000`).
- `--overwrite`: sobrescribe archivos existentes.
- `--dry-run`: no ejecuta `ffmpeg`, solo muestra las acciones previstas.

## Comportamiento

- Se ignoran archivos no-vídeo (ej. `.json` de perfiles de cámara).
- Se mapean subcarpetas `E6010DC` o `E6010AC` a `E6010` automáticamente.

Ejemplo de uso en seco (sin ejecutar ffmpeg):

```bash
python3 scripts/extract_and_organize_audio.py --dry-run
```
