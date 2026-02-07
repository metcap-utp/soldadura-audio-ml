#!/usr/bin/env python3
"""
Extrae audio de videos en `videos-soldadura` y organiza en la estructura
de `TARGET_SPLIT/audio/Placa_xxx/E####/` parecida a la del repo.

Requisitos: `ffmpeg` en PATH.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpg", ".mpeg"}


def is_video(p: Path):
    return p.suffix.lower() in VIDEO_EXTS


def electrode_label(name: str) -> str:
    """Extrae el código de electrodo (ej: E6011) del nombre de carpeta."""
    m = re.match(r"(E\d+)", name)
    if m:
        return m.group(1)
    return re.sub(r"[^A-Za-z0-9_\-]", "", name)


def current_type(name: str) -> str:
    """Extrae el tipo de corriente (DC o AC) del nombre de carpeta como E6011DC -> DC."""
    if "DC" in name.upper():
        return "DC"
    elif "AC" in name.upper():
        return "AC"
    return "DC"  # fallback


def video_timestamp(filename: str) -> str:
    """Extrae timestamp del nombre del archivo de video, ej: 240912-132141_6013-2.avi -> 240912-132141."""
    m = re.match(r"(\d{6}-\d{6})", filename)
    if m:
        return m.group(1)
    # fallback: usar el nombre sin extensión
    return Path(filename).stem


def extract_audio(
    input_path: Path, out_path: Path, samplerate: int, overwrite: bool
) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        print(f"Skipping existing: {out_path}")
        return False
    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(samplerate),
        "-ac",
        "1",
        "-y" if overwrite else "-n",
        str(out_path),
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        print(f"Extracted: {out_path}")
        return True
    except subprocess.CalledProcessError:
        print(f"ffmpeg failed for {input_path}")
        return False


def main():
    p = argparse.ArgumentParser(
        description="Extrae y organiza audio desde videos-soldadura"
    )
    p.add_argument(
        "--videos-dir", default="videos-soldadura", help="Ruta a videos-soldadura"
    )
    p.add_argument(
        "--target",
        default="05seg",
        help="Carpeta target (05seg,10seg,30seg) donde crear `audio/`",
    )
    p.add_argument(
        "--target-root", default=".", help="Raíz donde está la carpeta target"
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Ruta base de salida para audio (si se especifica, anula target/target-root). Ej: audio",
    )
    p.add_argument(
        "--samplerate", type=int, default=16000, help="Sample rate para salida WAV"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Sobrescribir archivos existentes"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No ejecutar ffmpeg, solo listar acciones",
    )
    args = p.parse_args()

    videos_root = Path(args.videos_dir)
    if not videos_root.exists():
        print(f"No existe {videos_root}")
        sys.exit(1)

    # Determinar carpeta de salida para los WAV:
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = Path(args.target_root) / args.target / "audio"

    # Recorrer: videos-soldadura/Placa_*/<electrode_dir>/*
    for placa_dir in sorted(videos_root.iterdir()):
        if not placa_dir.is_dir():
            continue
        placa_name = placa_dir.name
        for electrode_dir in sorted(placa_dir.iterdir()):
            if not electrode_dir.is_dir():
                continue
            elabel = electrode_label(electrode_dir.name)
            ctype = current_type(electrode_dir.name)
            # destino: out_root/Placa_xxx/E####/DC|AC/TIMESTAMP_Audio/
            for f in sorted(electrode_dir.iterdir()):
                if f.is_dir():
                    continue
                if not is_video(f):
                    # skip non-video (like .json camera profiles)
                    continue
                ts = video_timestamp(f.name)
                dest_dir = out_root / placa_name / elabel / ctype / f"{ts}_Audio"
                out_name = f.with_suffix(".wav").name
                out_path = dest_dir / out_name
                print(f"Will extract: {f} -> {out_path}")
                if args.dry_run:
                    continue
                extract_audio(f, out_path, args.samplerate, args.overwrite)


if __name__ == "__main__":
    main()
