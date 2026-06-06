from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_directory(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    reset_directory(target_dir)
    saved_paths: List[Path] = []

    for uploaded_file in uploaded_files:
        output_path = target_dir / uploaded_file.name
        output_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(output_path)

    return saved_paths
