"""Utility helpers shared across the RAG pipeline."""

from __future__ import annotations

import random
import re
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Creates a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def debug_print(enabled: bool, message: str) -> None:
    """Prints a debug message when debugging is enabled."""

    if enabled:
        print(f"[DEBUG] {message}")


def set_reproducibility(seed: int) -> None:
    """Sets reproducibility controls for supported libraries."""

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def normalize_whitespace(text: str) -> str:
    """Normalizes whitespace while preserving sentence content."""

    return re.sub(r"\s+", " ", text).strip()


def sanitize_collection_name(name: str) -> str:
    """Converts a collection name into a filesystem-safe identifier."""

    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return cleaned or "collection"
