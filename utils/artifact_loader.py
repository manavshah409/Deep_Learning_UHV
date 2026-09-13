"""Safe and defensive artifact loading utilities for Phase 1 project repository."""

import hashlib
import json
from pathlib import Path
from typing import Any, List, Optional
import pandas as pd
import yaml

# Project root directory (parent of utils)
ROOT = Path(__file__).resolve().parents[1]


def find_file(relative_path: str) -> Optional[Path]:
    """Check if a relative file path exists on disk and return resolved Path or None."""
    try:
        p = ROOT / relative_path
        if p.exists():
            return p
    except Exception:
        pass
    return None


def load_json(relative_path: str) -> Optional[Any]:
    """Safely load a JSON file, returning None if missing or corrupted."""
    p = find_file(relative_path)
    if not p or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_yaml(relative_path: str) -> Optional[Any]:
    """Safely load a YAML file, returning None if missing or corrupted."""
    p = find_file(relative_path)
    if not p or not p.is_file():
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_csv(relative_path: str) -> Optional[pd.DataFrame]:
    """Safely load a CSV file into a pandas DataFrame, returning None if missing or corrupted."""
    p = find_file(relative_path)
    if not p or not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def get_file_size(relative_path: str) -> Optional[str]:
    """Return a human-readable file size string or None."""
    p = find_file(relative_path)
    if not p or not p.is_file():
        return None
    try:
        size_bytes = p.stat().st_size
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    except Exception:
        return None


def get_sha256(relative_path: str) -> Optional[str]:
    """Calculate the SHA-256 digest of a file, returning None if missing or unreadable."""
    p = find_file(relative_path)
    if not p or not p.is_file():
        return None
    try:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def find_images(
    relative_dir: str, extensions=("*.jpg", "*.png", "*.jpeg")
) -> List[Path]:
    """Find all image paths in a given relative directory."""
    p = find_file(relative_dir)
    if not p or not p.is_dir():
        return []
    images = []
    for ext in extensions:
        images.extend(p.glob(ext))
    return sorted(images)
