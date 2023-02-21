"""Utils."""
import json
import os
from pathlib import Path

from typing import Any, Dict


def create_dir(path: os.PathLike) -> Path:
    """Create directory and parents if it doesn't already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(filepath: os.PathLike) -> Dict[Any, Any]:
    """Load from JSON file."""
    return json.loads(Path(filepath).read_bytes())
