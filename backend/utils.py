"""I/O and helper utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def timestamp_now() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dirs(*dirs: str | Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def save_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
