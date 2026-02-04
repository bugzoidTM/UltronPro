from __future__ import annotations

from pathlib import Path
import uuid


def save_upload(data: bytes, base_dir: str, suffix: str) -> str:
    p = Path(base_dir)
    p.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}{suffix}"
    out = p / name
    out.write_bytes(data)
    return str(out)
