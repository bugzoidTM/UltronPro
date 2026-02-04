from __future__ import annotations

from typing import Any
from PIL import Image


def image_basic_facts(path: str) -> dict[str, Any]:
    """Lightweight 'perception' placeholder.

    Real multimodal perception would require a vision model.
    We at least extract deterministic facts (size/mode) now.
    """
    img = Image.open(path)
    return {
        "type": "image",
        "width": img.size[0],
        "height": img.size[1],
        "mode": img.mode,
        "format": img.format,
    }
