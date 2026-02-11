from __future__ import annotations

from pathlib import Path
from typing import Any
import os


def scan_tree(root: str = '/app/ultronpro', limit: int = 400) -> dict[str, Any]:
    rp = Path(root)
    if not rp.exists():
        return {'ok': False, 'error': 'root_not_found', 'root': root}

    items = []
    for p in rp.rglob('*'):
        if not p.is_file():
            continue
        if any(x in p.parts for x in ('.git', '__pycache__', '.venv')):
            continue
        try:
            sz = p.stat().st_size
        except Exception:
            continue
        items.append({'path': str(p), 'size': int(sz), 'ext': p.suffix.lower()})
        if len(items) >= limit:
            break

    items.sort(key=lambda x: x['size'], reverse=True)
    py = [x for x in items if x['ext'] == '.py']
    top = py[:20]

    suggestions = []
    for x in top[:8]:
        if x['size'] > 60000:
            suggestions.append(f"Refatorar arquivo grande: {x['path']} ({x['size']} bytes)")
    if not suggestions and top:
        suggestions.append('Aplicar revisão de complexidade e funções longas nos maiores módulos Python.')

    return {
        'ok': True,
        'root': root,
        'files_scanned': len(items),
        'python_files': len(py),
        'largest_python': top,
        'suggestions': suggestions[:12],
    }
