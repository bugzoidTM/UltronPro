from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

PATH = Path('/app/data/self_model.json')


def _default() -> dict[str, Any]:
    return {
        'created_at': int(time.time()),
        'updated_at': int(time.time()),
        'identity': {
            'name': 'UltronPro',
            'role': 'Agente cognitivo autônomo orientado a objetivos',
            'mission': 'Aprender, planejar e agir com segurança usando guardrails simbólicos.',
        },
        'capabilities': [],
        'limits': [],
        'tooling': [],
        'recent_changes': [],
    }


def load() -> dict[str, Any]:
    try:
        if PATH.exists():
            d = json.loads(PATH.read_text())
            if isinstance(d, dict):
                return d
    except Exception:
        pass
    return _default()


def save(d: dict[str, Any]):
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def refresh_from_runtime(stats: dict[str, Any], capabilities: list[str], limits: list[str], tooling: list[str], notes: list[str] | None = None) -> dict[str, Any]:
    d = load()
    d['capabilities'] = sorted(list(dict.fromkeys((d.get('capabilities') or []) + [str(x) for x in capabilities if x])))[:120]
    d['limits'] = sorted(list(dict.fromkeys((d.get('limits') or []) + [str(x) for x in limits if x])))[:120]
    d['tooling'] = sorted(list(dict.fromkeys((d.get('tooling') or []) + [str(x) for x in tooling if x])))[:120]

    rc = list(d.get('recent_changes') or [])
    rc.append({
        'ts': int(time.time()),
        'stats': stats,
        'notes': [str(n)[:220] for n in (notes or [])[:6]],
    })
    d['recent_changes'] = rc[-200:]
    save(d)
    return d
