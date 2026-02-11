from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

PATH = Path('/app/data/grounding_claims.json')


def _load() -> dict[str, Any]:
    if PATH.exists():
        try:
            d = json.loads(PATH.read_text(encoding='utf-8'))
            if isinstance(d, dict):
                d.setdefault('claims', [])
                return d
        except Exception:
            pass
    return {'updated_at': int(time.time()), 'claims': []}


def _save(d: dict[str, Any]) -> None:
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def _score(sql_ok: bool, py_ok: bool, src_ok: bool) -> float:
    s = 0.2
    if sql_ok:
        s += 0.25
    if py_ok:
        s += 0.25
    if src_ok:
        s += 0.30
    return max(0.0, min(1.0, round(s, 4)))


def record_claim(
    *,
    claim: str,
    sql_result: dict[str, Any] | None,
    python_result: dict[str, Any] | None,
    source_result: dict[str, Any] | None,
    conclusion: str,
) -> dict[str, Any]:
    sql_ok = bool((sql_result or {}).get('ok'))
    py_ok = bool((python_result or {}).get('ok'))
    src_ok = bool((source_result or {}).get('ok'))
    reliability = _score(sql_ok, py_ok, src_ok)

    item = {
        'id': f"clm_{int(time.time()*1000)}",
        'ts': int(time.time()),
        'claim': str(claim or '')[:500],
        'checks': {
            'sql': sql_result or {'ok': False, 'skipped': True},
            'python': python_result or {'ok': False, 'skipped': True},
            'source': source_result or {'ok': False, 'skipped': True},
        },
        'reliability': reliability,
        'conclusion': str(conclusion or '')[:500],
    }

    d = _load()
    arr = list(d.get('claims') or [])
    arr.append(item)
    d['claims'] = arr[-600:]
    _save(d)
    return item


def latest(limit: int = 40) -> dict[str, Any]:
    d = _load()
    return {'ok': True, 'claims': (d.get('claims') or [])[-max(1, int(limit)):], 'path': str(PATH)}
