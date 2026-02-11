from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

PROOFS_PATH = Path('/app/data/neurosym_proofs.json')


def _load() -> list[dict[str, Any]]:
    try:
        if PROOFS_PATH.exists():
            d = json.loads(PROOFS_PATH.read_text())
            if isinstance(d, list):
                return d
    except Exception:
        pass
    return []


def _save(items: list[dict[str, Any]]):
    PROOFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROOFS_PATH.write_text(json.dumps(items[-600:], ensure_ascii=False, indent=2))


def add_proof(decision_type: str, premises: list[str], inference: str, conclusion: str, confidence: float = 0.5, action_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    item = {
        'id': f'pf_{int(time.time())}_{len(_load())+1}',
        'ts': int(time.time()),
        'decision_type': (decision_type or 'unknown')[:80],
        'premises': [str(x)[:240] for x in (premises or [])[:12]],
        'inference': (inference or '')[:800],
        'conclusion': (conclusion or '')[:500],
        'confidence': max(0.0, min(1.0, float(confidence or 0.5))),
        'action_meta': action_meta or {},
    }
    arr = _load()
    arr.append(item)
    _save(arr)
    return item


def history(limit: int = 50) -> list[dict[str, Any]]:
    arr = _load()
    return arr[-max(1, int(limit)):]


def consistency_check(limit: int = 200) -> dict[str, Any]:
    arr = history(limit=limit)
    contradictions = 0
    by_decision: dict[str, str] = {}
    for p in arr:
        dt = str(p.get('decision_type') or '')
        conc = str(p.get('conclusion') or '').strip().lower()
        key = f"{dt}:{(p.get('action_meta') or {}).get('kind') or ''}"
        if key in by_decision and by_decision[key] != conc:
            contradictions += 1
        by_decision[key] = conc

    score = max(0.0, min(1.0, 1.0 - (contradictions / max(1, len(arr)))))
    return {'samples': len(arr), 'contradictions': contradictions, 'consistency_score': round(score, 3)}


def explanation_fidelity(limit: int = 120) -> dict[str, Any]:
    arr = history(limit=limit)
    ok = 0
    for p in arr:
        conc = str(p.get('conclusion') or '').lower()
        am = p.get('action_meta') or {}
        kind = str(am.get('kind') or '').lower()
        # proxy fidelity: conclusion menciona kind/efeito principal
        if kind and (kind in conc or 'bloque' in conc or 'execut' in conc or 'skip' in conc):
            ok += 1
    score = (ok / max(1, len(arr))) if arr else 0.0
    return {'samples': len(arr), 'aligned': ok, 'fidelity_score': round(score, 3)}
