from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

PATH = Path('/app/data/critical_deliberation.json')

CRITICAL_KINDS = {
    'execute_procedure_active',
    'execute_python_sandbox',
    'route_toolchain',
    'verify_source_headless',
    'deliberate_task',
}


def _load() -> dict[str, Any]:
    if PATH.exists():
        try:
            d = json.loads(PATH.read_text(encoding='utf-8'))
            if isinstance(d, dict):
                d.setdefault('reports', [])
                return d
        except Exception:
            pass
    return {'updated_at': int(time.time()), 'reports': []}


def _save(d: dict[str, Any]) -> None:
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def _alt_plans(kind: str, text: str, meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    m = meta or {}
    return [
        {
            'id': 'safe_small',
            'plan': f'Executar versão mínima e reversível de {kind}.',
            'cost': 0.25,
            'risk': 0.20,
            'benefit': 0.55,
            'why_not': 'Pode gerar avanço limitado.',
        },
        {
            'id': 'balanced',
            'plan': f'Executar {kind} com validação intermediária e auditoria completa.',
            'cost': 0.45,
            'risk': 0.35,
            'benefit': 0.78,
            'why_not': 'Custo moderado e mais latência.',
        },
        {
            'id': 'aggressive',
            'plan': f'Executar {kind} em profundidade máxima para ganho rápido.',
            'cost': 0.75,
            'risk': 0.72,
            'benefit': 0.88,
            'why_not': 'Risco alto para cenário crítico; pode degradar coerência.',
        },
    ]


def _score(p: dict[str, Any]) -> float:
    return float(p.get('benefit') or 0.0) - 0.45 * float(p.get('risk') or 0.0) - 0.25 * float(p.get('cost') or 0.0)


def deliberate(kind: str, text: str, meta: dict[str, Any] | None = None, *, require_min_score: float = 0.30) -> dict[str, Any]:
    plans = _alt_plans(kind, text, meta)
    ranked = []
    for p in plans:
        pp = dict(p)
        pp['score'] = round(_score(pp), 4)
        ranked.append(pp)
    ranked.sort(key=lambda x: float(x.get('score') or 0.0), reverse=True)

    chosen = ranked[0] if ranked else None
    approved = bool(chosen and float(chosen.get('score') or 0.0) >= float(require_min_score))

    report = {
        'id': f"cdr_{int(time.time()*1000)}",
        'ts': int(time.time()),
        'kind': str(kind or ''),
        'text': str(text or '')[:240],
        'approved': approved,
        'require_min_score': float(require_min_score),
        'chosen': chosen,
        'alternatives': ranked,
    }

    d = _load()
    arr = list(d.get('reports') or [])
    arr.append(report)
    d['reports'] = arr[-500:]
    _save(d)
    return report


def latest(limit: int = 40) -> dict[str, Any]:
    d = _load()
    return {'ok': True, 'reports': (d.get('reports') or [])[-max(1, int(limit)):], 'path': str(PATH)}
