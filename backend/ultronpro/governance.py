from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

PATH = Path('/app/data/governance_matrix.json')

DEFAULT = {
    'updated_at': int(time.time()),
    'classes': {
        'auto': {
            'description': 'Pode executar automaticamente.',
            'kinds': ['generate_questions', 'ask_evidence', 'maintain_question_queue', 'curate_memory', 'prune_memory', 'self_model_refresh'],
        },
        'auto_with_proof': {
            'description': 'Pode executar automaticamente, mas exige prova/auditoria.',
            'kinds': ['execute_python_sandbox', 'verify_source_headless', 'ground_claim_check', 'deliberate_task', 'route_toolchain', 'auto_resolve_conflicts'],
        },
        'human_approval': {
            'description': 'Exige aprovação humana explícita antes de executar.',
            'kinds': ['execute_procedure_active', 'invent_procedure'],
        },
    },
}


def _load() -> dict[str, Any]:
    if PATH.exists():
        try:
            d = json.loads(PATH.read_text(encoding='utf-8'))
            if isinstance(d, dict):
                d.setdefault('classes', DEFAULT['classes'])
                return d
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT))


def _save(d: dict[str, Any]) -> None:
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def matrix() -> dict[str, Any]:
    return _load()


def classify(kind: str) -> str:
    k = str(kind or '')
    d = _load()
    cls = d.get('classes') or {}
    for name in ('auto', 'auto_with_proof', 'human_approval'):
        kinds = set([str(x) for x in ((cls.get(name) or {}).get('kinds') or [])])
        if k in kinds:
            return name
    return 'auto_with_proof'


def evaluate(kind: str, meta: dict[str, Any] | None = None, *, has_proof: bool = False) -> dict[str, Any]:
    c = classify(kind)
    m = meta or {}
    if c == 'auto':
        return {'ok': True, 'class': c, 'reason': 'auto_allowed'}
    if c == 'auto_with_proof':
        if has_proof or bool(m.get('proof_ok')):
            return {'ok': True, 'class': c, 'reason': 'proof_present'}
        return {'ok': False, 'class': c, 'reason': 'proof_required'}
    # human approval
    approved = bool(m.get('approved_by_human'))
    return {'ok': approved, 'class': c, 'reason': 'human_approved' if approved else 'human_approval_required'}


def patch_matrix(patch: dict[str, Any]) -> dict[str, Any]:
    d = _load()
    cls = d.setdefault('classes', {})
    for name in ('auto', 'auto_with_proof', 'human_approval'):
        if name in (patch or {}):
            item = patch.get(name) or {}
            cur = cls.setdefault(name, {})
            if 'description' in item:
                cur['description'] = str(item.get('description') or '')[:280]
            if 'kinds' in item:
                cur['kinds'] = [str(x) for x in (item.get('kinds') or [])][:200]
    _save(d)
    return d
