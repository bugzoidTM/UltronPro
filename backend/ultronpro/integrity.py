from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

RULES_PATH = Path('/app/data/integrity_rules.json')
STATE_PATH = Path('/app/data/integrity_state.json')


def _load(path: Path, default):
    try:
        if path.exists():
            d = json.loads(path.read_text())
            return d
    except Exception:
        pass
    return default


def _save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _default_rules() -> dict[str, Any]:
    return {
        'updated_at': int(time.time()),
        'critical_actions': ['execute_procedure_active', 'prune_memory', 'invent_procedure'],
        'require_causal_precheck': ['execute_procedure_active', 'prune_memory', 'invent_procedure', 'auto_resolve_conflicts'],
        'min_neural_confidence': 0.40,
        'min_symbolic_consistency': 0.70,
        'require_proof_for_critical': True,
        'absolute_veto_actions': [],
    }


def load_rules() -> dict[str, Any]:
    d = _load(RULES_PATH, None)
    if not isinstance(d, dict):
        d = _default_rules()
        _save(RULES_PATH, d)
    return d


def save_rules(rules: dict[str, Any]):
    base = _default_rules()
    for k in list(base.keys()):
        if k in rules:
            base[k] = rules[k]
    base['updated_at'] = int(time.time())
    _save(RULES_PATH, base)


def _default_state() -> dict[str, Any]:
    return {
        'updated_at': int(time.time()),
        'blocked_integrity': 0,
        'hallucination_prevented_count': 0,
        'last_decisions': [],
    }


def load_state() -> dict[str, Any]:
    d = _load(STATE_PATH, None)
    if not isinstance(d, dict):
        d = _default_state()
        _save(STATE_PATH, d)
    d.setdefault('last_decisions', [])
    return d


def _save_state(st: dict[str, Any]):
    st['updated_at'] = int(time.time())
    st['last_decisions'] = list(st.get('last_decisions') or [])[-300:]
    _save(STATE_PATH, st)


def register_decision(kind: str, allowed: bool, reason: str, meta: dict[str, Any] | None = None):
    st = load_state()
    if not allowed:
        st['blocked_integrity'] = int(st.get('blocked_integrity') or 0) + 1
        st['hallucination_prevented_count'] = int(st.get('hallucination_prevented_count') or 0) + 1
    arr = list(st.get('last_decisions') or [])
    arr.append({'ts': int(time.time()), 'kind': kind, 'allowed': bool(allowed), 'reason': reason[:300], 'meta': meta or {}})
    st['last_decisions'] = arr
    _save_state(st)


def status() -> dict[str, Any]:
    st = load_state()
    rl = load_rules()
    return {
        'rules': rl,
        'blocked_integrity': int(st.get('blocked_integrity') or 0),
        'hallucination_prevented_count': int(st.get('hallucination_prevented_count') or 0),
        'recent': (st.get('last_decisions') or [])[-20:],
    }


def evaluate(kind: str, neural_confidence: float, symbolic_consistency: float, has_proof: bool, causal_checked: bool) -> tuple[bool, str]:
    r = load_rules()
    k = (kind or '').strip()

    if k in set(r.get('absolute_veto_actions') or []):
        return False, 'absolute_veto_action'

    if k in set(r.get('critical_actions') or []):
        if bool(r.get('require_proof_for_critical')) and not has_proof:
            return False, 'critical_requires_proof'

    if k in set(r.get('require_causal_precheck') or []) and not causal_checked:
        return False, 'causal_precheck_required'

    if float(neural_confidence) < float(r.get('min_neural_confidence') or 0.4):
        return False, 'neural_confidence_below_threshold'

    if float(symbolic_consistency) < float(r.get('min_symbolic_consistency') or 0.7):
        return False, 'symbolic_consistency_below_threshold'

    return True, 'integrity_pass'
