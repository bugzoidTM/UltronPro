from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

PATH = Path('/app/data/adaptive_control.json')


def _default() -> dict[str, Any]:
    return {
        'updated_at': int(time.time()),
        'targets': {
            'coherence_min': 0.68,
            'uncertainty_max': 0.52,
            'contradiction_max': 0.50,
            'blocked_ratio_max': 0.15,
        },
        'thresholds': {
            'repair_coherence': 0.45,
            'repair_contradiction': 0.75,
            'conservative_uncertainty': 0.70,
        },
        'last_tune_at': 0,
        'notes': [],
    }


def _load() -> dict[str, Any]:
    if PATH.exists():
        try:
            d = json.loads(PATH.read_text(encoding='utf-8'))
            if isinstance(d, dict):
                d.setdefault('targets', _default()['targets'])
                d.setdefault('thresholds', _default()['thresholds'])
                d.setdefault('notes', [])
                d.setdefault('last_tune_at', 0)
                return d
        except Exception:
            pass
    return _default()


def _save(d: dict[str, Any]):
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def tune_from_homeostasis(history: list[dict[str, Any]], blocked_ratio: float, strategy_diversity: int) -> dict[str, Any]:
    d = _load()
    if not history:
        return {'ok': True, 'changed': False, 'config': d}

    tail = history[-40:]
    coh = sum(float((x.get('vitals') or {}).get('coherence_score') or 0.0) for x in tail) / max(1, len(tail))
    unc = sum(float((x.get('vitals') or {}).get('uncertainty_load') or 0.0) for x in tail) / max(1, len(tail))
    ctr = sum(float((x.get('vitals') or {}).get('contradiction_stress') or 0.0) for x in tail) / max(1, len(tail))

    th = dict(d.get('thresholds') or {})
    changed = False

    # If system is too stressed for long, make repair entry easier (more protective)
    if coh < 0.58 or ctr > 0.62 or blocked_ratio > 0.20:
        n = min(0.65, float(th.get('repair_coherence') or 0.45) + 0.02)
        if abs(n - float(th.get('repair_coherence') or 0.45)) > 1e-6:
            th['repair_coherence'] = round(n, 4)
            changed = True
        n2 = max(0.55, float(th.get('repair_contradiction') or 0.75) - 0.02)
        if abs(n2 - float(th.get('repair_contradiction') or 0.75)) > 1e-6:
            th['repair_contradiction'] = round(n2, 4)
            changed = True

    # If stable, relax a bit to allow more exploration
    if coh > 0.72 and unc < 0.48 and ctr < 0.45 and blocked_ratio < 0.12:
        n = max(0.40, float(th.get('repair_coherence') or 0.45) - 0.01)
        if abs(n - float(th.get('repair_coherence') or 0.45)) > 1e-6:
            th['repair_coherence'] = round(n, 4)
            changed = True

    # Encourage strategy diversity if too low
    if strategy_diversity < 4:
        d['notes'].append({'ts': int(time.time()), 'type': 'low_diversity', 'msg': 'Increase exploration: diversify budget_profile and strategies.'})
        changed = True

    d['thresholds'] = th
    d['last_tune_at'] = int(time.time())
    d['notes'] = (d.get('notes') or [])[-200:]
    if changed:
        _save(d)
    return {'ok': True, 'changed': changed, 'config': d, 'signals': {'coherence_avg': round(coh, 4), 'uncertainty_avg': round(unc, 4), 'contradiction_avg': round(ctr, 4), 'blocked_ratio': round(float(blocked_ratio), 4), 'strategy_diversity': int(strategy_diversity)}}


def status() -> dict[str, Any]:
    d = _load()
    return {'ok': True, **d, 'path': str(PATH)}
