from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

STATE_PATH = Path('/app/data/homeostasis_state.json')


def _default_state() -> dict[str, Any]:
    now = int(time.time())
    return {
        'updated_at': now,
        'mode': 'normal',  # normal | repair | conservative
        'vitals': {
            'coherence_score': 0.70,
            'uncertainty_load': 0.30,
            'memory_pressure': 0.20,
            'goal_drift': 0.20,
            'contradiction_stress': 0.20,
            'energy_budget': 0.70,
        },
        'history': [],
    }


def _load() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        d = json.loads(STATE_PATH.read_text(encoding='utf-8'))
        if isinstance(d, dict):
            d.setdefault('vitals', _default_state()['vitals'])
            d.setdefault('history', [])
            d.setdefault('mode', 'normal')
            return d
    except Exception:
        pass
    return _default_state()


def _save(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state['updated_at'] = int(time.time())
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _pick_mode(v: dict[str, float], prev_mode: str = 'normal', thresholds: dict[str, float] | None = None) -> str:
    th = thresholds or {}
    repair_coh = float(th.get('repair_coherence') or 0.45)
    repair_ctr = float(th.get('repair_contradiction') or 0.75)
    cons_unc = float(th.get('conservative_uncertainty') or 0.70)

    # hysteresis to avoid mode flapping
    if prev_mode == 'repair':
        if v['coherence_score'] > max(0.58, repair_coh + 0.1) and v['contradiction_stress'] < max(0.58, repair_ctr - 0.1):
            return 'conservative'
        return 'repair'
    if prev_mode == 'conservative':
        if v['coherence_score'] < max(0.42, repair_coh - 0.03) or v['contradiction_stress'] > min(0.9, repair_ctr + 0.03):
            return 'repair'
        if v['energy_budget'] > 0.55 and v['uncertainty_load'] < 0.55 and v['coherence_score'] > 0.65:
            return 'normal'
        return 'conservative'

    if v['coherence_score'] < repair_coh or v['contradiction_stress'] > repair_ctr:
        return 'repair'
    if v['energy_budget'] < 0.35 or v['uncertainty_load'] > cons_unc:
        return 'conservative'
    return 'normal'


def evaluate(
    *,
    stats: dict[str, Any] | None,
    open_conflicts: int,
    decision_quality: float,
    queue_size: int,
    used_last_minute: int,
    per_minute: int,
    active_goal: bool,
    blocked_ratio: float = 0.0,
    error_ratio: float = 0.0,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    st = stats or {}

    questions_open = int(st.get('questions_open') or 0)
    unc = _clip01(0.38 * min(1.0, questions_open / 12.0) + 0.47 * (1.0 - _clip01(decision_quality)) + 0.15 * _clip01(error_ratio))
    memp = _clip01(min(1.0, float(st.get('experiences') or 0) / 12000.0) * 0.35 + min(1.0, queue_size / 25.0) * 0.5 + 0.15 * _clip01(blocked_ratio))
    cstress = _clip01(min(1.0, float(open_conflicts) / 8.0) * 0.8 + 0.2 * _clip01(blocked_ratio + error_ratio))
    gdrift = 0.18 if active_goal else 0.55
    energy = _clip01(1.0 - (float(used_last_minute) / max(1.0, float(per_minute or 1))))

    coherence = _clip01(1.0 - (0.35 * unc + 0.25 * memp + 0.25 * cstress + 0.15 * gdrift))

    vitals = {
        'coherence_score': round(coherence, 4),
        'uncertainty_load': round(unc, 4),
        'memory_pressure': round(memp, 4),
        'goal_drift': round(gdrift, 4),
        'contradiction_stress': round(cstress, 4),
        'energy_budget': round(energy, 4),
    }
    state = _load()
    prev_mode = str(state.get('mode') or 'normal')
    mode = _pick_mode(vitals, prev_mode=prev_mode, thresholds=thresholds)
    state['mode'] = mode
    state['vitals'] = vitals
    h = list(state.get('history') or [])
    h.append({'ts': int(time.time()), 'mode': mode, 'vitals': vitals})
    state['history'] = h[-240:]
    _save(state)

    actions = []
    if mode == 'repair':
        actions = ['prioritize_conflict_resolution', 'pause_heavy_experiments', 'run_deliberative_recovery']
    elif mode == 'conservative':
        actions = ['reduce_parallel_actions', 'prefer_low_cost_routes', 'increase_evidence_threshold']
    else:
        actions = ['normal_operation', 'allow_experiments']

    operation_mode = mode
    if mode == 'conservative' and vitals.get('uncertainty_load', 0.0) > 0.60 and vitals.get('energy_budget', 0.0) > 0.55:
        operation_mode = 'investigative'
        actions = ['run_multi_hypothesis_deliberation', 'mandatory_grounding_for_key_claims', 'symbolic_disambiguation_priority']

    return {
        'ok': True,
        'mode': mode,
        'operation_mode': operation_mode,
        'mode_changed': mode != prev_mode,
        'previous_mode': prev_mode,
        'vitals': vitals,
        'recommended_actions': actions,
        'state_path': str(STATE_PATH),
    }


def status() -> dict[str, Any]:
    s = _load()
    return {
        'ok': True,
        'mode': s.get('mode') or 'normal',
        'vitals': s.get('vitals') or {},
        'updated_at': s.get('updated_at'),
        'state_path': str(STATE_PATH),
        'history_tail': (s.get('history') or [])[-20:],
    }
