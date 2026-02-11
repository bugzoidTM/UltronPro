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
        'causal': {
            'strategy_outcomes': {},
            'task_outcomes': {},
            'budget_profile_outcomes': {},
            'recent_events': [],
        },
    }


def load() -> dict[str, Any]:
    try:
        if PATH.exists():
            d = json.loads(PATH.read_text())
            if isinstance(d, dict):
                d.setdefault('causal', _default()['causal'])
                d['causal'].setdefault('strategy_outcomes', {})
                d['causal'].setdefault('task_outcomes', {})
                d['causal'].setdefault('budget_profile_outcomes', {})
                d['causal'].setdefault('recent_events', [])
                return d
    except Exception:
        pass
    return _default()


def save(d: dict[str, Any]):
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def _acc(stats_map: dict[str, Any], key: str, ok: bool, latency_ms: int | None = None):
    it = dict(stats_map.get(key) or {})
    it['count'] = int(it.get('count') or 0) + 1
    it['success'] = int(it.get('success') or 0) + (1 if ok else 0)
    it['error'] = int(it.get('error') or 0) + (0 if ok else 1)
    if latency_ms is not None:
        total_lat = float(it.get('lat_total_ms') or 0.0) + float(latency_ms)
        it['lat_total_ms'] = round(total_lat, 3)
        it['lat_avg_ms'] = round(total_lat / max(1, int(it['count'])), 3)
    it['success_rate'] = round(float(it['success']) / max(1, int(it['count'])), 4)
    # Bayesian-smoothed reliability (Beta prior alpha=1,beta=1)
    it['bayes_rate'] = round((float(it['success']) + 1.0) / (float(it['count']) + 2.0), 4)
    it['confidence'] = round(min(1.0, float(it['count']) / 20.0), 4)
    it['updated_at'] = int(time.time())
    stats_map[key] = it


def record_action_outcome(
    *,
    strategy: str,
    task_type: str,
    budget_profile: str,
    ok: bool,
    latency_ms: int | None = None,
    notes: str | None = None,
):
    d = load()
    c = d.setdefault('causal', _default()['causal'])

    _acc(c.setdefault('strategy_outcomes', {}), str(strategy or 'unknown')[:80], ok, latency_ms)
    _acc(c.setdefault('task_outcomes', {}), str(task_type or 'general')[:80], ok, latency_ms)
    _acc(c.setdefault('budget_profile_outcomes', {}), str(budget_profile or 'default')[:80], ok, latency_ms)

    ev = list(c.get('recent_events') or [])
    ev.append({
        'ts': int(time.time()),
        'strategy': str(strategy or 'unknown')[:80],
        'task_type': str(task_type or 'general')[:80],
        'budget_profile': str(budget_profile or 'default')[:80],
        'ok': bool(ok),
        'latency_ms': int(latency_ms or 0),
        'notes': str(notes or '')[:220],
    })
    c['recent_events'] = ev[-400:]
    d['causal'] = c
    save(d)


def causal_summary(limit: int = 12) -> dict[str, Any]:
    d = load()
    c = d.get('causal') or {}

    def top_items(m: dict[str, Any]) -> list[dict[str, Any]]:
        items = []
        for k, v in (m or {}).items():
            it = dict(v)
            it['key'] = k
            items.append(it)
        items.sort(key=lambda x: (float(x.get('bayes_rate') or x.get('success_rate') or 0.0), -int(x.get('count') or 0)), reverse=True)
        return items[:max(1, int(limit))]

    return {
        'ok': True,
        'strategy_outcomes': top_items(c.get('strategy_outcomes') or {}),
        'task_outcomes': top_items(c.get('task_outcomes') or {}),
        'budget_profile_outcomes': top_items(c.get('budget_profile_outcomes') or {}),
        'recent_events': (c.get('recent_events') or [])[-max(1, int(limit)):],
    }


def best_strategy_scores(limit: int = 60) -> dict[str, float]:
    cs = causal_summary(limit=limit)
    out: dict[str, float] = {}
    for it in (cs.get('strategy_outcomes') or []):
        key = str(it.get('key') or '')
        bayes = float(it.get('bayes_rate') or it.get('success_rate') or 0.0)
        conf = float(it.get('confidence') or 0.0)
        lat = float(it.get('lat_avg_ms') or 0.0)
        lat_penalty = min(0.15, lat / 12000.0)
        # blended utility favors reliable + confident + low-latency strategies
        utility = max(0.0, min(1.0, bayes * (0.65 + 0.35 * conf) - lat_penalty))
        out[key] = round(utility, 4)
    return out


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
