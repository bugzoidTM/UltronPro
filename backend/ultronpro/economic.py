from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import random
import time

PATH = Path('/app/data/economic_primitives.json')

PROFILES = ['cheap', 'balanced', 'deep']


def _default() -> dict[str, Any]:
    return {
        'updated_at': int(time.time()),
        'epsilon': 0.15,
        'min_epsilon': 0.05,
        'decay': 0.995,
        'task_stats': {},
        'recent': [],
    }


def _load() -> dict[str, Any]:
    if PATH.exists():
        try:
            d = json.loads(PATH.read_text(encoding='utf-8'))
            if isinstance(d, dict):
                d.setdefault('task_stats', {})
                d.setdefault('recent', [])
                d.setdefault('epsilon', 0.15)
                d.setdefault('min_epsilon', 0.05)
                d.setdefault('decay', 0.995)
                return d
        except Exception:
            pass
    return _default()


def _save(d: dict[str, Any]) -> None:
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def _task_bucket(task_type: str) -> dict[str, Any]:
    return {p: {'count': 0, 'reward_sum': 0.0, 'reward_avg': 0.0} for p in PROFILES}


def pick_profile(task_type: str) -> dict[str, Any]:
    d = _load()
    tt = str(task_type or 'general')[:40]
    ts = d.setdefault('task_stats', {})
    bucket = ts.setdefault(tt, _task_bucket(tt))

    eps = float(d.get('epsilon') or 0.15)
    explore = random.random() < eps
    if explore:
        chosen = random.choice(PROFILES)
    else:
        chosen = sorted(PROFILES, key=lambda p: float((bucket.get(p) or {}).get('reward_avg') or 0.0), reverse=True)[0]

    d['epsilon'] = max(float(d.get('min_epsilon') or 0.05), eps * float(d.get('decay') or 0.995))
    _save(d)
    return {'task_type': tt, 'profile': chosen, 'explore': explore, 'epsilon': d['epsilon']}


def reward(ok: bool, latency_ms: int, reliability: float | None = None) -> float:
    lat_penalty = min(0.45, max(0.0, float(latency_ms) / 15000.0))
    base = 1.0 if ok else 0.0
    rel = 0.0 if reliability is None else max(0.0, min(1.0, float(reliability))) * 0.35
    return round(max(0.0, min(1.0, 0.65 * base + rel - lat_penalty + 0.25)), 4)


def update(task_type: str, profile: str, reward_value: float, ok: bool, latency_ms: int) -> dict[str, Any]:
    d = _load()
    tt = str(task_type or 'general')[:40]
    pf = str(profile or 'balanced')
    if pf == 'default':
        pf = 'balanced'
    ts = d.setdefault('task_stats', {})
    bucket = ts.setdefault(tt, _task_bucket(tt))
    b = dict(bucket.get(pf) or {'count': 0, 'reward_sum': 0.0, 'reward_avg': 0.0})
    b['count'] = int(b.get('count') or 0) + 1
    b['reward_sum'] = round(float(b.get('reward_sum') or 0.0) + float(reward_value or 0.0), 6)
    b['reward_avg'] = round(float(b['reward_sum']) / max(1, int(b['count'])), 6)
    bucket[pf] = b
    ts[tt] = bucket

    recent = list(d.get('recent') or [])
    recent.append({'ts': int(time.time()), 'task_type': tt, 'profile': pf, 'reward': float(reward_value), 'ok': bool(ok), 'latency_ms': int(latency_ms)})
    d['recent'] = recent[-500:]
    _save(d)
    return {'ok': True, 'task_type': tt, 'profile': pf, 'stats': bucket[pf], 'epsilon': d.get('epsilon')}


def status(limit: int = 40) -> dict[str, Any]:
    d = _load()
    rec = (d.get('recent') or [])[-max(1, int(limit)):]
    prof_counts = {'cheap': 0, 'balanced': 0, 'deep': 0}
    for r in rec:
        p = str(r.get('profile') or 'balanced')
        if p in prof_counts:
            prof_counts[p] += 1
    return {
        'ok': True,
        'epsilon': d.get('epsilon'),
        'task_stats': d.get('task_stats') or {},
        'recent': rec,
        'profile_mix_recent': prof_counts,
        'path': str(PATH),
    }
