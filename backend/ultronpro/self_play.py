from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import random
import time

PATH = Path('/app/data/self_play_history.json')


def _load() -> dict[str, Any]:
    if PATH.exists():
        try:
            d = json.loads(PATH.read_text(encoding='utf-8'))
            if isinstance(d, dict):
                d.setdefault('runs', [])
                return d
        except Exception:
            pass
    return {'updated_at': int(time.time()), 'runs': []}


def _save(d: dict[str, Any]) -> None:
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def simulate_batch(size: int = 12) -> dict[str, Any]:
    task_types = ['heartbeat', 'research', 'review', 'coding', 'critical']
    profiles = ['cheap', 'balanced', 'deep']

    out = []
    for _ in range(max(1, min(int(size), 80))):
        tt = random.choice(task_types)
        pf = random.choice(profiles)
        # synthetic priors: deep better on critical, cheap better on heartbeat/review latency
        base_ok = 0.72
        if tt == 'critical':
            base_ok += 0.12 if pf == 'deep' else (-0.08 if pf == 'cheap' else 0.0)
        if tt in ('heartbeat', 'review') and pf == 'cheap':
            base_ok += 0.08
        if tt == 'coding' and pf == 'deep':
            base_ok += 0.05
        ok = random.random() < max(0.05, min(0.97, base_ok))

        lat = random.randint(80, 600)
        if pf == 'deep':
            lat = random.randint(600, 3000)
        elif pf == 'balanced':
            lat = random.randint(250, 1500)

        reliability = random.uniform(0.35, 0.95) if tt in ('research', 'critical') else random.uniform(0.2, 0.8)
        out.append({'task_type': tt, 'profile': pf, 'ok': ok, 'latency_ms': lat, 'reliability': round(reliability, 4)})

    d = _load()
    run = {'id': f"spr_{int(time.time()*1000)}", 'ts': int(time.time()), 'size': len(out), 'samples': out}
    arr = list(d.get('runs') or [])
    arr.append(run)
    d['runs'] = arr[-300:]
    _save(d)
    return {'ok': True, 'run': run, 'path': str(PATH)}


def status(limit: int = 10) -> dict[str, Any]:
    d = _load()
    return {'ok': True, 'runs': (d.get('runs') or [])[-max(1, int(limit)):], 'path': str(PATH)}
