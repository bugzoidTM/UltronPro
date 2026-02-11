from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import random
import time

STATE_PATH = Path('/app/data/emergence_state.json')
EVAL_PATH = Path('/app/data/emergence_eval_history.json')


def _load(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def state() -> dict[str, Any]:
    d = _load(STATE_PATH, None)
    if isinstance(d, dict):
        return d
    return {
        'created_at': int(time.time()),
        'updated_at': int(time.time()),
        'latent': {'curiosity': 0.5, 'risk_appetite': 0.35, 'exploration': 0.55, 'coherence_bias': 0.6},
        'noise_scale': 0.12,
        'last_policy': None,
    }


def save_state(st: dict[str, Any]):
    st['updated_at'] = int(time.time())
    _save(STATE_PATH, st)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def tick_latent(inputs: dict[str, Any]) -> dict[str, Any]:
    st = state()
    lat = dict(st.get('latent') or {})

    dq = float(inputs.get('decision_quality') or 0.5)
    conflicts = float(inputs.get('open_conflicts') or 0)
    novelty = float(inputs.get('novelty_index') or 0.4)

    # deterministic drift
    lat['curiosity'] = _clamp(0.82 * lat.get('curiosity', 0.5) + 0.18 * novelty)
    lat['coherence_bias'] = _clamp(0.8 * lat.get('coherence_bias', 0.6) + 0.2 * (1.0 - min(1.0, conflicts / 80.0)))
    lat['risk_appetite'] = _clamp(0.78 * lat.get('risk_appetite', 0.35) + 0.22 * max(0.0, dq - 0.25))
    lat['exploration'] = _clamp(0.75 * lat.get('exploration', 0.55) + 0.25 * (lat['curiosity'] * (1.0 - lat['coherence_bias'] * 0.3)))

    # controlled stochasticity (bounded)
    nz = float(st.get('noise_scale') or 0.12)
    for k in list(lat.keys()):
        lat[k] = _clamp(lat[k] + random.gauss(0, nz) * 0.2)

    st['latent'] = lat
    save_state(st)
    return st


def sample_policies(context: dict[str, Any], n: int = 4) -> list[dict[str, Any]]:
    st = state()
    lat = st.get('latent') or {}
    n = max(2, min(8, int(n)))

    base_actions = [
        ('maintain_question_queue', 0.6),
        ('generate_analogy_hypothesis', 0.5),
        ('invent_procedure', 0.55),
        ('unsupervised_discovery', 0.5),
        ('auto_resolve_conflicts', 0.7),
        ('curate_memory', 0.5),
    ]

    policies = []
    for i in range(n):
        weights = {}
        for a, b in base_actions:
            w = b
            if a in ('invent_procedure', 'generate_analogy_hypothesis'):
                w += float(lat.get('exploration', 0.5)) * 0.5
            if a in ('curate_memory', 'auto_resolve_conflicts'):
                w += float(lat.get('coherence_bias', 0.5)) * 0.4
            if a == 'unsupervised_discovery':
                w += float(lat.get('curiosity', 0.5)) * 0.5
            w += random.uniform(-0.12, 0.12)
            weights[a] = max(0.01, w)

        top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        policy = {
            'id': f'p{i+1}',
            'actions': [t[0] for t in top],
            'weights': weights,
        }
        policies.append(policy)

    return policies


def choose_policy(policies: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    # score with latent + tiny entropy bonus
    st = state()
    lat = st.get('latent') or {}
    best = None
    best_s = -999
    for p in policies:
        ws = p.get('weights') or {}
        score = 0.0
        score += float(ws.get('invent_procedure', 0)) * float(lat.get('exploration', 0.5))
        score += float(ws.get('auto_resolve_conflicts', 0)) * float(lat.get('coherence_bias', 0.5))
        score += float(ws.get('unsupervised_discovery', 0)) * float(lat.get('curiosity', 0.5))
        entropy = -sum((v/sum(ws.values())) * math.log((v/sum(ws.values())) + 1e-9) for v in ws.values()) if ws else 0
        score += 0.08 * entropy
        if score > best_s:
            best_s = score
            best = dict(p)
            best['score'] = round(score, 4)

    st['last_policy'] = {'ts': int(time.time()), 'policy': best}
    save_state(st)
    return best or {'id': 'none', 'actions': [], 'weights': {}, 'score': 0.0}


def log_eval(item: dict[str, Any]):
    arr = _load(EVAL_PATH, [])
    if not isinstance(arr, list):
        arr = []
    arr.append(item)
    _save(EVAL_PATH, arr[-500:])


def eval_history(limit: int = 50) -> list[dict[str, Any]]:
    arr = _load(EVAL_PATH, [])
    if not isinstance(arr, list):
        return []
    return arr[-max(1, int(limit)):]
