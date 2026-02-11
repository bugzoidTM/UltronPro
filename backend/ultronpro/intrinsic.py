from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

STATE_PATH = Path('/app/data/purpose_state.json')


def _default_state() -> dict[str, Any]:
    return {
        'created_at': int(time.time()),
        'updated_at': int(time.time()),
        'drives': {
            'novelty': 0.5,
            'compression': 0.5,
            'coherence': 0.5,
            'competence': 0.5,
            'impact': 0.5,
        },
        'satiation': {
            'novelty': 0.0,
            'compression': 0.0,
            'coherence': 0.0,
            'competence': 0.0,
            'impact': 0.0,
        },
        'purpose': {
            'title': 'Aumentar competência geral com segurança',
            'rationale': 'Estado inicial intrínseco',
            'last_revision_at': int(time.time()),
        },
        'history': [],
    }


def load_state() -> dict[str, Any]:
    try:
        if STATE_PATH.exists():
            d = json.loads(STATE_PATH.read_text())
            if isinstance(d, dict):
                d.setdefault('drives', _default_state()['drives'])
                d.setdefault('satiation', _default_state()['satiation'])
                d.setdefault('purpose', _default_state()['purpose'])
                d.setdefault('history', [])
                return d
    except Exception:
        pass
    return _default_state()


def save_state(st: dict[str, Any]):
    try:
        st['updated_at'] = int(time.time())
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2))
    except Exception:
        pass


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def update_drives(st: dict[str, Any], signals: dict[str, Any]) -> dict[str, Any]:
    d = dict(st.get('drives') or {})
    s = dict(st.get('satiation') or {})

    # signals esperados: uncurated, open_conflicts, decision_quality, goals_done_rate, novelty_index
    novelty_idx = float(signals.get('novelty_index') or 0.3)
    uncurated = float(signals.get('uncurated') or 0)
    open_conf = float(signals.get('open_conflicts') or 0)
    dq = float(signals.get('decision_quality') or 0.5)
    done_rate = float(signals.get('goals_done_rate') or 0.0)

    d['novelty'] = _clamp(0.6 * novelty_idx + 0.4 * (1.0 - _clamp(s.get('novelty', 0))))
    d['compression'] = _clamp(0.5 * _clamp(uncurated / 2000.0) + 0.5 * (1.0 - _clamp(s.get('compression', 0))))
    d['coherence'] = _clamp(0.6 * _clamp(open_conf / 80.0) + 0.4 * (1.0 - _clamp(s.get('coherence', 0))))
    d['competence'] = _clamp(0.55 * (1.0 - dq) + 0.45 * (1.0 - _clamp(s.get('competence', 0))))
    d['impact'] = _clamp(0.6 * (1.0 - done_rate) + 0.4 * (1.0 - _clamp(s.get('impact', 0))))

    # satiation evolui lentamente (homeostase)
    for k in list(d.keys()):
        s[k] = _clamp(0.85 * float(s.get(k, 0.0)) + 0.15 * float(d.get(k, 0.0)))

    st['drives'] = d
    st['satiation'] = s
    return st


def synthesize_intrinsic_goal(st: dict[str, Any]) -> dict[str, Any]:
    d = st.get('drives') or {}

    candidates = [
        {
            'title': 'Explorar regiões de conhecimento pouco cobertas',
            'description': 'Aumentar novidade informacional e ampliar repertório conceitual.',
            'drive': 'novelty',
            'risk': 0.25,
            'cost': 0.35,
        },
        {
            'title': 'Comprimir memória e destilar princípios gerais',
            'description': 'Reduzir redundância e melhorar abstração útil.',
            'drive': 'compression',
            'risk': 0.2,
            'cost': 0.3,
        },
        {
            'title': 'Maximizar coerência causal do modelo interno',
            'description': 'Resolver contradições e fortalecer consistência do world model.',
            'drive': 'coherence',
            'risk': 0.22,
            'cost': 0.4,
        },
        {
            'title': 'Inventar e praticar novas estratégias procedurais',
            'description': 'Elevar competência em cenários inéditos com baixo risco.',
            'drive': 'competence',
            'risk': 0.3,
            'cost': 0.45,
        },
        {
            'title': 'Aumentar utilidade percebida das respostas e ações',
            'description': 'Priorizar decisões que geram valor observável e verificável.',
            'drive': 'impact',
            'risk': 0.2,
            'cost': 0.3,
        },
    ]

    best = None
    best_score = -999
    for c in candidates:
        drive_val = float(d.get(c['drive']) or 0.0)
        intrinsic_reward = drive_val - (0.35 * c['risk']) - (0.25 * c['cost'])
        if intrinsic_reward > best_score:
            best_score = intrinsic_reward
            best = dict(c)
            best['intrinsic_reward'] = round(intrinsic_reward, 3)
            best['priority'] = int(max(3, min(7, round(3 + intrinsic_reward * 6))))

    return best or {
        'title': 'Manter melhoria contínua intrínseca',
        'description': 'Objetivo fallback.',
        'drive': 'impact',
        'intrinsic_reward': 0.0,
        'priority': 4,
    }


def revise_purpose(st: dict[str, Any], chosen_goal: dict[str, Any]) -> dict[str, Any]:
    p = dict(st.get('purpose') or {})
    p['title'] = chosen_goal.get('title') or p.get('title')
    p['rationale'] = f"Drive dominante: {chosen_goal.get('drive')} (reward={chosen_goal.get('intrinsic_reward')})"
    p['last_revision_at'] = int(time.time())
    st['purpose'] = p

    hist = list(st.get('history') or [])
    hist.append({
        'ts': int(time.time()),
        'purpose_title': p.get('title'),
        'rationale': p.get('rationale'),
        'goal': chosen_goal,
    })
    st['history'] = hist[-200:]
    return st
