from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

STATE_PATH = Path('/app/data/long_horizon_state.json')


def _default() -> dict[str, Any]:
    return {
        'created_at': int(time.time()),
        'updated_at': int(time.time()),
        'active_mission_id': None,
        'missions': [],
    }


def load_state() -> dict[str, Any]:
    try:
        if STATE_PATH.exists():
            d = json.loads(STATE_PATH.read_text())
            if isinstance(d, dict):
                d.setdefault('missions', [])
                d.setdefault('active_mission_id', None)
                return d
    except Exception:
        pass
    return _default()


def save_state(st: dict[str, Any]):
    st['updated_at'] = int(time.time())
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2))


def active_mission() -> dict[str, Any] | None:
    st = load_state()
    mid = st.get('active_mission_id')
    for m in st.get('missions') or []:
        if m.get('id') == mid and str(m.get('status') or 'active') in ('active', 'paused'):
            return m
    return None


def list_missions(limit: int = 50) -> list[dict[str, Any]]:
    st = load_state()
    arr = list(st.get('missions') or [])
    return arr[-max(1, int(limit)):]


def upsert_mission(title: str, objective: str, horizon_days: int = 14, context: str | None = None) -> dict[str, Any]:
    st = load_state()
    title = (title or '').strip()[:180]
    objective = (objective or '').strip()[:1200]
    if not title:
        title = 'Missão de longo horizonte'

    # de-dupe por título ativo
    for m in st.get('missions') or []:
        if (m.get('title') or '').strip().lower() == title.lower() and str(m.get('status') or 'active') == 'active':
            if objective:
                m['objective'] = objective
            if context:
                m['context'] = (context or '')[:2000]
            m['updated_at'] = int(time.time())
            st['active_mission_id'] = m.get('id')
            save_state(st)
            return m

    hid = f"mis_{int(time.time())}_{len(st.get('missions') or [])+1}"
    now = int(time.time())
    item = {
        'id': hid,
        'title': title,
        'objective': objective,
        'context': (context or '')[:2000],
        'status': 'active',
        'created_at': now,
        'updated_at': now,
        'horizon_days': max(1, min(60, int(horizon_days))),
        'due_at': now + max(1, min(60, int(horizon_days))) * 86400,
        'checkpoints': [],
        'progress': 0.0,
    }
    st.setdefault('missions', []).append(item)
    st['active_mission_id'] = hid
    save_state(st)
    return item


def add_checkpoint(mission_id: str, note: str, progress_delta: float = 0.0, signal: str = 'reflection') -> dict[str, Any] | None:
    st = load_state()
    for m in st.get('missions') or []:
        if m.get('id') != mission_id:
            continue
        cp = {
            'ts': int(time.time()),
            'signal': (signal or 'reflection')[:40],
            'note': (note or '')[:1200],
            'progress_delta': float(progress_delta or 0.0),
        }
        arr = list(m.get('checkpoints') or [])
        arr.append(cp)
        m['checkpoints'] = arr[-200:]
        p = float(m.get('progress') or 0.0) + float(progress_delta or 0.0)
        m['progress'] = max(0.0, min(1.0, p))
        m['updated_at'] = int(time.time())
        save_state(st)
        return cp
    return None


def mission_context_snippet(mission: dict[str, Any], max_items: int = 8) -> str:
    cps = (mission.get('checkpoints') or [])[-max(1, int(max_items)):]
    lines = [f"Mission: {mission.get('title')} | objective={mission.get('objective')}"]
    if mission.get('context'):
        lines.append(f"Context: {str(mission.get('context'))[:300]}")
    lines.append(f"Progress: {float(mission.get('progress') or 0.0):.2f}")
    for c in cps:
        lines.append(f"- [{c.get('signal')}] {str(c.get('note') or '')[:140]}")
    return '\n'.join(lines)[:1800]


def rollover_if_due() -> dict[str, Any]:
    m = active_mission()
    if not m:
        return {'rolled': False, 'reason': 'no_active'}
    now = int(time.time())
    if now < int(m.get('due_at') or now + 1):
        return {'rolled': False, 'reason': 'not_due'}

    st = load_state()
    for x in st.get('missions') or []:
        if x.get('id') == m.get('id'):
            x['status'] = 'completed' if float(x.get('progress') or 0.0) >= 0.8 else 'paused'
            x['updated_at'] = now
            break
    st['active_mission_id'] = None
    save_state(st)
    return {'rolled': True, 'previous_id': m.get('id')}
