from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DATA_DIR = Path('/app/data')
STATE_PATH = DATA_DIR / 'squad_phase_a.json'
MEM_DIR = DATA_DIR / 'memory' / 'agents'


DEFAULT = {
    'enabled': True,
    'interval_min': 15,
    'standup_hour_utc': 2,
    'last_standup_day': '',
    'agents': [
        {
            'id': 'coord',
            'name': 'Ultron-Coor',
            'role': 'Coordinator',
            'purpose': 'Orquestrar prioridades, delegar, destravar bloqueios.',
            'heartbeat_minute_offset': 0,
            'tools': ['planner', 'goals', 'project_kernel', 'integrity'],
        },
        {
            'id': 'research',
            'name': 'Ultron-Research',
            'role': 'Research & Grounding',
            'purpose': 'Buscar evidência confiável, validar fontes e reduzir alucinação.',
            'heartbeat_minute_offset': 5,
            'tools': ['verify_source_headless', 'sql_explorer', 'conflicts'],
        },
        {
            'id': 'engineer',
            'name': 'Ultron-Engineer',
            'role': 'Execution & Refactor',
            'purpose': 'Validar hipóteses via Python sandbox e propor melhorias no código.',
            'heartbeat_minute_offset': 10,
            'tools': ['execute_python_sandbox', 'filesystem_audit', 'project_experiment_cycle'],
        },
    ],
    'last_heartbeats': {},
}


def _load() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return json.loads(json.dumps(DEFAULT))
    try:
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return json.loads(json.dumps(DEFAULT))


def _save(state: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def _ensure_working_files(state: dict[str, Any]) -> list[str]:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for a in state.get('agents') or []:
        aid = str(a.get('id') or 'agent').strip()
        p = MEM_DIR / f'WORKING_{aid}.md'
        if p.exists():
            continue
        content = (
            f"# WORKING ({aid})\n\n"
            f"## Role\n{a.get('role')}\n\n"
            f"## Purpose\n{a.get('purpose')}\n\n"
            "## Current Task\n(none)\n\n"
            "## Status\nidle\n\n"
            "## Next Steps\n1. Check assigned project tasks\n2. Execute one concrete step\n3. Log result and blockers\n"
        )
        p.write_text(content, encoding='utf-8')
        created.append(str(p))
    return created


def bootstrap() -> dict[str, Any]:
    st = _load()
    if not st.get('agents'):
        st = json.loads(json.dumps(DEFAULT))
    created = _ensure_working_files(st)
    _save(st)
    return {
        'ok': True,
        'state_path': str(STATE_PATH),
        'working_dir': str(MEM_DIR),
        'agents': st.get('agents') or [],
        'created_working_files': created,
        'stagger_plan': {a.get('id'): a.get('heartbeat_minute_offset') for a in st.get('agents') or []},
    }


def status() -> dict[str, Any]:
    st = _load()
    _ensure_working_files(st)
    return {
        'ok': True,
        'enabled': bool(st.get('enabled', True)),
        'interval_min': int(st.get('interval_min') or 15),
        'agents': st.get('agents') or [],
        'last_heartbeats': st.get('last_heartbeats') or {},
        'state_path': str(STATE_PATH),
        'working_dir': str(MEM_DIR),
    }


def due_heartbeats(now_ts: float | None = None) -> list[dict[str, Any]]:
    st = _load()
    if not bool(st.get('enabled', True)):
        return []

    now = float(now_ts or time.time())
    interval = max(5, int(st.get('interval_min') or 15))
    minute = int(now // 60)
    out: list[dict[str, Any]] = []

    last = dict(st.get('last_heartbeats') or {})

    for a in st.get('agents') or []:
        aid = str(a.get('id') or '')
        off = int(a.get('heartbeat_minute_offset') or 0)
        if ((minute - off) % interval) != 0:
            continue
        last_min = int(last.get(aid) or -10**9)
        if minute - last_min < max(1, interval - 1):
            continue
        out.append(a)
        last[aid] = minute

    st['last_heartbeats'] = last
    _save(st)
    return out


def due_daily_standup(now_ts: float | None = None) -> bool:
    st = _load()
    now = float(now_ts or time.time())
    h = int(st.get('standup_hour_utc') or 2)
    tm = time.gmtime(now)
    day_key = f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d}"
    if int(tm.tm_hour) != h:
        return False
    if str(st.get('last_standup_day') or '') == day_key:
        return False
    st['last_standup_day'] = day_key
    _save(st)
    return True


def standup_from_events(events: list[dict[str, Any]], window_sec: int = 86400) -> dict[str, Any]:
    now = time.time()
    recent = [e for e in events if float(e.get('created_at') or 0) >= (now - float(window_sec))]

    completed = [e for e in recent if str(e.get('kind') or '').startswith('action_done')]
    blocked = [e for e in recent if 'blocked' in str(e.get('kind') or '') or 'falhou' in str(e.get('text') or '').lower()]
    in_progress = [e for e in recent if str(e.get('kind') or '') in ('autonomy_tick', 'planner', 'project_tick')]

    return {
        'ok': True,
        'window_sec': int(window_sec),
        'counts': {
            'completed': len(completed),
            'blocked': len(blocked),
            'in_progress_signals': len(in_progress),
            'events_scanned': len(recent),
        },
        'highlights': {
            'completed': [str(e.get('text') or '')[:180] for e in completed[-8:]],
            'blocked': [str(e.get('text') or '')[:180] for e in blocked[-6:]],
            'recent': [str(e.get('text') or '')[:180] for e in recent[-10:]],
        },
    }
