from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

PENDING_PATH = Path('/app/data/mutations_pending.json')
HISTORY_PATH = Path('/app/data/mutations_history.json')
ACTIVE_PATH = Path('/app/data/runtime_mutations_active.json')


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


def list_pending() -> list[dict[str, Any]]:
    arr = _load(PENDING_PATH, [])
    return arr if isinstance(arr, list) else []


def add_proposal(title: str, rationale: str, patch: dict[str, Any], author: str = 'autonomy') -> dict[str, Any]:
    arr = list_pending()
    pid = f"mut_{int(time.time())}_{len(arr)+1}"
    item = {
        'id': pid,
        'created_at': int(time.time()),
        'status': 'pending',
        'title': (title or 'Mutation proposal')[:180],
        'rationale': (rationale or '')[:1200],
        'patch': patch or {},
        'author': author,
        'shadow_metrics': None,
    }
    arr.append(item)
    _save(PENDING_PATH, arr)
    return item


def set_shadow_metrics(proposal_id: str, metrics: dict[str, Any]) -> dict[str, Any] | None:
    arr = list_pending()
    out = None
    for it in arr:
        if it.get('id') == proposal_id:
            it['shadow_metrics'] = metrics
            it['status'] = 'evaluated'
            out = it
            break
    _save(PENDING_PATH, arr)
    return out


def activate(proposal_id: str) -> dict[str, Any] | None:
    arr = list_pending()
    target = None
    for it in arr:
        if it.get('id') == proposal_id:
            it['status'] = 'active'
            target = it
            break
    if not target:
        return None

    active = _load(ACTIVE_PATH, {'active': []})
    if not isinstance(active, dict):
        active = {'active': []}
    active_list = active.get('active') or []
    active_list.append({'id': target.get('id'), 'patch': target.get('patch'), 'activated_at': int(time.time())})
    active['active'] = active_list[-20:]
    _save(ACTIVE_PATH, active)

    hist = _load(HISTORY_PATH, [])
    if not isinstance(hist, list):
        hist = []
    hist.append({'ts': int(time.time()), 'event': 'activate', 'proposal_id': proposal_id, 'title': target.get('title')})
    _save(HISTORY_PATH, hist[-500:])
    _save(PENDING_PATH, arr)
    return target


def revert(proposal_id: str, reason: str = 'manual_revert') -> bool:
    active = _load(ACTIVE_PATH, {'active': []})
    if not isinstance(active, dict):
        return False
    before = len(active.get('active') or [])
    active['active'] = [x for x in (active.get('active') or []) if x.get('id') != proposal_id]
    _save(ACTIVE_PATH, active)

    arr = list_pending()
    for it in arr:
        if it.get('id') == proposal_id:
            it['status'] = 'reverted'
            break
    _save(PENDING_PATH, arr)

    hist = _load(HISTORY_PATH, [])
    if not isinstance(hist, list):
        hist = []
    hist.append({'ts': int(time.time()), 'event': 'revert', 'proposal_id': proposal_id, 'reason': reason})
    _save(HISTORY_PATH, hist[-500:])
    return len(active.get('active') or []) < before


def active_runtime() -> dict[str, Any]:
    d = _load(ACTIVE_PATH, {'active': []})
    return d if isinstance(d, dict) else {'active': []}


def history(limit: int = 50) -> list[dict[str, Any]]:
    arr = _load(HISTORY_PATH, [])
    if not isinstance(arr, list):
        return []
    return arr[-max(1, int(limit)):]