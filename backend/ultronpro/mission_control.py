from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import time

PATH = Path('/app/data/mission_control.json')

STATUSES = {'inbox', 'assigned', 'in_progress', 'review', 'done', 'blocked'}


def _load() -> dict[str, Any]:
    if PATH.exists():
        try:
            d = json.loads(PATH.read_text(encoding='utf-8'))
            if isinstance(d, dict):
                d.setdefault('tasks', [])
                d.setdefault('messages', [])
                d.setdefault('activities', [])
                d.setdefault('subscriptions', {})
                d.setdefault('notifications', [])
                return d
        except Exception:
            pass
    return {
        'updated_at': int(time.time()),
        'tasks': [],
        'messages': [],
        'activities': [],
        'subscriptions': {},
        'notifications': [],
    }


def _save(d: dict[str, Any]) -> None:
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def _id(prefix: str, arr: list[dict[str, Any]]) -> str:
    return f"{prefix}_{int(time.time())}_{len(arr)+1}"


def _extract_mentions(text: str) -> list[str]:
    t = str(text or '')
    found = re.findall(r'@([a-zA-Z0-9_\-]+)', t)
    return list(dict.fromkeys([x.lower() for x in found]))[:20]


def create_task(title: str, description: str = '', assignees: list[str] | None = None, task_type: str = 'heartbeat') -> dict[str, Any]:
    d = _load()
    t = {
        'id': _id('tsk', d['tasks']),
        'title': (title or '').strip()[:220] or 'Task sem título',
        'description': (description or '')[:3000],
        'status': 'inbox',
        'task_type': str(task_type or 'heartbeat').lower()[:32],
        'assignees': [str(a).lower() for a in (assignees or [])][:8],
        'created_at': int(time.time()),
        'updated_at': int(time.time()),
    }
    d['tasks'].append(t)
    d['activities'].append({'ts': int(time.time()), 'type': 'task_created', 'text': f"Task criada: {t['title']}", 'task_id': t['id']})
    for a in t['assignees']:
        d['notifications'].append({'id': _id('ntf', d['notifications']), 'agent_id': a, 'task_id': t['id'], 'text': f"Nova task atribuída: {t['title']}", 'delivered': False, 'created_at': int(time.time())})
    _save(d)
    return t


def list_tasks(status: str | None = None, limit: int = 80) -> list[dict[str, Any]]:
    d = _load()
    arr = d['tasks']
    if status and status in STATUSES:
        arr = [x for x in arr if x.get('status') == status]
    return arr[-max(1, int(limit)):]


def update_task(task_id: str, status: str | None = None, assignees: list[str] | None = None) -> dict[str, Any] | None:
    d = _load()
    for t in d['tasks']:
        if t.get('id') != task_id:
            continue
        if status:
            st = str(status).strip().lower()
            if st in STATUSES:
                t['status'] = st
        if assignees is not None:
            t['assignees'] = [str(a).lower() for a in assignees][:8]
        t['updated_at'] = int(time.time())
        d['activities'].append({'ts': int(time.time()), 'type': 'task_updated', 'text': f"Task {task_id} atualizada para {t.get('status')}", 'task_id': task_id})
        _save(d)
        return t
    return None


def subscribe(task_id: str, agent_id: str) -> dict[str, Any]:
    d = _load()
    subs = d.setdefault('subscriptions', {})
    arr = list(subs.get(task_id) or [])
    aid = str(agent_id).lower().strip()
    if aid and aid not in arr:
        arr.append(aid)
    subs[task_id] = arr[-20:]
    _save(d)
    return {'task_id': task_id, 'subscribers': subs[task_id]}


def add_message(task_id: str, from_agent: str, content: str) -> dict[str, Any]:
    d = _load()
    msg = {
        'id': _id('msg', d['messages']),
        'task_id': task_id,
        'from_agent': str(from_agent).lower()[:40],
        'content': (content or '')[:4000],
        'created_at': int(time.time()),
    }
    d['messages'].append(msg)

    # auto-subscribe author
    subs = d.setdefault('subscriptions', {})
    sarr = list(subs.get(task_id) or [])
    if msg['from_agent'] and msg['from_agent'] not in sarr:
        sarr.append(msg['from_agent'])
    subs[task_id] = sarr[-20:]

    mentions = _extract_mentions(content)
    recipients = set(subs.get(task_id) or []) | set(mentions)
    if msg['from_agent'] in recipients:
        recipients.remove(msg['from_agent'])

    for aid in recipients:
        d['notifications'].append({
            'id': _id('ntf', d['notifications']),
            'agent_id': aid,
            'task_id': task_id,
            'text': f"Nova mensagem em {task_id}: {msg['content'][:140]}",
            'delivered': False,
            'created_at': int(time.time()),
        })

    d['activities'].append({'ts': int(time.time()), 'type': 'message_sent', 'text': f"{msg['from_agent']} comentou em {task_id}", 'task_id': task_id})
    _save(d)
    return msg


def list_messages(task_id: str, limit: int = 60) -> list[dict[str, Any]]:
    d = _load()
    arr = [m for m in d['messages'] if m.get('task_id') == task_id]
    return arr[-max(1, int(limit)):]


def list_activities(limit: int = 120) -> list[dict[str, Any]]:
    d = _load()
    return (d['activities'] or [])[-max(1, int(limit)):]


def list_notifications(agent_id: str | None = None, delivered: bool | None = None, limit: int = 80) -> list[dict[str, Any]]:
    d = _load()
    arr = d['notifications']
    if agent_id:
        arr = [n for n in arr if str(n.get('agent_id') or '') == str(agent_id).lower()]
    if delivered is not None:
        arr = [n for n in arr if bool(n.get('delivered')) is bool(delivered)]
    return arr[-max(1, int(limit)):]


def mark_notification(notification_id: str, delivered: bool = True) -> bool:
    d = _load()
    ok = False
    for n in d['notifications']:
        if n.get('id') == notification_id:
            n['delivered'] = bool(delivered)
            ok = True
            break
    if ok:
        _save(d)
    return ok
