from __future__ import annotations

from typing import Any
import statistics
import time


COST_POLICY = {
    'heartbeat': {'model_hint': 'cheap', 'max_tokens': 500, 'thinking': 'low'},
    'research': {'model_hint': 'balanced', 'max_tokens': 1200, 'thinking': 'medium'},
    'coding': {'model_hint': 'balanced', 'max_tokens': 1400, 'thinking': 'medium'},
    'review': {'model_hint': 'cheap', 'max_tokens': 700, 'thinking': 'low'},
    'critical': {'model_hint': 'deep', 'max_tokens': 2200, 'thinking': 'high'},
}


def policy_for_task(task_type: str | None, critical: bool = False) -> dict[str, Any]:
    if critical:
        return dict(COST_POLICY['critical'])
    key = str(task_type or '').strip().lower()
    if key in COST_POLICY:
        return dict(COST_POLICY[key])
    return dict(COST_POLICY['heartbeat'])


def suggest_assignee(title: str, description: str = '') -> str:
    txt = f"{title} {description}".lower()
    if any(k in txt for k in ('conflict', 'evid', 'fonte', 'source', 'research', 'ground', 'sql')):
        return 'research'
    if any(k in txt for k in ('code', 'python', 'refactor', 'bug', 'perf', 'deploy', 'sandbox')):
        return 'engineer'
    return 'coord'


def productivity_metrics(tasks: list[dict[str, Any]], activities: list[dict[str, Any]], window_sec: int = 86400 * 7) -> dict[str, Any]:
    now = int(time.time())
    start = now - int(window_sec)

    rel_tasks = [t for t in (tasks or []) if int(t.get('updated_at') or t.get('created_at') or 0) >= start]
    rel_acts = [a for a in (activities or []) if int(a.get('ts') or 0) >= start]

    agents = {'coord', 'research', 'engineer'}
    out = {a: {'created': 0, 'done': 0, 'blocked': 0, 'in_progress': 0, 'messages': 0, 'lead_times_h': []} for a in agents}

    for t in rel_tasks:
        ass = [str(x).lower() for x in (t.get('assignees') or [])]
        st = str(t.get('status') or 'inbox')
        created = int(t.get('created_at') or 0)
        updated = int(t.get('updated_at') or created)
        lt_h = max(0.0, (updated - created) / 3600.0)
        for a in ass or ['coord']:
            if a not in out:
                out[a] = {'created': 0, 'done': 0, 'blocked': 0, 'in_progress': 0, 'messages': 0, 'lead_times_h': []}
            out[a]['created'] += 1
            if st == 'done':
                out[a]['done'] += 1
                out[a]['lead_times_h'].append(lt_h)
            elif st == 'blocked':
                out[a]['blocked'] += 1
            elif st == 'in_progress':
                out[a]['in_progress'] += 1

    for a in rel_acts:
        txt = str(a.get('text') or '').lower()
        for aid in list(out.keys()):
            if aid in txt:
                out[aid]['messages'] += 1

    summary = {}
    for aid, m in out.items():
        done = int(m['done'])
        created = int(m['created'])
        lt = m.get('lead_times_h') or []
        summary[aid] = {
            'created': created,
            'done': done,
            'blocked': int(m['blocked']),
            'in_progress': int(m['in_progress']),
            'messages': int(m['messages']),
            'throughput': (done / created) if created else 0.0,
            'avg_lead_time_h': float(statistics.mean(lt)) if lt else None,
        }

    return {
        'ok': True,
        'window_sec': int(window_sec),
        'agents': summary,
    }
