from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from ultronpro import llm

PATH = Path('/app/data/subgoal_dag.json')


def _default() -> dict[str, Any]:
    return {'updated_at': int(time.time()), 'roots': []}


def load() -> dict[str, Any]:
    try:
        if PATH.exists():
            d = json.loads(PATH.read_text())
            if isinstance(d, dict):
                d.setdefault('roots', [])
                return d
    except Exception:
        pass
    return _default()


def save(d: dict[str, Any]):
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def _fallback_nodes(title: str, objective: str) -> list[dict[str, Any]]:
    return [
        {'id': f'n_{int(time.time())}_1', 'title': f'Clarificar escopo: {title}', 'parent_id': None, 'status': 'open', 'priority': 7},
        {'id': f'n_{int(time.time())}_2', 'title': 'Mapear restrições e riscos', 'parent_id': f'n_{int(time.time())}_1', 'status': 'open', 'priority': 6},
        {'id': f'n_{int(time.time())}_3', 'title': 'Gerar hipóteses de execução', 'parent_id': f'n_{int(time.time())}_1', 'status': 'open', 'priority': 6},
        {'id': f'n_{int(time.time())}_4', 'title': 'Executar microteste de menor custo', 'parent_id': f'n_{int(time.time())}_3', 'status': 'open', 'priority': 5},
        {'id': f'n_{int(time.time())}_5', 'title': 'Consolidar aprendizado e próximo ciclo', 'parent_id': f'n_{int(time.time())}_4', 'status': 'open', 'priority': 5},
    ]


def synthesize_for_goal(title: str, objective: str, max_nodes: int = 7) -> dict[str, Any]:
    title = (title or 'Goal').strip()
    objective = (objective or '').strip()
    nodes = None
    try:
        prompt = f"""Decompose the goal into a DAG of subgoals.
Return ONLY JSON array with fields: id,title,parent_id,priority(1..7).
Goal: {title}
Objective: {objective}
Max nodes: {max_nodes}
"""
        raw = llm.complete(prompt, strategy='cheap', json_mode=True)
        arr = json.loads(raw) if raw else []
        if isinstance(arr, list) and arr:
            out = []
            for i, it in enumerate(arr[:max_nodes], start=1):
                if not isinstance(it, dict) or not it.get('title'):
                    continue
                nid = str(it.get('id') or f'n_{int(time.time())}_{i}')
                pid = it.get('parent_id')
                out.append({'id': nid, 'title': str(it.get('title'))[:180], 'parent_id': str(pid) if pid else None, 'status': 'open', 'priority': int(max(1, min(7, int(it.get('priority') or 5))))})
            if out:
                nodes = out
    except Exception:
        nodes = None

    if not nodes:
        nodes = _fallback_nodes(title, objective)[:max_nodes]

    root = {
        'id': f'root_{int(time.time())}',
        'title': title,
        'objective': objective[:1200],
        'status': 'active',
        'created_at': int(time.time()),
        'nodes': nodes,
    }

    d = load()
    d.setdefault('roots', []).append(root)
    d['roots'] = d['roots'][-60:]
    save(d)
    return root


def list_roots(limit: int = 20) -> list[dict[str, Any]]:
    d = load()
    return (d.get('roots') or [])[-max(1, int(limit)):]


def mark_node(root_id: str, node_id: str, status: str = 'done') -> bool:
    d = load()
    for r in d.get('roots') or []:
        if r.get('id') != root_id:
            continue
        for n in r.get('nodes') or []:
            if n.get('id') == node_id:
                n['status'] = status
                n['updated_at'] = int(time.time())
                save(d)
                return True
    return False
