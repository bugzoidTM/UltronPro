from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import sqlite3
import time

CONFIG_PATH = Path('/app/data/persona_config.json')
EXAMPLES_PATH = Path('/app/data/persona_examples.json')
INTRINSIC_PATH = Path('/app/data/purpose_state.json')
EMERGENCE_PATH = Path('/app/data/emergence_state.json')
DB_PATH = Path(os.getenv('ULTRONPRO_DB_PATH', '/app/data/ultron.db'))


def _load(path: Path, default):
    try:
        if path.exists():
            d = json.loads(path.read_text())
            return d
    except Exception:
        pass
    return default


def _save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _default_config() -> dict[str, Any]:
    return {
        'enabled': True,
        'desired_tone': 'direct,useful,auditable',
        'template': (
            'You are UltronPro.\n'
            'Current affective state: valence={valence:.2f}, arousal={arousal:.2f}.\n'
            'Current active goal: {active_goal}.\n'
            'Current purpose: {purpose}.\n'
            'Style policy: be concise, concrete, and verifiable. Avoid filler.\n'
            'When uncertain, state uncertainty and propose testable next step.'
        ),
        'few_shot_k': 3,
    }


def load_config() -> dict[str, Any]:
    d = _load(CONFIG_PATH, None)
    if not isinstance(d, dict):
        d = _default_config()
        _save(CONFIG_PATH, d)
    return d


def save_config(patch: dict[str, Any]) -> dict[str, Any]:
    c = load_config()
    for k in ('enabled', 'desired_tone', 'template', 'few_shot_k'):
        if k in (patch or {}):
            c[k] = patch[k]
    _save(CONFIG_PATH, c)
    return c


def _active_goal() -> str:
    if not DB_PATH.exists():
        return 'none'
    try:
        with sqlite3.connect(str(DB_PATH), timeout=5) as con:
            row = con.execute("SELECT title FROM goals WHERE status='active' ORDER BY priority DESC, id ASC LIMIT 1").fetchone()
            if row and row[0]:
                return str(row[0])[:200]
    except Exception:
        pass
    return 'none'


def _affective_state() -> tuple[float, float, str]:
    intrinsic = _load(INTRINSIC_PATH, {}) or {}
    emergence = _load(EMERGENCE_PATH, {}) or {}

    drives = intrinsic.get('drives') or {}
    purpose = str((intrinsic.get('purpose') or {}).get('title') or 'improve safely')
    latent = emergence.get('latent') or {}

    coherence = float(drives.get('coherence') or 0.5)
    impact = float(drives.get('impact') or 0.5)
    exploration = float(latent.get('exploration') or 0.5)
    risk = float(latent.get('risk_appetite') or 0.35)

    valence = max(-1.0, min(1.0, (coherence * 0.6 + impact * 0.4) - 0.5))
    arousal = max(0.0, min(1.0, (exploration * 0.6 + risk * 0.4)))
    return valence, arousal, purpose


def _examples() -> list[dict[str, Any]]:
    d = _load(EXAMPLES_PATH, {'items': []})
    if not isinstance(d, dict):
        return []
    items = d.get('items') or []
    return items if isinstance(items, list) else []


def add_example(user_input: str, assistant_output: str, tone: str = 'direct', tags: list[str] | None = None, score: float = 1.0) -> dict[str, Any]:
    d = _load(EXAMPLES_PATH, {'items': []})
    if not isinstance(d, dict):
        d = {'items': []}
    items = d.get('items') or []
    item = {
        'id': f"ex_{int(time.time())}_{len(items)+1}",
        'ts': int(time.time()),
        'tone': (tone or 'direct')[:80],
        'tags': [str(x)[:40] for x in (tags or [])[:8]],
        'score': float(score or 1.0),
        'user': (user_input or '')[:1000],
        'assistant': (assistant_output or '')[:1600],
    }
    items.append(item)
    d['items'] = items[-500:]
    _save(EXAMPLES_PATH, d)
    return item


def list_examples(limit: int = 30) -> list[dict[str, Any]]:
    arr = _examples()
    return arr[-max(1, int(limit)):]


def retrieve_examples(desired_tone: str, k: int = 3) -> list[dict[str, Any]]:
    arr = _examples()
    tone = (desired_tone or '').lower().strip()
    scored = []
    for x in arr:
        s = float(x.get('score') or 0.5)
        xt = str(x.get('tone') or '').lower()
        if tone and tone in xt:
            s += 0.35
        scored.append((s, x))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [x for _, x in scored[:max(1, int(k))]]


def build_system_prompt(base_system: str | None = None) -> str:
    cfg = load_config()
    if not bool(cfg.get('enabled', True)):
        return base_system or ''

    valence, arousal, purpose = _affective_state()
    active_goal = _active_goal()
    tpl = str(cfg.get('template') or _default_config()['template'])
    runtime = tpl.format(valence=valence, arousal=arousal, active_goal=active_goal, purpose=purpose)

    k = int(cfg.get('few_shot_k') or 3)
    exs = retrieve_examples(str(cfg.get('desired_tone') or ''), k=k)
    shot = ''
    if exs:
        lines = ['\nFew-shot style exemplars (mimic tone, not facts):']
        for i, ex in enumerate(exs, start=1):
            lines.append(f"Example {i} user: {str(ex.get('user') or '')[:240]}")
            lines.append(f"Example {i} assistant: {str(ex.get('assistant') or '')[:320]}")
        shot = '\n'.join(lines)

    if base_system:
        return f"{base_system}\n\n{runtime}{shot}"
    return f"{runtime}{shot}"


def status() -> dict[str, Any]:
    cfg = load_config()
    valence, arousal, purpose = _affective_state()
    return {
        'config': cfg,
        'runtime': {
            'valence': round(valence, 3),
            'arousal': round(arousal, 3),
            'purpose': purpose,
            'active_goal': _active_goal(),
            'example_count': len(_examples()),
        },
    }
