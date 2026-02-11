from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from ultronpro import llm

HISTORY_PATH = Path('/app/data/itc_history.json')


def _load() -> list[dict[str, Any]]:
    try:
        if HISTORY_PATH.exists():
            d = json.loads(HISTORY_PATH.read_text())
            if isinstance(d, list):
                return d
    except Exception:
        pass
    return []


def _save(items: list[dict[str, Any]]):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(items[-400:], ensure_ascii=False, indent=2))


def history(limit: int = 40) -> list[dict[str, Any]]:
    arr = _load()
    return arr[-max(1, int(limit)):]


def run_episode(problem_text: str, max_steps: int = 4, budget_seconds: int = 35) -> dict[str, Any]:
    start = time.time()
    p = (problem_text or '').strip()
    if len(p) < 12:
        return {'status': 'insufficient_context'}

    steps = max(2, min(8, int(max_steps)))
    budget = max(10, min(180, int(budget_seconds)))
    traces: list[dict[str, Any]] = []
    hypotheses: list[str] = []

    for i in range(steps):
        if (time.time() - start) >= budget:
            break
        h_prev = '; '.join(hypotheses[-3:])
        prompt = (
            'You are running a System-2 deliberate reasoning pass. '\
            'Return ONLY JSON with keys: hypothesis, counter_hypothesis, test, confidence (0..1).\n'
            f'Problem: {p[:1800]}\n'
            f'Prior hypotheses: {h_prev[:600]}'
        )
        obj = None
        try:
            raw = llm.complete(prompt, strategy='reasoning', json_mode=True)
            d = json.loads(raw) if raw else {}
            obj = {
                'hypothesis': str(d.get('hypothesis') or '').strip(),
                'counter_hypothesis': str(d.get('counter_hypothesis') or '').strip(),
                'test': str(d.get('test') or '').strip(),
                'confidence': max(0.0, min(1.0, float(d.get('confidence') or 0.5))),
            }
        except Exception:
            obj = None

        if not obj or not obj.get('hypothesis'):
            # deterministic fallback
            h = f'Iteração {i+1}: reduzir incerteza em subproblema crítico.'
            obj = {
                'hypothesis': h,
                'counter_hypothesis': f'Iteração {i+1}: hipótese rival com menor custo.',
                'test': 'Executar microteste observável e comparar resultado.',
                'confidence': 0.52,
            }

        traces.append({'step': i + 1, **obj})
        hypotheses.append(obj['hypothesis'])

    best = sorted(traces, key=lambda x: float(x.get('confidence') or 0), reverse=True)[0] if traces else None
    out = {
        'status': 'ok' if traces else 'empty',
        'problem_text': p[:2200],
        'steps': traces,
        'chosen': best,
        'elapsed_sec': round(time.time() - start, 3),
        'budget_seconds': budget,
        'max_steps': steps,
        'quality_proxy': round(sum(float(s.get('confidence') or 0) for s in traces) / max(1, len(traces)), 3) if traces else 0.0,
    }

    arr = _load()
    arr.append({'ts': int(time.time()), **out})
    _save(arr)
    return out
