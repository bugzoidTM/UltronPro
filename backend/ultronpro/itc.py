from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import random
import time

from ultronpro import llm

HISTORY_PATH = Path('/app/data/itc_history.json')
POLICY_PATH = Path('/app/data/itc_policy.json')


ARMS = [
    {'name': 'fast', 'max_steps': 4, 'budget_seconds': 25},
    {'name': 'balanced', 'max_steps': 6, 'budget_seconds': 45},
    {'name': 'deep', 'max_steps': 8, 'budget_seconds': 70},
]


def _load_json(path: Path, default):
    try:
        if path.exists():
            d = json.loads(path.read_text())
            return d
    except Exception:
        pass
    return default


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load_history() -> list[dict[str, Any]]:
    d = _load_json(HISTORY_PATH, [])
    return d if isinstance(d, list) else []


def _save_history(items: list[dict[str, Any]]):
    _save_json(HISTORY_PATH, items[-600:])


def _default_policy() -> dict[str, Any]:
    return {
        'epsilon': 0.18,
        'counts': {a['name']: 0 for a in ARMS},
        'values': {a['name']: 0.5 for a in ARMS},
        'updated_at': int(time.time()),
    }


def _load_policy() -> dict[str, Any]:
    d = _load_json(POLICY_PATH, None)
    if not isinstance(d, dict):
        return _default_policy()
    d.setdefault('epsilon', 0.18)
    d.setdefault('counts', {})
    d.setdefault('values', {})
    for a in ARMS:
        d['counts'].setdefault(a['name'], 0)
        d['values'].setdefault(a['name'], 0.5)
    return d


def _save_policy(pol: dict[str, Any]):
    pol['updated_at'] = int(time.time())
    _save_json(POLICY_PATH, pol)


def history(limit: int = 40) -> list[dict[str, Any]]:
    arr = _load_history()
    return arr[-max(1, int(limit)):]


def policy_status() -> dict[str, Any]:
    return _load_policy()


def _choose_arm(problem_text: str, use_rl: bool = True) -> dict[str, Any]:
    if not use_rl:
        return ARMS[1]

    pol = _load_policy()
    eps = float(pol.get('epsilon') or 0.18)

    # contextual bias by complexity keywords
    p = (problem_text or '').lower()
    complex_hits = sum(1 for k in ['conflito', 'trade-off', 'ambígu', 'causal', 'risco', 'multiobjetivo'] if k in p)

    if random.random() < eps:
        arm = random.choice(ARMS)
    else:
        vals = pol.get('values') or {}
        arm = sorted(ARMS, key=lambda a: float(vals.get(a['name']) or 0.0), reverse=True)[0]

    if complex_hits >= 2 and arm['name'] == 'fast':
        arm = ARMS[1]
    if complex_hits >= 4:
        arm = ARMS[2]
    return arm


def _update_policy(arm_name: str, reward: float):
    pol = _load_policy()
    c = int((pol.get('counts') or {}).get(arm_name) or 0) + 1
    v = float((pol.get('values') or {}).get(arm_name) or 0.5)
    alpha = 1.0 / max(1, c)
    nv = (1.0 - alpha) * v + alpha * float(reward)
    pol['counts'][arm_name] = c
    pol['values'][arm_name] = max(0.0, min(1.0, nv))
    # anneal epsilon slowly
    pol['epsilon'] = max(0.06, float(pol.get('epsilon') or 0.18) * 0.998)
    _save_policy(pol)


def _generate_step(problem_text: str, prior: str) -> dict[str, Any]:
    prompt = (
        'Run a private deliberate reasoning step. Return ONLY JSON with keys: '\
        'hypothesis, counter_hypothesis, test, confidence (0..1).\n'
        f'Problem: {problem_text[:1800]}\nPrior: {prior[:700]}'
    )
    try:
        raw = llm.complete(prompt, strategy='reasoning', json_mode=True)
        d = json.loads(raw) if raw else {}
        h = str(d.get('hypothesis') or '').strip()
        if not h:
            raise ValueError('empty hypothesis')
        return {
            'hypothesis': h,
            'counter_hypothesis': str(d.get('counter_hypothesis') or '').strip()[:260],
            'test': str(d.get('test') or '').strip()[:260],
            'confidence': max(0.0, min(1.0, float(d.get('confidence') or 0.5))),
        }
    except Exception:
        return {
            'hypothesis': 'Reduzir incerteza no subproblema mais crítico.',
            'counter_hypothesis': 'Abordagem alternativa de menor custo.',
            'test': 'Executar microteste observável e comparar impacto/risco.',
            'confidence': 0.52,
        }


def _verify_step(problem_text: str, step: dict[str, Any]) -> dict[str, Any]:
    # quick deterministic checks first
    hyp = str(step.get('hypothesis') or '').strip()
    tst = str(step.get('test') or '').strip()
    if len(hyp) < 10 or len(tst) < 10:
        return {'valid': False, 'issue': 'underspecified', 'fix': 'Detalhar hipótese e teste observável.', 'delta': -0.15}

    prompt = (
        'Verify this reasoning step. Return ONLY JSON with keys: valid(true/false), '\
        'issue, fix, confidence_delta(-0.5..0.2).\n'
        f'Problem: {problem_text[:1500]}\n'
        f'Step hypothesis: {hyp[:300]}\nStep test: {tst[:300]}'
    )
    try:
        raw = llm.complete(prompt, strategy='cheap', json_mode=True)
        d = json.loads(raw) if raw else {}
        valid = bool(d.get('valid'))
        return {
            'valid': valid,
            'issue': str(d.get('issue') or '')[:220],
            'fix': str(d.get('fix') or '')[:240],
            'delta': max(-0.5, min(0.2, float(d.get('confidence_delta') or (0.03 if valid else -0.08)))),
        }
    except Exception:
        return {'valid': True, 'issue': '', 'fix': '', 'delta': 0.0}


def run_episode(problem_text: str, max_steps: int = 0, budget_seconds: int = 0, use_rl: bool = True) -> dict[str, Any]:
    start = time.time()
    p = (problem_text or '').strip()
    if len(p) < 12:
        return {'status': 'insufficient_context'}

    arm = _choose_arm(p, use_rl=use_rl)
    steps = int(max_steps) if int(max_steps or 0) > 0 else int(arm['max_steps'])
    budget = int(budget_seconds) if int(budget_seconds or 0) > 0 else int(arm['budget_seconds'])

    steps = max(2, min(10, steps))
    budget = max(12, min(240, budget))

    traces: list[dict[str, Any]] = []
    hidden_trace: list[dict[str, Any]] = []
    prior = ''
    corrected = 0

    for i in range(steps):
        if (time.time() - start) >= budget:
            break

        st = _generate_step(p, prior)
        ver = _verify_step(p, st)
        conf = max(0.0, min(1.0, float(st.get('confidence') or 0.5) + float(ver.get('delta') or 0.0)))

        if not bool(ver.get('valid')) and ver.get('fix'):
            corrected += 1
            st['hypothesis'] = str(ver.get('fix'))[:260]
            st['test'] = f"Revalidar: {st.get('test') or 'microteste'}"
            conf = max(0.0, min(1.0, conf + 0.05))

        step_public = {
            'step': i + 1,
            'hypothesis': str(st.get('hypothesis') or '')[:220],
            'test': str(st.get('test') or '')[:220],
            'confidence': round(conf, 3),
            'corrected': bool(not ver.get('valid')),
        }
        traces.append(step_public)
        hidden_trace.append({'step': i + 1, 'raw': st, 'verification': ver, 'confidence': conf})
        prior = '; '.join([x['hypothesis'] for x in traces[-3:]])

    chosen = sorted(traces, key=lambda x: float(x.get('confidence') or 0), reverse=True)[0] if traces else None
    quality = round(sum(float(s.get('confidence') or 0) for s in traces) / max(1, len(traces)), 3) if traces else 0.0
    elapsed = round(time.time() - start, 3)

    # RL reward: quality gains, penalize latency and correction burden
    reward = max(0.0, min(1.0, quality - min(0.35, elapsed / 240.0) - min(0.25, corrected * 0.05)))
    if use_rl:
        _update_policy(arm['name'], reward)

    out = {
        'status': 'ok' if traces else 'empty',
        'problem_text': p[:2200],
        'steps': traces,
        'chosen': chosen,
        'elapsed_sec': elapsed,
        'budget_seconds': budget,
        'max_steps': steps,
        'policy_arm': arm['name'],
        'quality_proxy': quality,
        'corrections': corrected,
        'reward': round(reward, 3),
    }

    arr = _load_history()
    arr.append({'ts': int(time.time()), **out, 'internal': {'trace_len': len(hidden_trace)}})
    _save_history(arr)
    return out
