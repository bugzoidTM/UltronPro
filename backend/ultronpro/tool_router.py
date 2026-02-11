from __future__ import annotations

from typing import Any


def _score(c: dict[str, Any], prefer_low_cost: bool = True) -> float:
    util = float(c.get('utility') or 0.0)
    cost = float(c.get('cost') or 0.0)
    lat = float(c.get('latency') or 0.0)
    return util - ((0.45 if prefer_low_cost else 0.25) * cost) - (0.2 * lat)


def plan_route(intent: str, context: dict[str, Any] | None = None, prefer_low_cost: bool = True) -> dict[str, Any]:
    intent = (intent or 'general').strip().lower()
    ctx = context or {}

    candidates: list[dict[str, Any]] = []
    if intent in ('conflict_recovery', 'conflict_stalemate'):
        candidates = [
            {'kind': 'generate_analogy_hypothesis', 'utility': 0.72, 'cost': 0.38, 'latency': 0.42},
            {'kind': 'deliberate_task', 'utility': 0.78, 'cost': 0.62, 'latency': 0.66},
            {'kind': 'ask_evidence', 'utility': 0.61, 'cost': 0.15, 'latency': 0.18},
        ]
    elif intent in ('timeout_recovery', 'tool_failure'):
        candidates = [
            {'kind': 'ask_evidence', 'utility': 0.55, 'cost': 0.1, 'latency': 0.1},
            {'kind': 'deliberate_task', 'utility': 0.74, 'cost': 0.6, 'latency': 0.64},
            {'kind': 'maintain_question_queue', 'utility': 0.43, 'cost': 0.08, 'latency': 0.08},
        ]
    elif intent in ('hypothesis_validation', 'itc_test'):
        candidates = [
            {'kind': 'deliberate_task', 'utility': 0.82, 'cost': 0.64, 'latency': 0.68},
            {'kind': 'ask_evidence', 'utility': 0.59, 'cost': 0.14, 'latency': 0.2},
            {'kind': 'generate_analogy_hypothesis', 'utility': 0.63, 'cost': 0.32, 'latency': 0.38},
        ]
    else:
        candidates = [
            {'kind': 'ask_evidence', 'utility': 0.5, 'cost': 0.1, 'latency': 0.15},
            {'kind': 'deliberate_task', 'utility': 0.7, 'cost': 0.6, 'latency': 0.62},
            {'kind': 'generate_analogy_hypothesis', 'utility': 0.58, 'cost': 0.3, 'latency': 0.35},
        ]

    scored = []
    for c in candidates:
        x = dict(c)
        x['score'] = round(_score(c, prefer_low_cost=prefer_low_cost), 4)
        scored.append(x)

    scored.sort(key=lambda x: float(x.get('score') or 0.0), reverse=True)
    chain = [x.get('kind') for x in scored]
    return {
        'intent': intent,
        'prefer_low_cost': bool(prefer_low_cost),
        'context': {k: ctx.get(k) for k in list(ctx.keys())[:8]},
        'candidates': scored,
        'chain': chain,
        'primary': chain[0] if chain else None,
        'fallbacks': chain[1:3],
    }
