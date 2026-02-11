from __future__ import annotations

from typing import Any
import re

CAUSAL_PREDICATES = {
    'causa', 'causes', 'leva_a', 'implica', 'aumenta', 'reduz', 'influencia', 'impacta', 'provoca'
}

NEGATIVE_WORDS = {'erro', 'falha', 'risco', 'dano', 'instável', 'instavel', 'bloqueio'}
POSITIVE_WORDS = {'melhora', 'estável', 'estavel', 'sucesso', 'ganho', 'eficiente'}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or '').strip().lower())


def build_world_model(db, limit: int = 4000) -> dict[str, Any]:
    triples = db.list_triples_since(since_id=0, limit=max(200, int(limit)))
    nodes = {}
    edges = []

    for t in triples:
        s = _norm(t.get('subject') or '')
        p = _norm(t.get('predicate') or '')
        o = _norm(t.get('object') or '')
        if not s or not o:
            continue
        conf = float(t.get('confidence') or 0.5)
        nodes[s] = nodes.get(s, {'id': s, 'prior': 0.0})
        nodes[o] = nodes.get(o, {'id': o, 'prior': 0.0})

        # causal edges from specific predicates OR heuristic verbs
        if p in CAUSAL_PREDICATES or any(k in p for k in ['caus', 'impact', 'influ', 'aument', 'reduz', 'provoc', 'implic']):
            sign = -1.0 if ('reduz' in p or 'diminu' in p) else 1.0
            w = max(0.05, min(1.0, conf)) * sign
            edges.append({'from': s, 'to': o, 'w': w, 'predicate': p, 'confidence': conf})

    return {'nodes': nodes, 'edges': edges, 'triples_scanned': len(triples)}


def simulate_intervention(model: dict[str, Any], interventions: list[dict[str, Any]], steps: int = 3) -> dict[str, Any]:
    nodes = dict((k, {'id': v['id'], 'state': 0.0}) for k, v in (model.get('nodes') or {}).items())
    edges = model.get('edges') or []

    for it in interventions or []:
        n = _norm(str(it.get('node') or ''))
        d = float(it.get('delta') or 0.0)
        if n and n in nodes:
            nodes[n]['state'] += d

    for _ in range(max(1, int(steps))):
        delta = dict((k, 0.0) for k in nodes.keys())
        for e in edges:
            a = e['from']; b = e['to']; w = float(e.get('w') or 0.0)
            if a in nodes and b in nodes:
                delta[b] += nodes[a]['state'] * w * 0.6
        for k, dv in delta.items():
            nodes[k]['state'] += dv
            nodes[k]['state'] = max(-2.5, min(2.5, nodes[k]['state']))

    # evaluate risk/benefit
    risk = 0.0
    benefit = 0.0
    top = sorted(nodes.values(), key=lambda x: abs(float(x.get('state') or 0.0)), reverse=True)[:20]
    for n in top:
        name = n['id']; st = float(n['state'] or 0.0)
        if any(w in name for w in NEGATIVE_WORDS):
            risk += max(0.0, st)
        if any(w in name for w in POSITIVE_WORDS):
            benefit += max(0.0, st)

    return {
        'risk_score': round(risk, 3),
        'benefit_score': round(benefit, 3),
        'net_score': round(benefit - risk, 3),
        'top_effects': [{'node': n['id'], 'state': round(float(n['state'] or 0.0), 3)} for n in top],
        'edges': len(edges),
        'nodes': len(nodes),
    }


def infer_intervention_from_action(kind: str, text: str | None = None, meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    k = _norm(kind)
    t = _norm(text or '')
    ints = []
    if 'prune' in k:
        ints.append({'node': 'memória limpa', 'delta': 0.9})
        ints.append({'node': 'risco de perda de contexto', 'delta': 0.4})
    if 'curate' in k:
        ints.append({'node': 'qualidade da memória', 'delta': 0.8})
    if 'execute_procedure_active' in k:
        ints.append({'node': 'execução ativa', 'delta': 0.9})
        ints.append({'node': 'risco operacional', 'delta': 0.5})
    if 'auto_resolve_conflicts' in k:
        ints.append({'node': 'consistência do conhecimento', 'delta': 0.7})
        ints.append({'node': 'risco de decisão errada', 'delta': 0.35})
    if 'notify' in t:
        ints.append({'node': 'interrupção humana', 'delta': 0.3})
    return ints or [{'node': 'ação genérica', 'delta': 0.2}]
