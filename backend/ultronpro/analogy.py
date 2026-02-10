from __future__ import annotations

import json
from typing import Any

from ultronpro import llm


def propose_analogy(problem_text: str, target_domain: str | None = None, context_snippets: list[str] | None = None) -> dict[str, Any] | None:
    txt = (problem_text or '').strip()
    if len(txt) < 12:
        return None
    ctx = "\n".join((context_snippets or [])[:5])

    prompt = f"""Given the target problem below, propose ONE useful cross-domain analogy.
Return ONLY JSON with keys:
source_domain, target_domain, source_concept, target_concept, mapping (object), inference_rule, confidence (0..1), notes.
Target domain hint: {target_domain or 'general'}
Problem:\n{txt[:2200]}
Context:\n{ctx[:1800]}
"""
    try:
        raw = llm.complete(prompt, strategy='cheap', json_mode=True)
        d = json.loads(raw) if raw else {}
        if isinstance(d, dict) and d.get('mapping'):
            d['confidence'] = max(0.0, min(1.0, float(d.get('confidence') or 0.55)))
            return d
    except Exception:
        pass

    # fallback determinístico (baixo custo, sem LLM)
    tl = txt.lower()
    if 'gravidade' in tl and ('maré' in tl or 'marea' in tl):
        return {
            'source_domain': 'física',
            'target_domain': target_domain or 'oceanografia',
            'source_concept': 'força gravitacional',
            'target_concept': 'variação de maré',
            'mapping': {
                'massa_corpo': 'intensidade_atração',
                'distância': 'amplitude_efeito',
                'atração_diferencial': 'elevação_rebaixamento_nível'
            },
            'inference_rule': 'Se a atração diferencial aumenta, a amplitude da maré tende a aumentar sob mesmas condições locais.',
            'confidence': 0.62,
            'notes': 'fallback heuristic',
        }

    # fallback genérico
    return {
        'source_domain': 'sistemas_dinâmicos',
        'target_domain': target_domain or 'general',
        'source_concept': 'forças e restrições',
        'target_concept': 'estado do problema',
        'mapping': {
            'força': 'pressão causal',
            'restrição': 'limite operacional',
            'equilíbrio': 'solução estável'
        },
        'inference_rule': 'Mapear forças e restrições do domínio fonte para estimar estados estáveis no domínio alvo.',
        'confidence': 0.51,
        'notes': 'generic fallback',
    }


def validate_analogy(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {'valid': False, 'confidence': 0.0, 'reasons': ['empty candidate']}

    conf = float(candidate.get('confidence') or 0.5)
    mapping = candidate.get('mapping')
    reasons: list[str] = []

    if not isinstance(mapping, dict) or len(mapping) < 1:
        return {'valid': False, 'confidence': 0.0, 'reasons': ['mapping missing']}

    if len(mapping) >= 2:
        conf += 0.08
    if candidate.get('inference_rule'):
        conf += 0.06
    if (candidate.get('source_domain') or '').lower() == (candidate.get('target_domain') or '').lower():
        conf -= 0.15
        reasons.append('domains too similar')

    conf = max(0.0, min(1.0, conf))
    valid = conf >= 0.5
    if valid:
        reasons.append('structural mapping accepted')
    else:
        reasons.append('low confidence')
    return {'valid': valid, 'confidence': conf, 'reasons': reasons}


def apply_analogy(candidate: dict[str, Any], problem_text: str) -> dict[str, Any]:
    mapping = candidate.get('mapping') or {}
    rule = (candidate.get('inference_rule') or '').strip()
    if not rule:
        # small deterministic fallback
        pairs = [f"{k}->{v}" for k, v in list(mapping.items())[:4]]
        rule = f"Transferir estrutura relacional mapeando: {', '.join(pairs)}"

    hypothesis = f"Analogia aplicada: {rule}. Problema alvo: {(problem_text or '')[:280]}"
    return {
        'derived_rule': rule,
        'hypothesis': hypothesis,
        'mapping': mapping,
    }
