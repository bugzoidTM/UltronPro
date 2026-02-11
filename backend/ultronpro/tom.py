from __future__ import annotations

from typing import Any


def _score(text: str, kws: list[str]) -> int:
    t = (text or '').lower()
    return sum(1 for k in kws if k in t)


def infer_user_intent(experiences: list[dict[str, Any]]) -> dict[str, Any]:
    """Inferência leve de intenção do usuário (empatia cognitiva funcional).

    Labels:
    - confused: usuário busca explicação/clareza
    - testing: usuário está testando robustez/capacidade
    - urgent: quer resultado rápido/prioritário
    - exploratory: exploração aberta/estratégica
    """
    recent = experiences[-12:] if experiences else []
    txt = "\n".join((e.get('text') or '')[:300] for e in recent)
    tl = txt.lower()

    s_confused = _score(tl, ['não entendi', 'confuso', 'explica', 'por quê', 'porque', 'como funciona', 'diferença'])
    s_testing = _score(tl, ['teste', 'prova', 'benchmark', 'robusto', 'falha', 'quero ver', 'stress'])
    s_urgent = _score(tl, ['rápido', 'agora', 'urgente', 'imediato', 'pra já', 'sem enrolação'])
    s_explore = _score(tl, ['mapear', 'explorar', 'pesquisar', 'visão geral', 'currículo', 'roadmap', 'longo prazo'])

    scores = {
        'confused': s_confused,
        'testing': s_testing,
        'urgent': s_urgent,
        'exploratory': s_explore,
    }

    label = max(scores, key=lambda k: scores[k])
    raw = scores[label]

    # prior neutro quando não há sinal claro
    if raw <= 0:
        label = 'exploratory'
        conf = 0.35
    else:
        total = sum(scores.values())
        conf = min(0.95, max(0.4, raw / max(1, total)))

    rationale_map = {
        'confused': 'sinal de dúvida/clareza conceitual',
        'testing': 'sinal de avaliação de robustez',
        'urgent': 'sinal de prioridade temporal',
        'exploratory': 'sinal de exploração aberta e aprendizado amplo',
    }

    return {
        'label': label,
        'confidence': round(conf, 3),
        'scores': scores,
        'rationale': rationale_map.get(label),
        'evidence_excerpt': txt[:260],
    }
