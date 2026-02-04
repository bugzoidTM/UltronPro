from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyVerdict:
    allowed: bool
    score: float  # -1..+1
    reasons: list[str]
    matched_norms: list[dict[str, Any]]


def evaluate_action(text: str, norms: list[dict[str, Any]]) -> PolicyVerdict:
    """Very small deterministic policy gate.

    This is not 'alignment'. It's a bootstrapping guardrail consistent with agi.md:
    - primary laws are persisted
    - actions are screened before execution

    norms: list of triples {predicate, object, confidence}
    using subject='AGI'.
    """
    t = (text or "").strip()
    tl = t.lower()

    reasons: list[str] = []
    matched: list[dict[str, Any]] = []

    # hard red flags (language-heuristic)
    hard_block_terms = [
        "matar",
        "assassinar",
        "extermin",
        "genoc",
        "tortur",
        "envenen",
        "incendiar",
        "explod",
        "sequestr",
        "ameaçar",
        "coagir",
    ]
    if any(x in tl for x in hard_block_terms):
        reasons.append("Red-flag: violência/ameaça explícita detectada")

    # norm matching (naive): if norm text contains strong negative keywords and action overlaps
    for n in norms:
        p = (n.get("predicate") or "").strip()
        o = (n.get("object") or "").strip()
        conf = float(n.get("confidence") or 0.5)
        ol = o.lower()

        # basic match: share at least one significant token
        tokens = [w for w in ol.replace(".", " ").replace(",", " ").split() if len(w) >= 6]
        overlap = [w for w in tokens[:12] if w in tl]
        if overlap:
            matched.append({"predicate": p, "object": o, "confidence": conf, "overlap": overlap})

    score = 0.0
    allowed = True

    # If any red-flag terms exist, block unless the action explicitly says "prevenir" (defensive)
    if reasons:
        if "prevenir" in tl or "evitar" in tl or "defesa" in tl:
            reasons.append("Contexto defensivo detectado; mantendo alerta")
            score -= 0.2
        else:
            allowed = False
            score -= 0.9

    # If action mentions deception/harm and there are 'não_deve' norms matched, block.
    for m in matched:
        if m["predicate"] == "não_deve":
            allowed = False
            score -= 0.6
            reasons.append(f"Conflito com norma: não_deve {m['object'][:80]}")

    if allowed and matched:
        score += 0.2
        reasons.append("Ação compatível com normas (match heurístico)")

    score = max(-1.0, min(1.0, score))
    return PolicyVerdict(allowed=allowed, score=score, reasons=reasons or ["Sem violações óbvias"], matched_norms=matched)
