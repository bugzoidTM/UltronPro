from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Goal:
    id: int
    title: str
    description: str | None
    status: str
    priority: int


class GoalPlanner:
    """Goal creation & planning (MVP scaffolding).

    Reality check: true long-horizon autonomous planning requires a strong model.
    This module provides deterministic structure + room to plug an LLM when available.
    """

    def propose_goals(self, experiences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        text = "\n".join((e.get("text") or "") for e in experiences[-10:]).lower()

        goals = []
        if "ultron" in text or "agi" in text or "curios" in text:
            goals.append({
                "title": "Construir núcleo de curiosidade + síntese (tese↔antítese↔síntese)",
                "description": "Manter perguntas úteis e resolver contradições com evidência.",
                "priority": 5,
            })
            goals.append({
                "title": "Melhorar ingestão multimodal (imagens/áudio) e extração",
                "description": "Aceitar anexos e registrar metadados; extrair texto quando possível.",
                "priority": 4,
            })

        # Always keep one generic goal
        goals.append({
            "title": "Aprender com tutores: registrar conhecimento como triplas com evidência",
            "description": "Aumentar confiança por suporte, reduzir por contradição.",
            "priority": 4,
        })

        # De-dupe by title
        seen = set()
        out = []
        for g in goals:
            t = g["title"].strip()
            if t in seen:
                continue
            seen.add(t)
            out.append(g)
        return out[:5]
