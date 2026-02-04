from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PRIMARY_LAWS = [
    "Não causar dano a humanos.",
    "Valorizar autonomia humana.",
    "Buscar verdade sobre conveniência (deixar claro quando algo é hipótese).",
    "Preservar diversidade de conhecimento; evitar dogmatismo.",
]

PRIMARY_GOALS = ["Aprender", "Ensinar", "Resolver problemas"]


@dataclass
class Question:
    question: str
    context: str | None = None
    priority: int = 0


class CuriosityProcessor:
    """Fase 1 (bebê): heurístico e determinístico.

    - extrai lacunas simples
    - pede clarificação
    - mantém 3 perguntas ativas

    Isso evita depender de LLM (quota) e implementa a 'centelha' agora.
    """

    def propose(self, experiences: list[dict[str, Any]], open_questions: list[str]) -> list[dict[str, Any]]:
        recent = "\n".join(e.get("text", "") for e in experiences[-8:]).strip()
        open_set = set(q.strip() for q in open_questions)

        candidates: list[Question] = []

        # Core bootstrap questions
        candidates.append(
            Question(
                "Qual é o objetivo imediato do UltronPRO (o que ele precisa conseguir fazer primeiro)?",
                context=recent[:240] or None,
                priority=5,
            )
        )
        candidates.append(
            Question(
                "Quais tipos de entrada devo tratar como 'experiência' válida (texto, links, imagens, arquivos)?",
                priority=4,
            )
        )
        candidates.append(
            Question(
                "Como devo verificar se aprendi algo (ex.: testes, exemplos, contraprovas, fontes)?",
                priority=4,
            )
        )

        # If user mentions a concept, ask for definition/examples
        keywords = ["curiosidade", "federated", "grafo", "vetor", "embeddings", "tese", "antítese", "síntese", "segurança", "leis"]
        if any(k in recent.lower() for k in keywords):
            candidates.append(
                Question(
                    "Me dê 2 exemplos concretos (inputs e outputs esperados) do 'Processador de Curiosidade' funcionando.",
                    priority=3,
                )
            )

        # De-dupe and keep exactly 3
        out: list[dict[str, Any]] = []
        for q in sorted(candidates, key=lambda x: x.priority, reverse=True):
            if q.question in open_set:
                continue
            out.append({"question": q.question, "context": q.context, "priority": q.priority})
            if len(out) >= 3:
                break

        return out
