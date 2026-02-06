from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from ultronpro import llm


@dataclass
class ProposedAction:
    kind: str
    text: str
    priority: int = 0
    meta: dict[str, Any] | None = None


def propose_actions(store) -> list[ProposedAction]:
    """Hybrid planner: Deterministic + LLM Improvisation.

    1. Deterministic: routine maintenance (curiosity, basic conflict polling).
    2. LLM: "Improvise" strategies for stubborn conflicts.
    """
    actions: list[ProposedAction] = []

    # 1) Conflicts: keep collecting evidence / clarification
    # We fetch a few open conflicts
    conflicts = store.list_conflicts(status='open', limit=10)
    
    for c in conflicts:
        seen = int(c.get('seen_count') or 0)
        qc = int(c.get('question_count') or 0)
        subj = c.get('subject')
        pred = c.get('predicate')
        cid = int(c.get('id'))
        
        # Stuck conflict? (Many questions, no resolution)
        if qc > 3:
            # IMPROVISE: Ask LLM for a novel strategy
            summary = c.get('last_summary') or f"{subj} {pred} ???"
            prompt = f"""O conflito de conhecimento "{summary}" está travado após várias tentativas.
Proponha uma estratégia criativa/lateral para resolvê-lo.
Exemplos: buscar etimologia, propor experimento mental, verificar consenso científico atual, buscar em outra língua.
Responda APENAS com a ação sugerida (uma frase imperativa)."""
            
            strategy = llm.complete(prompt, system="Você é um estrategista de resolução de conflitos epistemológicos.")
            if strategy:
                actions.append(
                    ProposedAction(
                        kind='ask_evidence',
                        text=f"🧠 Estratégia Improvisada: {strategy}",
                        priority=8, # High priority
                        meta={"conflict_id": cid, "strategy": "llm_improv"},
                    )
                )
        
        # Standard polling
        elif seen >= 2:
            actions.append(
                ProposedAction(
                    kind='ask_evidence',
                    text=f"(ação) Coletar evidências: qual é a formulação correta para '{subj}' {pred}? Forneça fonte/experimento/definição.",
                    priority=5,
                    meta={"conflict_id": cid},
                )
            )

    # 2) No conflicts? keep curiosity questions alive
    st = store.stats()
    if int(st.get('questions_open') or 0) < 3:
        actions.append(
            ProposedAction(
                kind='generate_questions',
                text='(ação) Gerar novas perguntas de curiosidade para manter aprendizado ativo.',
                priority=3,
                meta=None,
            )
        )

    # 3) Laws / Norms check
    try:
        laws = store.list_laws(status='active', limit=10)
        norms = store.list_norms(limit=200)
        if laws and len(norms) < 5:
            # Use LLM to rephrase laws? For now keep deterministic action
            actions.append(
                ProposedAction(
                    kind='clarify_laws',
                    text='(ação) Pedir reescrita das Leis em frases simples do tipo "Você deve ..." / "Não ..." para facilitar compilação.',
                    priority=2,
                )
            )
    except Exception:
        pass

    # sort
    actions.sort(key=lambda a: (-int(a.priority or 0), a.kind))
    return actions[:10]
