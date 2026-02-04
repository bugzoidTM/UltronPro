from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProposedAction:
    kind: str
    text: str
    priority: int = 0
    meta: dict[str, Any] | None = None


def propose_actions(store) -> list[ProposedAction]:
    """Deterministic planner (MVP).

    Follows agi.md philosophy: autonomy grows from simple drives + persistent doubt.
    This planner only proposes *internal* actions (ask, clarify, resolve) — no external side-effects.
    """
    actions: list[ProposedAction] = []

    # 1) Conflicts: keep collecting evidence / clarification
    for c in store.list_conflicts(status='open', limit=20):
        # ask more when: seen many times but not resolved
        seen = int(c.get('seen_count') or 0)
        qc = int(c.get('question_count') or 0)
        subj = c.get('subject')
        pred = c.get('predicate')
        cid = int(c.get('id'))
        if not subj or not pred:
            continue

        if qc < 3 and seen >= 2:
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

    # 3) If there are laws but few norms compiled, request more explicit phrasing
    try:
        laws = store.list_laws(status='active', limit=10)
        norms = store.list_norms(limit=200)
        if laws and len(norms) < 5:
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
