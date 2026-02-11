from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from ultronpro import llm, tom, mission_control, self_model


@dataclass
class ProposedAction:
    kind: str
    text: str
    priority: int = 0
    meta: dict[str, Any] | None = None


def propose_actions(store) -> list[ProposedAction]:
    """Hybrid planner with Life-Drive:

    1. Goal-first (when not emergency): inject active goals and generate proactive actions.
    2. Deterministic maintenance (curiosity/laws/conflicts).
    3. LLM improvisation (conflicts + goal strategies).
    """
    actions: list[ProposedAction] = []

    # Teoria da Mente (empatia cognitiva): inferir intenção do humano
    recent_exp = store.list_experiences(limit=20)
    intent = tom.infer_user_intent(recent_exp)
    ilabel = intent.get('label')
    iconf = float(intent.get('confidence') or 0.0)

    emergency = (ilabel == 'urgent' and iconf >= 0.5)

    # Impulso de Vida: objetivo ativo injeta direção proativa quando não há emergência
    active_goal = None
    try:
        active_goal = store.get_active_goal()
    except Exception:
        active_goal = None

    if active_goal and not emergency:
        gtitle = str(active_goal.get('title') or '').strip()
        gdesc = str(active_goal.get('description') or '').strip()
        gtxt = f"{gtitle} {gdesc}".lower()

        # ações proativas determinísticas orientadas ao objetivo
        if any(k in gtxt for k in ['python', 'program', 'codigo', 'código']):
            actions.append(
                ProposedAction(
                    kind='absorb_lightrag_general',
                    text='(impulso-vida) Absorver conhecimento Python no LightRAG para avançar objetivo ativo.',
                    priority=9,
                    meta={'domains': 'python', 'max_topics': 20, 'doc_limit': 12, 'goal_id': active_goal.get('id')},
                )
            )
            actions.append(
                ProposedAction(
                    kind='execute_python_sandbox',
                    text='(impulso-vida) Validar hipótese com código Python em sandbox (prova executável).',
                    priority=9,
                    meta={
                        'goal_id': active_goal.get('id'),
                        'code': "print('sandbox-check: python goal active')\nprint(sum(i*i for i in range(10)))",
                        'timeout_sec': 10,
                    },
                )
            )
            actions.append(
                ProposedAction(
                    kind='ask_evidence',
                    text='(impulso-vida) Pesquisar no LightRAG tópicos críticos de Python ligados ao objetivo ativo e sintetizar plano de estudo/execução.',
                    priority=8,
                    meta={'goal_id': active_goal.get('id'), 'goal_title': gtitle},
                )
            )
        else:
            if any(k in gtxt for k in ['otimizar', 'database', 'banco', 'sql', 'desempenho', 'performance']):
                actions.append(
                    ProposedAction(
                        kind='execute_python_sandbox',
                        text='(impulso-vida) Rodar script Python sandbox para medir hipótese técnica do objetivo.',
                        priority=8,
                        meta={
                            'goal_id': active_goal.get('id'),
                            'code': "import sqlite3, os\nprint('db-path', os.getenv('ULTRONPRO_DB_PATH','/app/data/ultron.db'))\nprint('probe-ok')",
                            'timeout_sec': 12,
                        },
                    )
                )
            actions.append(
                ProposedAction(
                    kind='ask_evidence',
                    text=f"(impulso-vida) Próximo passo objetivo para avançar '{gtitle}' com menor custo e evidência verificável.",
                    priority=8,
                    meta={'goal_id': active_goal.get('id'), 'goal_title': gtitle},
                )
            )

        # improvisação LLM guiada por objetivo (não só conflito)
        try:
            prompt = f"""Objetivo ativo: {gtitle}\nDescrição: {gdesc}\n
Proponha UMA ação proativa de alto impacto para avançar o objetivo hoje,
com foco em busca de conhecimento e verificação factual.
Responda APENAS com uma frase imperativa."""
            strat = llm.complete(prompt, system='Você é um planner proativo orientado por objetivos.')
            if strat:
                actions.append(
                    ProposedAction(
                        kind='ask_evidence',
                        text=f"🧭 Estratégia Pró-Objetivo: {strat}",
                        priority=8,
                        meta={'goal_id': active_goal.get('id'), 'strategy': 'goal_improv'},
                    )
                )
        except Exception:
            pass

    if ilabel == 'confused':
        actions.append(
            ProposedAction(
                kind='ask_evidence',
                text='(ação-TOM) Reformular explicação em passos simples e verificar entendimento do humano com uma pergunta de checagem.',
                priority=7,
                meta={'tom_intent': ilabel, 'tom_confidence': iconf},
            )
        )
    elif ilabel == 'testing':
        actions.append(
            ProposedAction(
                kind='ask_evidence',
                text='(ação-TOM) Fornecer resposta auditável com critérios de validação (o que funciona, limite e como testar).',
                priority=7,
                meta={'tom_intent': ilabel, 'tom_confidence': iconf},
            )
        )
    elif ilabel == 'urgent':
        actions.append(
            ProposedAction(
                kind='auto_resolve_conflicts',
                text='(ação-TOM) Priorizar resolução rápida do bloqueio principal antes de exploração ampla.',
                priority=8,
                meta={'tom_intent': ilabel, 'tom_confidence': iconf},
            )
        )
    else:  # exploratory
        actions.append(
            ProposedAction(
                kind='generate_analogy_hypothesis',
                text='(ação-TOM) Expandir entendimento com analogia estrutural de domínio adjacente.',
                priority=5,
                meta={'tom_intent': ilabel, 'tom_confidence': iconf, 'problem_text': 'exploração de contexto atual', 'target_domain': 'general'},
            )
        )

    # 1) Conflicts: keep collecting evidence / clarification
    # Goal-first policy: when no emergency and active goal exists, conflicts are handled but with lower priority.
    conflicts = store.list_conflicts(status='open', limit=10)
    conflict_base_priority = 4 if (active_goal and not emergency) else 6

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
                        priority=max(5, conflict_base_priority + 1),
                        meta={"conflict_id": cid, "strategy": "llm_improv"},
                    )
                )
        
        # Standard polling
        elif seen >= 2:
            actions.append(
                ProposedAction(
                    kind='ask_evidence',
                    text=f"(ação) Coletar evidências: qual é a formulação correta para '{subj}' {pred}? Forneça fonte/experimento/definição.",
                    priority=conflict_base_priority,
                    meta={"conflict_id": cid},
                )
            )

            # grounding: verificação headless sob demanda para reduzir alucinação em conflito persistente
            if subj and pred and seen >= 2:
                topic = str(subj).strip().replace(' ', '_')
                actions.append(
                    ProposedAction(
                        kind='verify_source_headless',
                        text=f"(ação) Verificar fonte canônica para conflito #{cid} via fetch headless.",
                        priority=max(5, conflict_base_priority + 1),
                        meta={
                            'conflict_id': cid,
                            'url': f'https://en.wikipedia.org/wiki/{topic}',
                            'max_chars': 6000,
                        },
                    )
                )
                actions.append(
                    ProposedAction(
                        kind='ground_claim_check',
                        text=f"(ação) Validar claim crítica do conflito #{cid} com grounding empírico (source+python/sql quando aplicável).",
                        priority=max(5, conflict_base_priority + 1),
                        meta={
                            'conflict_id': cid,
                            'claim': f"{subj} {pred}",
                            'url': f'https://en.wikipedia.org/wiki/{topic}',
                            'require_reliability': 0.55,
                        },
                    )
                )

            # transferência ativa por analogia quando conflito persiste
            if seen >= 3:
                q = f"{subj} {pred}"
                actions.append(
                    ProposedAction(
                        kind='generate_analogy_hypothesis',
                        text=f"(ação) Tentar analogia estrutural para resolver conflito: {subj} {pred}",
                        priority=max(5, conflict_base_priority + 1),
                        meta={"conflict_id": cid, "problem_text": q, "target_domain": str(pred)},
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

    # 4) Mission Control delegation awareness (Phase C)
    try:
        m_tasks = mission_control.list_tasks(limit=40)
        hot = [t for t in m_tasks if str(t.get('status') or '') in ('assigned', 'in_progress', 'blocked')]
        if hot:
            top = hot[-1]
            ttitle = str(top.get('title') or '')
            tstatus = str(top.get('status') or '')
            ass = ','.join(top.get('assignees') or [])
            actions.append(
                ProposedAction(
                    kind='ask_evidence',
                    text=f"(mission-control) Atualizar task '{ttitle}' [{tstatus}] com próximo passo e evidência objetiva (assignees={ass}).",
                    priority=6,
                    meta={'mission_task_id': top.get('id'), 'mission_status': tstatus, 'assignees': top.get('assignees') or []},
                )
            )
    except Exception:
        pass

    # 5) Causal priors from self-model (M2)
    try:
        by_strategy = self_model.best_strategy_scores(limit=60)
        for a in actions:
            sr = by_strategy.get(str(a.kind), None)
            if sr is None:
                continue
            if sr >= 0.72:
                a.priority = int(a.priority or 0) + 2
            elif sr >= 0.58:
                a.priority = int(a.priority or 0) + 1
            elif sr <= 0.32:
                a.priority = int(a.priority or 0) - 2
            elif sr <= 0.45:
                a.priority = int(a.priority or 0) - 1
    except Exception:
        pass

    # sort
    actions.sort(key=lambda a: (-int(a.priority or 0), a.kind))
    return actions[:10]
