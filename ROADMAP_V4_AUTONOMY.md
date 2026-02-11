# UltronPro Roadmap V4 — Autonomia Coerente e Auto-Regulada

Objetivo: evoluir de "autonomia operacional" para "autonomia robusta com autoconsciência funcional avançada".

## Princípios
- Safety-first (integrity/policy/causal sempre na frente de execução crítica)
- Evidência > opinião (sandbox/sql/source-verify como ciclo padrão)
- Continuidade temporal (identidade e compromissos entre dias)
- Custo controlado (policy por tipo de tarefa)

## Horizonte e metas
- 2 semanas: autoconsciência funcional ~85%
- 4–8 semanas: autonomia confiável ~90%

---

## M1 — Homeostase Cognitiva (Semana 1)
### Entrega
- Novo módulo `homeostasis.py` com variáveis vitais:
  - `coherence_score`
  - `uncertainty_load`
  - `memory_pressure`
  - `goal_drift`
  - `contradiction_stress`
  - `energy_budget`
- Loop de regulação no `autonomy_loop`:
  - modo normal / modo repair / modo conservative

### KPI
- Redução >= 25% de ações bloqueadas por inconsistência recorrente (janela 7 dias)
- Redução >= 20% de conflitos em stalemate (>3 ciclos)

### Critério de aceite
- Endpoint `/api/homeostasis/status`
- Eventos de transição de modo auditáveis no feed

---

## M2 — Self-Model Causal (Semana 1–2)
### Entrega
- Expandir `self_model.py` para guardar relações causais sobre o próprio comportamento:
  - `strategy -> outcome`
  - `task_type -> error_rate`
  - `budget_profile -> throughput`
- Atualização automática por resultado das ações

### KPI
- Melhora >= 15% no `decision_quality` médio (7 dias)
- Queda >= 20% em retrabalho (ações repetidas sem progresso)

### Critério de aceite
- Endpoint `/api/self-model/causal`
- Planner usando causal priors na ordenação de ações

---

## M3 — Loop de Identidade e Compromisso Diário (Semana 2)
### Entrega
- Rotina diária de identidade:
  - promessas de ontem
  - cumprimento/falha
  - ajuste de protocolo
- Persistência em memória longa (`MEMORY.md` + índice interno)

### KPI
- >= 80% das promessas com status explícito no dia seguinte
- >= 70% dos dias com "next protocol update" registrado

### Critério de aceite
- Endpoint `/api/identity/daily-review`
- Registro diário com checksum de compromissos

---

## M4 — Deliberação Contrafactual Obrigatória (Semana 2–3)
### Entrega
- Para classes críticas, exigir:
  - 2–3 planos alternativos
  - score custo/risco/benefício
  - justificativa de rejeição dos planos perdedores
- Integrar com `itc.py`, `integrity.py` e `neurosym.py`

### KPI
- Redução >= 30% de decisões críticas revertidas pós-execução
- Aumento >= 20% de fidelidade neuro-simbólica em ações críticas

### Critério de aceite
- Endpoint `/api/deliberation/critical-report`
- Evidence trail completo por decisão crítica

---

## M5 — Grounding Empírico Unificado (Semana 3)
### Entrega
- Ciclo padrão obrigatório para claims relevantes:
  - hipótese -> teste (`sandbox/sql/source_verify`) -> atualização de crença
- Score de confiabilidade por claim

### KPI
- Queda >= 35% de respostas vagas em benchmark de conhecimento
- Aumento >= 25% no benchmark `/api/benchmark/lightrag`

### Critério de aceite
- Endpoint `/api/grounding/claim-check`
- Registro de claim com fonte + teste + conclusão

---

## M6 — Governança de Autonomia por Classe de Ação (Semana 3–4)
### Entrega
- Matriz formal:
  - Auto-executável
  - Auto-executável com prova
  - Aprovação humana obrigatória
- Veto hard por integrity para violações de classe

### KPI
- 0 execuções críticas fora de política
- 100% ações críticas com trilha de decisão auditável

### Critério de aceite
- Endpoint `/api/governance/matrix`
- Relatório semanal de conformidade

---

## Plano de implementação (ordem)
1. M1 Homeostase
2. M2 Self-model causal
3. M4 Deliberação contrafactual (críticas)
4. M5 Grounding unificado
5. M3 Identidade diária
6. M6 Governança final

> Nota: M4 antes de M3 para reduzir risco operacional cedo.

---

## Definição de pronto (DoD)
- Endpoint + teste funcional + evento de auditoria
- KPI com medição em janela 7d
- Fallback de baixo custo (sem explodir tokens)
- Documentado no README/ops notes
