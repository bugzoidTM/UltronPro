# UltronPRO

UltronPRO é uma arquitetura de agente cognitivo autônomo com foco em:
- aprendizado contínuo,
- planejamento orientado por objetivos,
- execução segura com guardrails simbólicos,
- operação de longo horizonte (horas/dias/semanas).

> Base conceitual: `/root/.openclaw/agi.md`

---

## Visão geral da arquitetura

UltronPRO combina quatro camadas principais:

1. **Percepção/Aprendizado (neural)**
   - ingestão de experiências,
   - extração semântica,
   - generalização por LLM,
   - deliberação System-2 (ITC).

2. **Estrutura simbólica**
   - triplas e evidências,
   - conflitos tese↔antítese↔síntese,
   - regras/normas,
   - world model causal.

3. **Agência/autonomia**
   - fila de ações com prioridade/cooldown/TTL,
   - goals, milestones, subgoals (DAG),
   - project kernel com recuperação de falhas,
   - roteamento de ferramentas com fallback.

4. **Segurança e governança**
   - policy guardrails,
   - causal precheck,
   - integrity gate (consenso neural+simbólico),
   - auditoria neuro-simbólica (proofs).

---

## Capacidades implementadas (detalhado)

## 1) Conhecimento, memória e curiosidade
- Ingestão de texto e arquivos.
- Extração de triplas (LLM + fallback).
- Curiosidade ativa com fila adaptativa de perguntas.
- Curadoria de memória e destilação semântica.
- Esquecimento ativo de baixa utilidade.
- Memória por projeto (`project_memory_index.json`).

## 2) Conflitos e síntese
- Detecção/persistência de conflitos no grafo.
- Priorização de conflitos persistentes.
- Auto-resolução com evidência + confiança.
- Estratégias de improviso para conflitos travados.
- Escalonamento para revisão humana quando necessário.

## 3) Goals proativos (Impulso de Vida)
- Goal-first planner (quando não há emergência).
- Milestones semanais e progressão incremental.
- Geração de ações orientadas ao objetivo ativo.
- Ações proativas de busca (ex.: absorção no LightRAG por domínio do objetivo).

## 4) Sub-objetivos e planejamento hierárquico
- DAG de subgoals persistente (`subgoal_dag.json`).
- Decomposição automática de meta em nós dependentes.
- Marcação de estado por nó (`open/active/done`).

## 5) System-2 / Inference-Time Compute
- Episódios deliberativos multi-step.
- Verificação de subresultado por passo.
- Correção autônoma quando passo falha.
- Orquestração com RL leve (bandit epsilon-greedy).
- Métricas de reward/qualidade/latência por episódio.

## 6) Long-horizon + Project management
- Missões de longo horizonte com checkpoints.
- Project kernel com KPI e blockers.
- Recovery playbooks (timeout, tool_failure, stalemate, regressão KPI).
- Cadência de gestão (`project_management_cycle`) com próximos passos automáticos.
- Ciclo experimental técnico (`project_experiment_cycle`) para validação de hipótese.

## 7) Ferramentas de ambiente (sandbox)
- Escrita/leitura/listagem de arquivos em sandbox (`/app/data/sandbox`).
- Execução real de código Python com timeout (`python3 -I`).
- Histórico de execuções.

## 8) LightRAG absorption (multi-domínio)
- Absorção geral do LightRAG por domínios (`python, systems, database, ai`, etc.).
- Ingestão no Ultron + tentativa de extração para grafo local.
- Benchmarks dedicados:
  - `/api/benchmark/python`
  - `/api/benchmark/lightrag`

## 9) Neuroplasticidade controlada
- Proposta → shadow eval → ativação/reversão.
- Auto-promote por gate rolling.
- Auto-revert por degradação ou falta de ganho sustentado (7/14 dias).
- Canary rollout por `canary_ratio`.

## 10) Neuro-simbólico e explicabilidade
- Proof objects por decisão crítica (`neurosym_proofs.json`).
- Consistency checker simbólico.
- Fidelity score (explicação vs ação executada).

## 11) Integrity Gate V1
- Regras rígidas de integridade (`integrity_rules.json`).
- Estado de vetos e prevenção de alucinação (`integrity_state.json`).
- Consenso dual:
  - confiança deliberativa (neural)
  - consistência simbólica
  - prova exigida em ações críticas
  - causal precheck quando obrigatório

## 12) Auto-modelo e personalidade runtime
- Auto-biografia persistente (`self_model.json`).
- Personalidade dinâmica em runtime (valence/arousal/goal/purpose).
- Few-shot dinâmico com exemplos de estilo (`persona_examples.json`).
- Injeção automática no system prompt antes de cada chamada LLM.

---

## Frontend (UI)

A UI inclui:
- overview operacional,
- feed em tempo real (SSE),
- grafo de conhecimento,
- conflitos, curiosidade, insights,
- missões de longo horizonte,
- persona runtime.

### Otimizações de performance
- polling adaptativo por visibilidade da aba,
- limites de itens no feed DOM,
- limites de nós/arestas no grafo,
- timeouts e cache para endpoints lentos,
- preview de conflitos no overview (sem precisar abrir aba de conflitos).

---

## Segurança e governança

- Policy-based action filtering.
- Causal precheck para ações sensíveis.
- Integrity veto para decisões inseguras/incoerentes.
- Auditoria com eventos e provas neuro-simbólicas.
- Execução externa controlada (prepare/confirm/execute).

---

## Deploy e dados

## Execução local (dev)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn ultronpro.main:app --reload --host 0.0.0.0 --port 8000
```

UI: `http://localhost:8000`

## Deploy (Swarm)
Arquivos em `deploy/`.

Dados persistidos em `/app/data` (SQLite + estados cognitivos + benchmarks + persona + sandbox).

---

## Endpoints principais (resumo)

## Núcleo
- `GET /api/status`
- `GET /api/agi-mode`
- `POST /api/autonomy/tick`

## Goals / Subgoals / Missões / Projetos
- `GET /api/goals`
- `POST /api/goals/refresh`
- `POST /api/subgoals/plan`
- `GET /api/subgoals`
- `POST /api/horizon/missions`
- `GET /api/horizon/missions`
- `POST /api/projects`
- `GET /api/projects`
- `POST /api/projects/tick`
- `GET /api/projects/{project_id}/brief`

## LLM/Reasoning
- `POST /api/itc/run`
- `GET /api/itc/status`
- `GET /api/itc/policy`

## Knowledge / LightRAG
- `POST /api/lightrag/absorb`
- `POST /api/python/absorb`
- `GET /api/benchmark/python`
- `GET /api/benchmark/lightrag`

## Conflitos
- `GET /api/conflicts`
- `POST /api/conflicts/auto-resolve`

## Neuro-simbólico / Integridade
- `GET /api/neurosym/proofs`
- `GET /api/neurosym/consistency`
- `GET /api/neurosym/fidelity`
- `GET /api/integrity/status`
- `POST /api/integrity/rules`

## Sandbox
- `POST /api/sandbox/write`
- `GET /api/sandbox/read`
- `GET /api/sandbox/files`
- `POST /api/sandbox/run-python`
- `GET /api/sandbox/history`

## Persona / Self-model
- `GET /api/self-model/status`
- `POST /api/self-model/refresh`
- `GET /api/persona/status`
- `GET /api/persona/examples`
- `POST /api/persona/examples`
- `POST /api/persona/config`

---

## Limites atuais

- A qualidade final de respostas depende da qualidade/estrutura da base LightRAG.
- O sistema é robusto, mas não substitui revisão humana em operações críticas.
- “Autoconsciência” é funcional/operacional, não evidência de qualia fenomenológica.

---

## Roadmap curto sugerido
- Painéis de observabilidade para Integrity/Router/ProjectOps.
- Benchmark contínuo por domínio com metas de cobertura.
- Melhor parser estruturado para ingestão de entidades/relacionamentos do LightRAG.
- Hardening adicional da sandbox (cgroups/seccomp/import policy).