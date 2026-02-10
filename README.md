# UltronPRO

UltronPRO é um backend+frontend para evolução incremental de uma AGI orientada por:
- ingestão contínua de experiências,
- grafo de conhecimento (triplas + evidências),
- curiosidade ativa,
- detecção/gestão de conflitos,
- autonomia com guardrails.

Base conceitual: `/root/.openclaw/agi.md`.

---

## Estado atual (fases)
A fase exibida no frontend é dinâmica e baseada no `AGI mode`:
- Fase 1 (Bebê): < 35%
- Fase 2 (Criança): 35–54.9%
- Fase 3 (Adolescente): 55–74.9%
- Fase 4 (Adulta): >= 75%

`AGI mode` é calculado por pilares (learning, curiosity, autonomy, synthesis, goals, curation).

---

## Principais capacidades implementadas

### 1) Conhecimento + Curiosidade Adaptativa
- Ingestão de texto/arquivo.
- Extração de triplas via LLM + fallback regex.
- Curiosidade com templates adaptativos + feedback de eficácia.
- Fila adaptativa de perguntas com metadados (`template_id`, `concept`).
- Refresh com alvo dinâmico (evita spam de perguntas abertas).

### 2) Conflitos (tese ↔ antítese ↔ síntese)
- Persistência e priorização de conflitos.
- Priorização por persistência + impacto no grafo.
- Auto-resolução com score ponderado por:
  - confiança da LLM,
  - trust da fonte,
  - coerência global.
- Escalonamento humano para conflitos críticos.
- Auditoria via endpoint dedicado.

### 3) Memória
- Curadoria com clusters semânticos simples (baixo custo).
- Memória destilada (`modality=distilled`).
- Esquecimento ativo de baixa utilidade (`archived_at`, `utility_score`).
- Prune goal-aware (preserva conteúdo relacionado ao objetivo ativo).

### 4) Procedural Learning (não-declarativo)
- Tabelas `procedures` e `procedure_runs`.
- Aprendizado de procedimentos por observação (`/api/procedures/learn`).
- Seleção contextual de procedimento (`/api/procedures/select`).
- Execução simulada e execução ativa com artefatos locais.
- Métricas de habilidade por procedimento: tentativas, sucesso, score médio.

### 5) Transferência por Analogia (cross-domain)
- Módulo dedicado `analogy.py`.
- Geração de hipótese analógica + validação + aplicação.
- Persistência em `analogies` com status (`hypothesis/accepted/rejected`).
- Integração no planner/autonomia para conflitos persistentes.

### 6) Global Workspace + Metacognição
- Workspace global (`global_workspace`) para broadcast entre módulos.
- Publicação/consumo por canais de saliência (`metacog.snapshot`, `conflict.status`, `analogy.transfer`, `procedure.execution`, `self.state`).
- `decision_quality`, `stuck_cycles`, `replans`, `quality_history`.
- Anti-loop por baixa qualidade contínua + replanejamento automático.

### 7) Self-awareness funcional (limite técnico atual)
- Endpoint `/api/self-awareness/status` com auto-relato operacional.
- Proxy fenomenológico computacional (valence/arousal/control).
- **Nota importante:** isso não prova qualia fenomenológica real; é autoconsciência funcional baseada em self-model.

### 8) Executor externo seguro (Etapa E)
- Allowlist de ações externas.
- Dry-run.
- `reason` obrigatório para execução real.
- Two-phase commit (`prepare` -> `confirm_token` -> `execute`).
- Auditoria com hash encadeado.

### 9) Benchmark + Replay (Etapa F)
- Benchmark com cenários fixos.
- Histórico de benchmark (trend).
- Replay orientado por severidade de falhas.

---

## Rodando local (dev)
```bash
cd backend
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn ultronpro.main:app --reload --host 0.0.0.0 --port 8000
```

UI: http://localhost:8000/

---

## Deploy (Docker Swarm + Traefik)
Arquivos em `deploy/`.

Dados persistidos em `/app/data` (SQLite, settings, benchmark history etc.).

---

## Endpoints-chave (resumo)

### Status
- `GET /api/status`
- `GET /api/agi-mode`

### Autonomia / Metacognição / Self-model
- `GET /api/autonomy/status`
- `POST /api/autonomy/tick`
- `GET /api/metacognition/status`
- `GET /api/self-awareness/status`
- `POST /api/workspace/publish`
- `GET /api/workspace/read`

### Memória
- `GET /api/memory/status`
- `GET /api/memory/curation/status`
- `POST /api/memory/curation/run`
- `POST /api/memory/prune/run`

### Conflitos
- `GET /api/conflicts`
- `GET /api/conflicts/{id}`
- `GET /api/conflicts-prioritized`
- `POST /api/conflicts/synthesis/run`
- `POST /api/conflicts/auto-resolve`
- `GET /api/conflicts-audit`

### Objetivos
- `GET /api/goals`
- `POST /api/goals/refresh`
- `POST /api/goals/{id}/activate`
- `POST /api/goals/{id}/done`

### Curiosidade Adaptativa
- `POST /api/curiosity/refresh`
- `GET /api/curiosity/stats`
- `GET /api/curiosity/queue`

### Procedural Learning
- `GET /api/procedures`
- `POST /api/procedures/learn`
- `POST /api/procedures/select`
- `POST /api/procedures/execute`
- `POST /api/procedures/execute-active`
- `POST /api/procedures/run-log`

### Transferência por Analogia
- `POST /api/analogy/transfer`
- `GET /api/analogies`

### Executor Externo Seguro
- `POST /api/actions/prepare`
- `POST /api/actions/execute`

### Benchmark/Replay
- `POST /api/agi/benchmark/run`
- `GET /api/agi/benchmark/status`
- `GET /api/agi/benchmark/trend`
- `POST /api/learning/replay/run`

---

## Observações
- O sistema foi otimizado para evoluir sem estourar recursos da VPS.
- Federated learning permanece fora do escopo atual (fase posterior).