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

### 1) Conhecimento + Curiosidade
- Ingestão de texto/arquivo.
- Extração de triplas via LLM + fallback regex.
- Geração adaptativa de perguntas de curiosidade.

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

### 4) Autonomia segura (Etapa A)
- Fila de ações com prioridade.
- TTL/expiração de ações.
- Cooldown inteligente por tipo/chave.
- Budget por minuto.
- Circuit breaker em falhas consecutivas.

### 5) Metacognição (Etapa D)
- `decision_quality`, `stuck_cycles`, `replans`, `quality_history`.
- Anti-loop por baixa qualidade contínua.
- Replanejamento automático quando há atividade sem progresso real.

### 6) Executor externo seguro (Etapa E)
- Allowlist de ações externas.
- Dry-run.
- `reason` obrigatório para execução real.
- Two-phase commit (`prepare` -> `confirm_token` -> `execute`).
- Auditoria com hash encadeado.

### 7) Benchmark + Replay (Etapa F)
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

### Autonomia / Metacognição
- `GET /api/autonomy/status`
- `POST /api/autonomy/tick`
- `GET /api/metacognition/status`

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