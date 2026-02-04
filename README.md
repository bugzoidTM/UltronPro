# UltronPRO

MVP baseado em `/root/.openclaw/agi.md`.

**Ideia central (Fase 1 – “bebê”):**
- coletar experiências (inputs humanos)
- gerar *perguntas de curiosidade* (reduzir incerteza)
- aceitar respostas
- organizar conhecimento mínimo em:
  - **Grafo** (triplas sujeito–relação–objeto + evidências)
  - **Memória episódica** (experiências)

## Rodando (dev)
```bash
cd backend
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn ultronpro.main:app --reload --host 0.0.0.0 --port 8000
```

Abra: http://localhost:8000/

## Deploy (Swarm + Traefik)
Arquivos em `deploy/`.

Dados persistem em `/app/data`.
