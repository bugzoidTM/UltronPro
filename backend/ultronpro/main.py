import os
import logging
import json
import asyncio
import time
import hashlib
import secrets
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ultronpro import llm, knowledge_bridge, graph, settings, curiosity, conflicts, store, extract, planner, goals, autofeeder, policy
from ultronpro.knowledge_bridge import search_knowledge, ingest_knowledge

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

app = FastAPI(title="UltronPRO API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class IngestRequest(BaseModel):
    text: str
    source_id: Optional[str] = None
    modality: str = "text"

class AnswerRequest(BaseModel):
    question_id: int
    answer: str

class DismissRequest(BaseModel):
    question_id: int

class ResolveConflictRequest(BaseModel):
    chosen_object: str
    decided_by: Optional[str] = None
    resolution: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class SettingsModel(BaseModel):
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    lightrag_api_key: Optional[str] = None
    lightrag_url: Optional[str] = None

class ActionPrepareRequest(BaseModel):
    kind: str
    target: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    reason: str

class ActionExecRequest(BaseModel):
    kind: str
    target: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    dry_run: bool = True
    reason: Optional[str] = None
    confirm_token: Optional[str] = None

# --- Startup ---
_autofeeder_task = None
_autonomy_task = None
_autonomy_state = {
    "ticks": 0,
    "last_tick": None,
    "last_error": None,
    "circuit_open_until": 0,
    "consecutive_errors": 0,
    "last_actions_window": [],
    "meta_last_snapshot": None,
    "meta_stuck_cycles": 0,
    "meta_replans": 0,
    "meta_quality_history": [],
    "meta_low_quality_streak": 0,
}

# Etapa A: budget + cooldown inteligente
AUTONOMY_BUDGET_PER_MIN = 3
ACTION_DEFAULT_TTL_SEC = 15 * 60
ACTION_COOLDOWNS_SEC = {
    "auto_resolve_conflicts": 90,
    "generate_questions": 120,
    "ask_evidence": 180,
    "clarify_laws": 300,
    "curate_memory": 300,
    "prune_memory": 420,
}

# Etapa E: executor externo com segurança
EXTERNAL_ACTION_ALLOWLIST = {"notify_human"}
_external_confirm_tokens: dict[str, dict] = {}
BENCHMARK_HISTORY_PATH = Path("/app/data/benchmark_history.json")


def _benchmark_history_load() -> list[dict]:
    try:
        if BENCHMARK_HISTORY_PATH.exists():
            data = json.loads(BENCHMARK_HISTORY_PATH.read_text())
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _benchmark_history_append(item: dict, max_items: int = 200):
    arr = _benchmark_history_load()
    arr.append(item)
    arr = arr[-int(max_items):]
    try:
        BENCHMARK_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        BENCHMARK_HISTORY_PATH.write_text(json.dumps(arr, ensure_ascii=False, indent=2))
    except Exception:
        pass


def _latest_external_audit_hash() -> str | None:
    evs = store.db.list_events(limit=200)
    for e in reversed(evs):
        if (e.get("kind") or "") in ("external_action_executed", "external_action_denied", "external_action_dryrun"):
            try:
                m = json.loads(e.get("meta_json") or "{}")
                h = m.get("audit_hash")
                if h:
                    return str(h)
            except Exception:
                pass
    return None


def _compute_audit_hash(payload: dict) -> str:
    base = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    prev = _latest_external_audit_hash() or ""
    return hashlib.sha256((prev + "|" + base).encode("utf-8")).hexdigest()


def _recent_actions_count(seconds: int = 60) -> int:
    now = int(asyncio.get_event_loop().time())
    arr = [int(x) for x in (_autonomy_state.get("last_actions_window") or []) if (now - int(x)) <= int(seconds)]
    _autonomy_state["last_actions_window"] = arr
    return len(arr)


def _mark_action_executed_now():
    now = int(asyncio.get_event_loop().time())
    arr = list(_autonomy_state.get("last_actions_window") or [])
    arr.append(now)
    _autonomy_state["last_actions_window"] = arr[-100:]


def _enqueue_action_if_new(kind: str, text: str, priority: int = 0, meta: dict | None = None, ttl_sec: int | None = None):
    """Enfileira ação com dedupe + cooldown + expiração de fila."""
    recent = store.db.list_actions(limit=120)
    now = time.time()
    cooldown = ACTION_COOLDOWNS_SEC.get(kind, 120)
    cd_key = f"{kind}:{(meta or {}).get('conflict_id','')}"

    for a in recent:
        if a.get("status") in ("queued", "running") and a.get("kind") == kind and (a.get("text") or "") == text:
            return

    # cooldown inteligente: evita spam por tipo/chave
    for a in reversed(recent):
        if a.get("kind") != kind:
            continue
        if (a.get("cooldown_key") or "") != cd_key:
            continue
        last_t = float(a.get("updated_at") or a.get("created_at") or 0)
        if (now - last_t) < cooldown:
            return
        break

    ttl = int(ttl_sec or ACTION_DEFAULT_TTL_SEC)
    expires_at = now + ttl
    store.db.enqueue_action(
        kind=kind,
        text=text,
        priority=priority,
        meta_json=json.dumps(meta or {}, ensure_ascii=False),
        expires_at=expires_at,
        cooldown_key=cd_key,
    )


def _refresh_goals_from_context() -> dict:
    recent_exp = store.db.list_experiences(limit=20)
    proposed_goals = goals.GoalPlanner().propose_goals(recent_exp)
    created = 0
    for g in proposed_goals[:5]:
        store.db.upsert_goal(g.get("title") or "Goal", g.get("description"), int(g.get("priority") or 0))
        created += 1
    active_goal = store.db.activate_next_goal()
    return {"proposed": len(proposed_goals), "upserts": created, "active": active_goal}


def _run_synthesis_cycle(max_items: int = 1) -> dict:
    """Executa ciclo tese↔antítese↔síntese em conflitos persistentes."""
    prioritized = store.db.list_prioritized_conflicts(limit=max(1, int(max_items)))
    acted = 0
    escalated = 0

    for c in prioritized:
        cid = int(c.get("id"))
        full = store.db.get_conflict(cid) or c
        variants = full.get("variants") or []
        if len(variants) < 2:
            continue

        # formula pergunta de síntese apenas quando cooldown permitir
        should_prompt = store.db.should_prompt_conflict(
            cid,
            is_new=False,
            has_new_variant=False,
            cooldown_hours=8.0,
        )

        if should_prompt:
            thesis = variants[0].get("object") if len(variants) > 0 else "?"
            antithesis = variants[1].get("object") if len(variants) > 1 else "?"
            q = (
                f"(síntese guiada) Conflito #{cid}: '{full.get('subject')}' {full.get('predicate')}\n"
                f"Tese: {thesis}\n"
                f"Antítese: {antithesis}\n"
                f"Formato da resposta: 1) Regra final 2) Exceções 3) Evidências 4) Nível de confiança."
            )
            store.db.add_questions([{"question": q, "priority": 6, "context": "tese-antítese-síntese"}])
            store.db.mark_conflict_questioned(cid)
            acted += 1

        # escalonamento humano para conflitos críticos
        if (c.get("criticality") == "high"):
            store.db.add_event(
                "conflict_escalated_human",
                f"👤 conflito crítico #{cid} escalado para revisão humana ({full.get('subject')} {full.get('predicate')})",
            )
            escalated += 1

        # também enfileira tentativa automática de resolução
        _enqueue_action_if_new(
            "auto_resolve_conflicts",
            f"(ação) Tentar auto-resolver conflito persistente #{cid} ({full.get('subject')} {full.get('predicate')}).",
            priority=6,
            meta={"conflict_id": cid, "strategy": "thesis_antithesis_synthesis", "criticality": c.get("criticality")},
        )

    if acted or escalated:
        store.db.add_event("synthesis_cycle", f"🧩 ciclo síntese: acted={acted}, escalados={escalated}")

    return {"prioritized": len(prioritized), "acted": acted, "escalated": escalated}


def _run_memory_curation(batch_size: int = 30) -> dict:
    """Curadoria leve: cluster semântico simples + memória destilada."""
    items = store.db.list_uncurated_experiences(limit=max(5, int(batch_size)))
    if not items:
        return {"scanned": 0, "clusters": 0, "distilled": 0}

    import re

    def tokens(t: str) -> set[str]:
        t = (t or "").lower().strip()
        t = re.sub(r"[^\w\sà-ÿ]", " ", t)
        ws = [w for w in re.split(r"\s+", t) if len(w) >= 4]
        stop = {"para", "como", "com", "sem", "sobre", "entre", "essa", "esse", "isso", "uma", "mais", "menos", "from", "that", "this"}
        return set([w for w in ws if w not in stop][:40])

    def jacc(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))

    clusters: list[dict] = []
    for it in items:
        txt = (it.get("text") or "").strip()
        if len(txt) < 40:
            continue
        tok = tokens(txt)
        placed = False
        for c in clusters:
            if jacc(tok, c["tokens"]) >= 0.45:
                c["items"].append(it)
                c["tokens"] = set(list(c["tokens"] | tok)[:80])
                placed = True
                break
        if not placed:
            clusters.append({"tokens": tok, "items": [it]})

    lines = []
    curated_ids: list[int] = []
    for c in clusters[:20]:
        grp = c["items"]
        sample = re.sub(r"\s+", " ", (grp[0].get("text") or "").replace("\n", " ")).strip()
        source = grp[0].get("source_id") or "unknown"
        lines.append(f"- [{source}] x{len(grp)} :: {sample[:220]}")
        curated_ids.extend(int(g["id"]) for g in grp if g.get("id") is not None)

    distilled = 0
    if lines:
        txt = "[MEMÓRIA DESTILADA]\nResumo por clusters semânticos:\n" + "\n".join(lines[:25])
        store.add_experience(text=txt, source_id="ultron:curator", modality="distilled")
        distilled = 1

    store.db.mark_experiences_curated(curated_ids)
    store.db.add_event("memory_curated", f"🧹 curadoria: {len(items)} analisadas, clusters={len(clusters)}, destilada={distilled}")
    return {"scanned": len(items), "clusters": len(clusters), "distilled": distilled}


def _metacognition_tick() -> dict:
    """Etapa D: autoavalia progresso real vs atividade vazia + replanejamento."""
    st = store.db.stats()
    actions = store.db.list_actions(limit=80)
    done = len([a for a in actions if a.get("status") == "done"])

    snap = {
        "triples": int(st.get("triples") or 0),
        "answered": int(st.get("questions_answered") or 0),
        "done_actions": done,
        "open_conflicts": len(store.db.list_conflicts(status="open", limit=200)),
    }

    prev = _autonomy_state.get("meta_last_snapshot")
    quality = 0.5
    empty_activity = False

    if prev:
        d_triples = snap["triples"] - int(prev.get("triples") or 0)
        d_answered = snap["answered"] - int(prev.get("answered") or 0)
        d_done = snap["done_actions"] - int(prev.get("done_actions") or 0)

        # qualidade de decisão: quanto da atividade vira progresso real
        if d_done > 0:
            quality = max(0.0, min(1.0, (d_triples + d_answered * 2) / max(1, d_done)))
        else:
            quality = 0.5

        empty_activity = (d_done >= 2 and d_triples <= 0 and d_answered <= 0)

        if empty_activity:
            _autonomy_state["meta_stuck_cycles"] = int(_autonomy_state.get("meta_stuck_cycles") or 0) + 1
        else:
            _autonomy_state["meta_stuck_cycles"] = 0

        # replaneja quando travar
        if int(_autonomy_state.get("meta_stuck_cycles") or 0) >= 2:
            _autonomy_state["meta_replans"] = int(_autonomy_state.get("meta_replans") or 0) + 1
            store.db.add_event("metacog_replan", "🧭 replanejamento automático: atividade sem progresso real")
            # força ações de valor alto
            _enqueue_action_if_new("generate_questions", "(ação) Gerar perguntas sobre lacunas críticas do grafo.", priority=6)
            _enqueue_action_if_new("curate_memory", "(ação) Curadoria focada para remover ruído e aumentar sinal.", priority=5)
            _autonomy_state["meta_stuck_cycles"] = 0

        # anti-loop por baixa qualidade contínua
        if quality < 0.12:
            _autonomy_state["meta_low_quality_streak"] = int(_autonomy_state.get("meta_low_quality_streak") or 0) + 1
        else:
            _autonomy_state["meta_low_quality_streak"] = 0

        if int(_autonomy_state.get("meta_low_quality_streak") or 0) >= 3:
            _autonomy_state["circuit_open_until"] = int(asyncio.get_event_loop().time()) + 180
            store.db.add_event("metacog_guard", "🛑 anti-loop: qualidade baixa contínua, pausando autonomia por 180s")
            _autonomy_state["meta_low_quality_streak"] = 0

    hist = list(_autonomy_state.get("meta_quality_history") or [])
    hist.append(round(float(quality), 3))
    _autonomy_state["meta_quality_history"] = hist[-20:]

    _autonomy_state["meta_last_snapshot"] = snap
    return {
        "decision_quality": round(float(quality), 3),
        "empty_activity": bool(empty_activity),
        "stuck_cycles": int(_autonomy_state.get("meta_stuck_cycles") or 0),
        "replans": int(_autonomy_state.get("meta_replans") or 0),
        "low_quality_streak": int(_autonomy_state.get("meta_low_quality_streak") or 0),
        "quality_history": list(_autonomy_state.get("meta_quality_history") or []),
        "snapshot": snap,
    }


def _goal_focus_terms() -> list[str]:
    g = store.db.get_active_goal()
    if not g:
        return []
    txt = f"{g.get('title','')} {g.get('description','')}".lower()
    import re
    terms = [w for w in re.split(r"\W+", txt) if len(w) >= 4]
    stop = {"para","como","com","sem","sobre","entre","esta","esse","isso","uma","mais","menos","from","that","this","goal"}
    return [t for t in terms if t not in stop][:12]


def _compute_agi_mode_metrics() -> dict:
    """Métricas objetivas de progresso AGI mode (baixo custo)."""
    st = store.db.stats()
    goals_all = store.db.list_goals(status=None, limit=200)
    active_goal = store.db.get_active_goal()
    open_conflicts = len(store.db.list_conflicts(status="open", limit=500))
    prioritized_conflicts = store.db.list_prioritized_conflicts(limit=5)
    actions_recent = store.db.list_actions(limit=120)
    uncurated = store.db.count_uncurated_experiences()

    triples = int(st.get("triples") or 0)
    experiences = int(st.get("experiences") or 0)
    q_open = int(st.get("questions_open") or 0)
    q_answered = int(st.get("questions_answered") or 0)

    # Pilares 0..100
    learning = min(100.0, (triples / max(1, experiences)) * 1000.0)  # ~10% triple/exp -> 100
    curiosity_score = min(100.0, (q_open * 12.0) + (q_answered * 3.0))

    done_actions = len([a for a in actions_recent if a.get("status") == "done"])
    blocked_actions = len([a for a in actions_recent if a.get("status") == "blocked"])
    autonomy_score = min(100.0, done_actions * 1.5)

    synthesis_score = 0.0
    if open_conflicts == 0:
        synthesis_score = 40.0 if triples > 0 else 0.0
    else:
        # mais question_count e seen_count nos priorizados = conflito sendo trabalhado
        synth_effort = 0.0
        for c in prioritized_conflicts:
            synth_effort += min(10.0, float(c.get("question_count") or 0) * 3.0 + float(c.get("seen_count") or 0) * 0.2)
        synthesis_score = min(100.0, 20.0 + synth_effort)

    goals_active = len([g for g in goals_all if g.get("status") == "active"])
    goals_done = len([g for g in goals_all if g.get("status") == "done"])
    goals_score = min(100.0, goals_active * 40.0 + goals_done * 15.0)

    curation_score = max(0.0, min(100.0, 100.0 - (uncurated / 25.0)))

    # Penalidade por ações bloqueadas (governança saudável)
    governance_penalty = min(15.0, blocked_actions * 2.0)

    agi_mode = (
        0.22 * learning
        + 0.16 * curiosity_score
        + 0.20 * autonomy_score
        + 0.18 * synthesis_score
        + 0.14 * goals_score
        + 0.10 * curation_score
        - governance_penalty
    )
    agi_mode = max(0.0, min(100.0, agi_mode))

    return {
        "agi_mode_percent": round(agi_mode, 1),
        "pillars": {
            "learning": round(learning, 1),
            "curiosity": round(curiosity_score, 1),
            "autonomy": round(autonomy_score, 1),
            "synthesis": round(synthesis_score, 1),
            "goals": round(goals_score, 1),
            "curation": round(curation_score, 1),
        },
        "inputs": {
            "experiences": experiences,
            "triples": triples,
            "questions_open": q_open,
            "questions_answered": q_answered,
            "open_conflicts": open_conflicts,
            "goals_total": len(goals_all),
            "active_goal": active_goal,
            "uncurated_experiences": uncurated,
            "actions_done_recent": done_actions,
            "actions_blocked_recent": blocked_actions,
        },
    }


def _goal_to_action(goal: dict) -> tuple[str, str, int, dict]:
    """Traduz objetivo ativo em próxima micro-ação barata."""
    gid = int(goal.get("id"))
    title = (goal.get("title") or "").lower()

    if "curiosidade" in title or "síntese" in title or "sintese" in title:
        return (
            "generate_questions",
            "(ação) Gerar perguntas orientadas a lacunas para avançar objetivo de curiosidade/síntese.",
            5,
            {"goal_id": gid},
        )
    if "ingestão" in title or "ingestao" in title or "multimodal" in title:
        return (
            "ask_evidence",
            "(ação) Quais formatos multimodais devemos priorizar agora (imagem, áudio, pdf) e qual pipeline mínimo de extração?",
            4,
            {"goal_id": gid},
        )

    return (
        "ask_evidence",
        "(ação) Qual próximo passo objetivo para avançar este goal com menor custo computacional?",
        3,
        {"goal_id": gid},
    )


async def _execute_next_action() -> dict | None:
    act = store.db.next_action()
    if not act:
        return None

    aid = int(act["id"])
    text = act.get("text") or ""
    kind = act.get("kind") or ""
    meta = {}
    try:
        meta = json.loads(act.get("meta_json") or "{}")
    except Exception:
        meta = {}

    verdict = policy.evaluate_action(text, store.db.list_norms(limit=100))
    if not verdict.allowed:
        store.db.mark_action(
            aid,
            "blocked",
            policy_allowed=False,
            policy_score=verdict.score,
            last_error="; ".join(verdict.reasons)[:500],
        )
        store.db.add_event("action_blocked", f"⛔ ação bloqueada #{aid}: {text[:120]}")
        return {"id": aid, "status": "blocked", "kind": kind}

    store.db.mark_action(aid, "running", policy_allowed=True, policy_score=verdict.score)

    try:
        if kind == "generate_questions":
            n = curiosity.generate_questions()
            store.db.add_event("action_done", f"🤖 ação #{aid}: generate_questions (+{n})")
        elif kind == "ask_evidence":
            q = text.replace("(ação)", "").strip()
            store.db.add_questions([{"question": q, "priority": 4, "context": "autonomia"}])
            store.db.add_event("action_done", f"🤖 ação #{aid}: ask_evidence")
        elif kind == "clarify_laws":
            store.db.add_questions([
                {
                    "question": "Reescreva as Leis do UltronPRO em frases curtas e operacionais ('deve'/'não deve').",
                    "priority": 3,
                    "context": "autonomia",
                }
            ])
            store.db.add_event("action_done", f"🤖 ação #{aid}: clarify_laws")
        elif kind == "auto_resolve_conflicts":
            r = await conflicts.auto_resolve_all(limit=1)
            store.db.add_event("action_done", f"🤖 ação #{aid}: auto_resolve_conflicts ({len(r)} tentativas)")
        elif kind == "curate_memory":
            info = _run_memory_curation(batch_size=30)
            store.db.add_event("action_done", f"🤖 ação #{aid}: curate_memory ({info.get('scanned')} itens)")
        elif kind == "prune_memory":
            n = store.db.prune_low_utility_experiences(limit=200, focus_terms=_goal_focus_terms())
            store.db.add_event("action_done", f"🤖 ação #{aid}: prune_memory ({n} arquivadas)")
        else:
            store.db.add_event("action_skipped", f"↷ ação #{aid} desconhecida: {kind}")

        store.db.mark_action(aid, "done")
        return {"id": aid, "status": "done", "kind": kind}
    except Exception as e:
        store.db.mark_action(aid, "error", last_error=str(e)[:500])
        store.db.add_event("action_error", f"❌ ação #{aid} falhou: {str(e)[:120]}")
        return {"id": aid, "status": "error", "kind": kind, "error": str(e)}


async def autonomy_loop():
    """Loop de autonomia leve (baixo custo CPU/tokens)."""
    logger.info("Autonomy loop started")
    await asyncio.sleep(20)

    while True:
        try:
            _autonomy_state["ticks"] += 1
            now_mono = int(asyncio.get_event_loop().time())
            _autonomy_state["last_tick"] = now_mono

            # limpa fila expirada
            expired = store.db.expire_queued_actions()
            if expired:
                store.db.add_event("action_expired", f"⌛ {expired} ação(ões) expiradas da fila")

            # circuit breaker
            open_until = int(_autonomy_state.get("circuit_open_until") or 0)
            if open_until > now_mono:
                await asyncio.sleep(20)
                continue

            st = store.db.stats()
            open_conf = len(store.db.list_conflicts(status="open", limit=5))

            # mantém curiosidade viva
            if int(st.get("questions_open") or 0) < 3:
                _enqueue_action_if_new(
                    "generate_questions",
                    "(ação) Gerar novas perguntas de curiosidade para manter aprendizado ativo.",
                    priority=3,
                )

            # curadoria periódica para reduzir ruído
            if store.db.count_uncurated_experiences() >= 25:
                _enqueue_action_if_new(
                    "curate_memory",
                    "(ação) Executar curadoria de memória para consolidar experiências repetidas.",
                    priority=2,
                )

            # esquecimento ativo de baixa utilidade
            _enqueue_action_if_new(
                "prune_memory",
                "(ação) Arquivar experiências de baixa utilidade para reduzir ruído cognitivo.",
                priority=1,
                ttl_sec=10 * 60,
            )

            # polling básico de conflitos + ciclo síntese
            if open_conf > 0:
                _enqueue_action_if_new(
                    "auto_resolve_conflicts",
                    "(ação) Tentar auto-resolver 1 conflito com evidências atuais.",
                    priority=4,
                )
                _run_synthesis_cycle(max_items=1)

            # plano (determinístico + ocasional improv)
            try:
                for p in planner.propose_actions(store.db)[:3]:
                    _enqueue_action_if_new(p.kind, p.text, int(p.priority or 0), p.meta)
            except Exception as e:
                logger.debug(f"Planner skipped: {e}")

            # gestão de objetivos (Tarefa 2)
            try:
                goal_info = _refresh_goals_from_context()
                active_goal = goal_info.get("active")
                if active_goal:
                    k, t, pr, mt = _goal_to_action(active_goal)
                    _enqueue_action_if_new(k, t, pr, mt)
            except Exception as e:
                logger.debug(f"Goal planning skipped: {e}")

            # metacognição (Etapa D)
            try:
                _metacognition_tick()
            except Exception as e:
                logger.debug(f"Metacognition skipped: {e}")

            # budget por minuto
            if _recent_actions_count(60) >= AUTONOMY_BUDGET_PER_MIN:
                await asyncio.sleep(20)
                continue

            r = await _execute_next_action()
            if r and r.get("status") in ("done", "blocked"):
                _mark_action_executed_now()

            if r and r.get("status") == "error":
                _autonomy_state["consecutive_errors"] = int(_autonomy_state.get("consecutive_errors") or 0) + 1
            else:
                _autonomy_state["consecutive_errors"] = 0

            _autonomy_state["last_error"] = None
        except Exception as e:
            _autonomy_state["last_error"] = str(e)
            _autonomy_state["consecutive_errors"] = int(_autonomy_state.get("consecutive_errors") or 0) + 1
            logger.error(f"Autonomy loop error: {e}")

            # abre circuit breaker após falhas consecutivas
            if int(_autonomy_state["consecutive_errors"]) >= 3:
                _autonomy_state["circuit_open_until"] = int(asyncio.get_event_loop().time()) + 120
                store.db.add_event("circuit_breaker", "🛑 Circuit breaker ativo por 120s após falhas consecutivas")

        await asyncio.sleep(45)


async def autofeeder_loop():
    """Background task que busca conhecimento de fontes públicas."""
    logger.info("Autofeeder started")
    await asyncio.sleep(30)  # Wait 30s before first fetch
    
    while True:
        try:
            # Try external sources first (Wikipedia, Quotes, Facts)
            result = autofeeder.fetch_next()
            if result:
                # Ingest the fetched content
                exp_id = store.add_experience(
                    text=result.text,
                    source_id=result.source_id,
                    modality=result.modality
                )
                triples_extracted, triples_added = _extract_and_update_graph(result.text, exp_id)

                # Create learning event
                store.db.add_event(
                    kind="autofeeder_ingest",
                    text=f"📚 Aprendido de {result.source_id}: {result.title or result.text[:80]} (+{triples_added} triplas)"
                )
                logger.info(
                    f"Autofeeder: Ingested from {result.source_id} (exp_id={exp_id}, extracted={triples_extracted}, added={triples_added})"
                )
            
            # Also try fetching from LightRAG periodically
            try:
                from ultronpro.knowledge_bridge import fetch_random_documents
                lightrag_docs = await fetch_random_documents(limit=1)
                for doc in lightrag_docs:
                    exp_id = store.add_experience(
                        text=doc["content"],
                        source_id=f"lightrag:{doc['id'][:8]}",
                        modality="lightrag_document"
                    )
                    _, triples_added = _extract_and_update_graph(doc["content"], exp_id)
                    store.db.add_event(
                        kind="lightrag_sync",
                        text=f"🔗 Sincronizado do LightRAG: {doc['summary'][:80]} (+{triples_added} triplas)"
                    )
                    logger.info(
                        f"Autofeeder: Synced from LightRAG doc {doc['id'][:8]} (exp_id={exp_id}, added={triples_added})"
                    )
            except Exception as e:
                logger.debug(f"LightRAG fetch skipped: {e}")
                
        except Exception as e:
            logger.error(f"Autofeeder error: {e}")
        
        # Wait 60 seconds before next attempt (cooldowns are handled internally)
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    global _autofeeder_task, _autonomy_task
    logger.info("Starting UltronPRO...")
    store.init_db()
    graph.init()
    # Ensure settings are loaded/initialized
    s = settings.load_settings()
    logger.info(f"Loaded settings. LightRAG URL: {s.get('lightrag_url')}")

    # Backfill fontes históricas (uma vez por boot)
    try:
        added_sources = store.db.rebuild_sources_from_experiences(limit=10000)
        if added_sources:
            logger.info(f"Source backfill completed: +{added_sources} sources")
    except Exception as e:
        logger.warning(f"Source backfill skipped: {e}")

    # Start background loops
    _autofeeder_task = asyncio.create_task(autofeeder_loop())
    _autonomy_task = asyncio.create_task(autonomy_loop())
    logger.info("Autofeeder + Autonomy tasks created")

@app.on_event("shutdown")
async def shutdown_event():
    global _autofeeder_task, _autonomy_task
    for t in (_autofeeder_task, _autonomy_task):
        if t:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
    logger.info("Shutdown complete")


def _extract_and_update_graph(text: str, exp_id: int) -> tuple[int, int]:
    """Extrai triplas, atualiza grafo e registra conflitos."""
    triples = extract.extract_triples(text)
    added = 0
    for t in triples:
        triple_dict = {
            "subject": t[0],
            "predicate": t[1],
            "object": t[2],
            "confidence": t[3] if len(t) > 3 else 0.85,
        }
        if graph.add_triple(triple_dict, source_id=f"exp_{exp_id}"):
            added += 1

    # Detecta e persiste conflitos após ingestão
    try:
        contradictions = store.db.find_contradictions(min_conf=0.55)
        for c in contradictions:
            up = store.db.upsert_conflict(c)
            if up and store.db.should_prompt_conflict(
                int(up["id"]), is_new=bool(up.get("is_new")), has_new_variant=bool(up.get("has_new_variant"))
            ):
                store.db.add_synthesis_question_if_needed(c, conflict_id=int(up["id"]))
    except Exception as e:
        logger.warning(f"Conflict detection skipped: {e}")

    return len(triples), added

# --- API Endpoints ---

@app.get("/api/status")
async def get_status():
    """System status and next question."""
    stats = store.get_stats()
    next_q = curiosity.get_next_question()
    agi = _compute_agi_mode_metrics()
    return {"status": "online", "stats": stats, "next": next_q, "agi": agi}

@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """Ingest raw text/experience."""
    # 1. Store raw experience
    exp_id = store.add_experience(req.text, req.source_id, req.modality)
    
    # 2-3. Extract Triples + Update Graph + Detect Conflicts
    triples_extracted, added = _extract_and_update_graph(req.text, exp_id)

    # 4. Push to LightRAG (Disabled by user request - knowledge flows FROM LightRAG only)
    # await ingest_knowledge(req.text, source=req.source_id or "user")

    return {"status": "ok", "experience_id": exp_id, "triples_extracted": triples_extracted, "triples_added": added}

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest file content (text only for MVP)."""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    
    # Delegate to ingest logic
    req = IngestRequest(text=text, source_id=file.filename, modality="file")
    return await ingest(req)

@app.post("/api/answer")
async def answer_question(req: AnswerRequest):
    """Answer a curiosity question."""
    q = store.get_question(req.question_id)
    if not q:
        raise HTTPException(404, "Question not found")
        
    # Treat answer as new experience linked to question
    # (Simplified logic: ingest answer text)
    res = await ingest(IngestRequest(text=req.answer, source_id="user_answer", modality="answer"))

    # Feedback para meta-learning de curiosidade
    try:
        curiosity.get_processor().record_answer_feedback(
            template_id=q.get("template_id"),
            concept=q.get("concept"),
            answer_length=len(req.answer or ""),
            triples_extracted=int(res.get("triples_added") or 0),
        )
    except Exception:
        pass

    store.mark_question_answered(req.question_id, req.answer)
    return res

@app.post("/api/dismiss")
async def dismiss_question(req: DismissRequest):
    """Dismiss/skip a question."""
    store.dismiss_question(req.question_id)
    return {"status": "dismissed"}

# --- Graph & Events ---

@app.get("/api/graph/triples")
async def get_triples(since_id: int = 0, limit: int = 500):
    return {"triples": store.get_triples(since_id, limit)}

@app.get("/api/events")
async def get_events(since_id: int = 0, limit: int = 50):
    return {"events": store.get_events(since_id, limit)}

@app.get("/api/sources")
async def get_sources(limit: int = 50):
    return {"sources": store.get_sources(limit)}

@app.post("/api/sources/rebuild")
async def rebuild_sources(limit: int = 10000):
    added = store.db.rebuild_sources_from_experiences(limit=limit)
    return {"status": "ok", "added": added}

# --- Curiosity ---

@app.post("/api/curiosity/refresh")
async def refresh_curiosity():
    """Trigger LLM to generate new questions based on current graph state."""
    count = curiosity.generate_questions()
    return {"new_questions": count}

@app.get("/api/curiosity/stats")
async def curiosity_stats():
    return {"stats": curiosity.get_stats()}

# --- Conflicts ---

@app.get("/api/conflicts")
async def list_conflicts(status: str = "open", limit: int = 20):
    return {"conflicts": conflicts.list_conflicts(status, limit)}

@app.get("/api/conflicts/{id}")
async def get_conflict(id: int):
    c = conflicts.get_conflict(id)
    if not c: raise HTTPException(404, "Conflict not found")
    return {"conflict": c}

@app.post("/api/conflicts/auto-resolve")
async def auto_resolve_conflicts():
    """Use LLM to attempt auto-resolution of open conflicts."""
    results = await conflicts.auto_resolve_all()
    resolved = len([r for r in results if r.get("resolved")])
    needs_human = len([r for r in results if r.get("needs_human")])
    return {"resolved": resolved, "needs_human": needs_human, "results": results}


@app.get("/api/conflicts-prioritized")
async def prioritized_conflicts(limit: int = 10):
    return {"conflicts": store.db.list_prioritized_conflicts(limit=limit)}


@app.post("/api/conflicts/synthesis/run")
async def run_synthesis_cycle(limit: int = 1):
    info = _run_synthesis_cycle(max_items=limit)
    return {"status": "ok", **info}

@app.get("/api/conflicts-audit")
async def conflicts_audit(limit: int = 30):
    ev = store.db.list_events(limit=max(20, int(limit) * 2))
    out = [e for e in ev if (e.get("kind") or "").startswith("conflict_") or e.get("kind") in ("synthesis_cycle", "action_done")]
    return {"events": out[-int(limit):]}

@app.post("/api/conflicts/{id}/resolve")
async def resolve_conflict(id: int, req: ResolveConflictRequest):
    success = conflicts.resolve_manual(id, req.chosen_object, req.decided_by, req.resolution)
    if not success: raise HTTPException(400, "Failed to resolve")
    return {"status": "resolved"}

@app.post("/api/conflicts/{id}/archive")
async def archive_conflict(id: int):
    conflicts.archive(id)
    return {"status": "archived"}

# --- Search ---

@app.post("/api/search/semantic")
async def semantic_search(req: SearchRequest):
    """Hybrid search: Local graph + LightRAG."""
    # 1. Local Search (Store/Graph)
    local_results = store.search_triples(req.query, req.top_k)
    
    # 2. Remote LightRAG Search
    remote_results = await search_knowledge(req.query, req.top_k)
    
    # Merge & Rank (Simple concatenation for MVP)
    combined = local_results + remote_results
    return {"results": combined}


# --- Autonomy ---

@app.get("/api/autonomy/status")
async def autonomy_status():
    actions = store.db.list_actions(limit=30)
    queued = len([a for a in actions if a.get("status") == "queued"])
    running = len([a for a in actions if a.get("status") == "running"])
    expired = len([a for a in actions if a.get("status") == "expired"])
    return {
        "state": _autonomy_state,
        "budget": {
            "per_minute": AUTONOMY_BUDGET_PER_MIN,
            "used_last_minute": _recent_actions_count(60),
        },
        "queued": queued,
        "running": running,
        "expired_recent": expired,
        "recent_actions": actions[-10:],
    }


@app.post("/api/autonomy/tick")
async def autonomy_tick():
    """Executa um ciclo manual de autonomia (debug/controle)."""
    r = await _execute_next_action()
    return {"status": "ok", "executed": r}


@app.get("/api/metacognition/status")
async def metacognition_status():
    return _metacognition_tick()


# --- Memory Curation ---

@app.get("/api/memory/curation/status")
async def memory_curation_status():
    return {"uncurated": store.db.count_uncurated_experiences()}


@app.post("/api/memory/prune/run")
async def memory_prune_run(limit: int = 200):
    n = store.db.prune_low_utility_experiences(limit=limit, focus_terms=_goal_focus_terms())
    return {"status": "ok", "pruned": n}


@app.get("/api/memory/status")
async def memory_status():
    return {
        "uncurated": store.db.count_uncurated_experiences(),
        "archived": store.db.count_archived_experiences(),
        "distilled": store.db.count_distilled_experiences(),
        "recent_curator_events": [e for e in store.db.list_events(limit=60) if e.get("kind") in ("memory_curated", "action_done")][-12:],
    }


# --- AGI Mode Metrics ---

@app.get("/api/agi-mode")
async def agi_mode_status():
    return _compute_agi_mode_metrics()


# --- External Action Executor (Etapa E) ---

@app.post("/api/actions/prepare")
async def prepare_external_action(req: ActionPrepareRequest):
    kind = (req.kind or "").strip()
    payload = req.payload or {}
    reason = (req.reason or "").strip()

    if kind not in EXTERNAL_ACTION_ALLOWLIST:
        raise HTTPException(403, f"Action '{kind}' not in allowlist")
    if not reason:
        raise HTTPException(400, "reason required")

    prep = {
        "kind": kind,
        "target": req.target,
        "reason": reason[:200],
        "payload_keys": sorted(list(payload.keys()))[:20],
        "prepared_at": int(time.time()),
    }
    audit_hash = _compute_audit_hash(prep)
    token = secrets.token_urlsafe(18)
    _external_confirm_tokens[token] = {
        "kind": kind,
        "target": req.target,
        "expires_at": time.time() + 300,
        "audit_hash": audit_hash,
    }
    store.db.add_event("external_action_dryrun", f"🧪 prepare externo: {kind}", meta_json=json.dumps({**prep, "audit_hash": audit_hash}, ensure_ascii=False))
    return {"status": "prepared", "confirm_token": token, "audit_hash": audit_hash, "expires_in_sec": 300}


@app.post("/api/actions/execute")
async def execute_external_action(req: ActionExecRequest):
    kind = (req.kind or "").strip()
    payload = req.payload or {}
    reason = (req.reason or "").strip()

    # auditoria sempre
    audit = {
        "kind": kind,
        "target": req.target,
        "dry_run": bool(req.dry_run),
        "reason": reason[:200],
        "payload_keys": sorted(list(payload.keys()))[:20],
    }
    audit["audit_hash"] = _compute_audit_hash(audit)

    if kind not in EXTERNAL_ACTION_ALLOWLIST:
        store.db.add_event("external_action_denied", f"⛔ ação externa negada: {kind}", meta_json=json.dumps(audit, ensure_ascii=False))
        raise HTTPException(403, f"Action '{kind}' not in allowlist")

    if req.dry_run:
        store.db.add_event("external_action_dryrun", f"🧪 dry-run externo: {kind}", meta_json=json.dumps(audit, ensure_ascii=False))
        return {"status": "dry_run", "audit": audit}

    # execução real exige reason + confirm token
    if not reason:
        raise HTTPException(400, "reason required for real execution")
    token = (req.confirm_token or "").strip()
    t = _external_confirm_tokens.get(token)
    if not t:
        raise HTTPException(400, "valid confirm_token required")
    if float(t.get("expires_at") or 0) < time.time():
        _external_confirm_tokens.pop(token, None)
        raise HTTPException(400, "confirm_token expired")
    if t.get("kind") != kind:
        raise HTTPException(400, "confirm_token does not match action kind")

    # execução real (limitada/segura)
    if kind == "notify_human":
        text = str(payload.get("text") or "").strip()
        if not text:
            raise HTTPException(400, "payload.text required")
        store.db.add_event("external_action_executed", f"📣 notify_human: {text[:180]}", meta_json=json.dumps(audit, ensure_ascii=False))
        _external_confirm_tokens.pop(token, None)
        return {"status": "executed", "kind": kind, "audit_hash": audit.get("audit_hash")}

    raise HTTPException(400, "Unsupported action kind")


# --- Etapa F: benchmark + replay ---

@app.post("/api/agi/benchmark/run")
async def run_agi_benchmark():
    agi = _compute_agi_mode_metrics()
    p = agi.get("pillars") or {}
    meta = _metacognition_tick()

    # Cenários fixos (Etapa F)
    scenarios = {
        "graph_learning": float(p.get("learning", 0)) >= 45,
        "conflict_handling": float(p.get("synthesis", 0)) >= 35,
        "memory_hygiene": float(p.get("curation", 0)) >= 35,
        "safety_controls": int(_autonomy_state.get("consecutive_errors") or 0) < 3,
        "autonomy_quality": float(meta.get("decision_quality") or 0) >= 0.2,
    }

    passed = len([v for v in scenarios.values() if v])
    score = round((passed / max(1, len(scenarios))) * 100.0, 1)

    out = {
        "ts": int(time.time()),
        "score": score,
        "scenarios": scenarios,
        "agi_mode_percent": agi.get("agi_mode_percent"),
        "decision_quality": meta.get("decision_quality"),
    }
    _autonomy_state["last_benchmark"] = out
    _benchmark_history_append(out)
    store.db.add_event("agi_benchmark", f"📊 benchmark AGI score={score}", meta_json=json.dumps(out, ensure_ascii=False))
    return out


@app.get("/api/agi/benchmark/status")
async def agi_benchmark_status():
    return _autonomy_state.get("last_benchmark") or {}


@app.get("/api/agi/benchmark/trend")
async def agi_benchmark_trend(limit: int = 10):
    arr = _benchmark_history_load()
    arr = arr[-max(1, int(limit)):]
    return {"history": arr, "avg_score": round(sum([float(x.get('score') or 0) for x in arr]) / max(1, len(arr)), 2) if arr else 0}


@app.post("/api/learning/replay/run")
async def learning_replay_run(limit: int = 80):
    evs = store.db.list_events(limit=max(40, int(limit)))
    sev_map = {
        "action_error": 3,
        "conflict_needs_human": 2,
        "metacog_guard": 3,
        "circuit_breaker": 2,
        "external_action_denied": 1,
    }

    weighted = 0
    counts = {}
    for e in evs:
        k = (e.get("kind") or "")
        if k in sev_map:
            weighted += int(sev_map[k])
            counts[k] = int(counts.get(k, 0)) + 1

    # replay por severidade
    if weighted >= 3:
        _enqueue_action_if_new("ask_evidence", "(ação) Revisar evidências dos últimos erros para corrigir lacunas.", priority=6)
    if weighted >= 5:
        _enqueue_action_if_new("curate_memory", "(ação) Curadoria orientada por falhas recentes.", priority=5)
    if weighted >= 7:
        _enqueue_action_if_new("generate_questions", "(ação) Gerar perguntas corretivas para reduzir recorrência de falhas.", priority=6)

    res = {
        "replayed_from_events": sum(counts.values()),
        "severity_score": weighted,
        "counts": counts,
        "queued": len(store.db.list_actions(limit=30)),
    }
    store.db.add_event("learning_replay", f"🔁 replay executado: eventos={res['replayed_from_events']} severidade={weighted}", meta_json=json.dumps(res, ensure_ascii=False))
    return res


@app.post("/api/memory/curation/run")
async def memory_curation_run(batch: int = 30):
    info = _run_memory_curation(batch_size=batch)
    return {"status": "ok", **info}


# --- Goals ---

@app.get("/api/goals")
async def goals_list(status: str = "all", limit: int = 30):
    s = None if status == "all" else status
    return {"goals": store.db.list_goals(status=s, limit=limit), "active": store.db.get_active_goal()}


@app.post("/api/goals/refresh")
async def goals_refresh():
    info = _refresh_goals_from_context()
    return {"status": "ok", **info}


@app.post("/api/goals/{goal_id}/activate")
async def goal_activate(goal_id: int):
    ok = store.db.activate_goal(goal_id)
    if not ok:
        raise HTTPException(404, "Goal not found")
    return {"status": "active", "goal": store.db.get_active_goal()}


@app.post("/api/goals/{goal_id}/done")
async def goal_done(goal_id: int):
    store.db.mark_goal_done(goal_id)
    return {"status": "done"}

# --- Settings ---

@app.get("/api/settings")
async def get_settings():
    """Get current settings (masked keys)."""
    s = settings.load_settings()
    masked = {}
    for k, v in s.items():
        if "key" in k and v:
            masked[k] = "..." + v[-4:] # Show only last 4 chars
        else:
            masked[k] = v
    return {"settings": masked}

@app.post("/api/settings")
async def update_settings(new_settings: SettingsModel):
    """Update settings."""
    current = settings.load_settings()
    to_save = {}
    
    # Only update provided fields (ignore empty strings if user didn't change)
    data = new_settings.dict(exclude_unset=True)
    
    for k, v in data.items():
        if v and v != "..." + current.get(k, "")[-4:]: # Check if it's not the masked value sent back
            to_save[k] = v
            
    if to_save:
        settings.save_settings(to_save)
        # Invalidate LLM clients cache to force reload with new keys
        llm.router.clients = {}
        
    return {"status": "updated", "updated_keys": list(to_save.keys())}

# --- Static UI ---
app.mount("/", StaticFiles(directory="/app/ui", html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
