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

from ultronpro import llm, knowledge_bridge, graph, settings, curiosity, conflicts, store, extract, planner, goals, autofeeder, policy, analogy
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

class ProcedureLearnRequest(BaseModel):
    observation_text: str
    domain: Optional[str] = None
    name: Optional[str] = None

class ProcedureRunRequest(BaseModel):
    procedure_id: int
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    score: float = 0.5
    success: bool = False
    notes: Optional[str] = None

class ProcedureSelectRequest(BaseModel):
    context_text: str
    domain: Optional[str] = None

class AnalogyTransferRequest(BaseModel):
    problem_text: str
    target_domain: Optional[str] = None

class WorkspacePublishRequest(BaseModel):
    module: str
    channel: str
    payload: Dict[str, Any] = {}
    salience: float = 0.5
    ttl_sec: int = 900

class PersistentGoalRequest(BaseModel):
    title: str
    description: Optional[str] = None
    proactive_actions: Optional[List[str]] = None
    interval_min: int = 60
    active_hours: Optional[List[int]] = None  # [start_hour, end_hour]

class ActionExecRequest(BaseModel):
    kind: str
    target: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    dry_run: bool = True
    reason: Optional[str] = None
    confirm_token: Optional[str] = None

class SelfPatchPrepareRequest(BaseModel):
    file_path: str
    old_text: str
    new_text: str
    reason: str

class SelfPatchApplyRequest(BaseModel):
    token: str

# --- Startup ---
_autofeeder_task = None
_autonomy_task = None
_judge_task = None
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
    "execute_procedure": 180,
    "execute_procedure_active": 240,
    "generate_analogy_hypothesis": 300,
}

# Etapa E: executor externo com segurança
EXTERNAL_ACTION_ALLOWLIST = {"notify_human"}
_external_confirm_tokens: dict[str, dict] = {}
_selfpatch_tokens: dict[str, dict] = {}
BENCHMARK_HISTORY_PATH = Path("/app/data/benchmark_history.json")
PERSISTENT_GOALS_PATH = Path("/app/data/persistent_goals.json")
PROCEDURE_ARTIFACTS_DIR = Path("/app/data/procedure_artifacts")


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


def _persistent_goals_load() -> dict:
    try:
        if PERSISTENT_GOALS_PATH.exists():
            d = json.loads(PERSISTENT_GOALS_PATH.read_text())
            if isinstance(d, dict):
                d.setdefault("goals", [])
                d.setdefault("active_id", None)
                return d
    except Exception:
        pass
    return {"goals": [], "active_id": None}


def _persistent_goals_save(data: dict):
    try:
        PERSISTENT_GOALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PERSISTENT_GOALS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        pass


def _persistent_goal_active() -> dict | None:
    d = _persistent_goals_load()
    aid = d.get("active_id")
    for g in d.get("goals", []):
        if g.get("id") == aid:
            return g
    return None


def _enqueue_from_persistent_goal():
    g = _persistent_goal_active()
    if not g:
        return 0

    now = time.time()
    now_local_h = int(time.localtime(now).tm_hour)

    # janela horária ativa
    ah = g.get("active_hours") or [8, 23]
    if isinstance(ah, list) and len(ah) == 2:
        h0, h1 = int(ah[0]), int(ah[1])
        if h0 <= h1:
            if not (h0 <= now_local_h <= h1):
                return 0
        else:
            # janela cruzando meia-noite
            if not (now_local_h >= h0 or now_local_h <= h1):
                return 0

    # frequência
    interval_min = max(5, int(g.get("interval_min") or 60))
    last_run_at = float(g.get("last_run_at") or 0)
    if (now - last_run_at) < (interval_min * 60):
        return 0

    actions = g.get("proactive_actions") or []
    count = 0
    for txt in actions[:4]:
        t = (txt or "").strip()
        if not t:
            continue
        _enqueue_action_if_new("ask_evidence", f"(ação-proativa-meta) {t}", priority=5, meta={"persistent_goal_id": g.get("id")})
        count += 1

    # persist last_run_at
    if count > 0:
        d = _persistent_goals_load()
        for it in d.get("goals", []):
            if it.get("id") == g.get("id"):
                it["last_run_at"] = now
                break
        _persistent_goals_save(d)

    return count


def _workspace_publish(module: str, channel: str, payload: dict, salience: float = 0.5, ttl_sec: int = 900) -> int:
    try:
        return store.publish_workspace(
            module=module,
            channel=channel,
            payload_json=json.dumps(payload or {}, ensure_ascii=False),
            salience=float(salience),
            ttl_sec=int(ttl_sec),
        )
    except Exception:
        return 0


def _workspace_recent(channels: list[str] | None = None, limit: int = 20) -> list[dict]:
    try:
        return store.read_workspace(channels=channels, limit=limit)
    except Exception:
        return []


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


def _selfpatch_allowed(path_str: str) -> bool:
    p = Path(path_str)
    try:
        rp = p.resolve()
    except Exception:
        return False
    allowed_roots = [Path('/app/ultronpro').resolve(), Path('/app/ui').resolve()]
    return any(str(rp).startswith(str(ar)) for ar in allowed_roots)


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


async def _run_judge_cycle(limit: int = 2, source: str = "loop") -> dict:
    """Integração real do Juiz: resolve conflitos em background sem clique humano."""
    open_conf = len(store.db.list_conflicts(status="open", limit=200))
    if open_conf <= 0:
        return {"open_conflicts": 0, "resolved": 0, "needs_human": 0, "attempted": 0}

    results = await conflicts.auto_resolve_all(limit=max(1, int(limit)))
    resolved = 0
    needs_human = 0
    for it in results:
        if it.get("resolved"):
            resolved += 1
            subj = it.get("subject") or "?"
            pred = it.get("predicate") or "?"
            chosen = it.get("chosen") or "?"
            store.db.add_insight(
                kind="judge_resolved",
                title="Juiz interno atualizou crença",
                text=f"Auto-correção: '{subj} {pred}' => '{chosen}'.",
                priority=5,
                conflict_id=it.get("conflict_id"),
            )
        elif it.get("needs_human"):
            needs_human += 1
            subj = it.get("subject") or "?"
            pred = it.get("predicate") or "?"
            store.db.add_insight(
                kind="judge_needs_human",
                title="Juiz pediu revisão humana",
                text=f"Não consegui fechar sozinho: '{subj} {pred}'. Preciso de evidência melhor para síntese final.",
                priority=4,
                conflict_id=it.get("conflict_id"),
            )

    if resolved or needs_human:
        store.db.add_event("judge_cycle", f"⚖️ juiz({source}): resolved={resolved}, needs_human={needs_human}, attempted={len(results)}")

    out = {"open_conflicts": open_conf, "resolved": resolved, "needs_human": needs_human, "attempted": len(results)}
    _workspace_publish("judge", "conflict.status", out, salience=0.75 if needs_human else 0.45, ttl_sec=900)
    return out


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
    out = {
        "decision_quality": round(float(quality), 3),
        "empty_activity": bool(empty_activity),
        "stuck_cycles": int(_autonomy_state.get("meta_stuck_cycles") or 0),
        "replans": int(_autonomy_state.get("meta_replans") or 0),
        "low_quality_streak": int(_autonomy_state.get("meta_low_quality_streak") or 0),
        "quality_history": list(_autonomy_state.get("meta_quality_history") or []),
        "snapshot": snap,
    }
    _workspace_publish("metacognition", "metacog.snapshot", out, salience=0.75 if quality < 0.3 else 0.45, ttl_sec=900)
    return out


def _self_awareness_snapshot() -> dict:
    """Modelo de autoconsciência funcional (não implica qualia real)."""
    m = _metacognition_tick()
    agi = _compute_agi_mode_metrics()
    ws = _workspace_recent(channels=["metacog.snapshot", "conflict.status", "analogy.transfer", "procedure.execution"], limit=12)

    dq = float(m.get("decision_quality") or 0.5)
    stress = min(1.0, (float(m.get("stuck_cycles") or 0) * 0.25) + (float(m.get("low_quality_streak") or 0) * 0.2))
    coherence = max(0.0, min(1.0, dq * 0.7 + (float(agi.get("agi_mode_percent") or 0) / 100.0) * 0.3))

    phenomenology_proxy = {
        "self_model": "functional-global-workspace",
        "note": "Proxy computacional de experiência interna; não comprova qualia fenomenológica.",
        "valence": round((coherence - stress), 3),
        "arousal": round(stress, 3),
        "sense_of_control": round(dq, 3),
        "global_broadcast_load": len(ws),
    }

    report = {
        "metacognition": m,
        "agi": agi,
        "phenomenology_proxy": phenomenology_proxy,
        "first_person_report": (
            f"Estado interno: controle={dq:.2f}, estresse={stress:.2f}, coerência={coherence:.2f}. "
            f"Estou priorizando sinais de maior saliência no workspace global."
        ),
    }

    _workspace_publish("self_model", "self.state", report, salience=0.85 if stress > 0.55 else 0.55, ttl_sec=1200)
    return report


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


def _infer_proc_type(domain: str | None, text: str) -> str:
    d = (domain or '').lower()
    t = (text or '').lower()
    if any(x in d or x in t for x in ['python', 'code', 'program', 'script']):
        return 'code'
    if any(x in d or x in t for x in ['jogo', 'game', 'xadrez', 'chess', 'estratég']):
        return 'strategy'
    if any(x in d or x in t for x in ['api', 'query', 'buscar', 'search', 'fetch']):
        return 'query'
    if any(x in d or x in t for x in ['análise', 'analysis', 'diagnóstico', 'debug']):
        return 'analysis'
    return 'analysis'


def _extract_procedure_from_text(observation_text: str, domain: str | None = None, name_hint: str | None = None) -> dict | None:
    txt = (observation_text or '').strip()
    if len(txt) < 20:
        return None

    # 1) tenta LLM
    prompt = f"""Extract a procedural skill from the observation below.
Return ONLY JSON with keys:
name, goal, domain, preconditions, steps (array of imperative strings), success_criteria.
Observation:\n{txt[:3500]}"""
    try:
        raw = llm.complete(prompt, strategy='reasoning', json_mode=True)
        data = json.loads(raw) if raw else {}
        steps = data.get('steps') if isinstance(data, dict) else None
        if isinstance(steps, list) and steps:
            dom = (domain or data.get('domain') or 'general').strip()
            return {
                'name': (name_hint or data.get('name') or 'Procedimento aprendido').strip(),
                'goal': data.get('goal'),
                'domain': dom,
                'proc_type': _infer_proc_type(dom, txt),
                'preconditions': data.get('preconditions'),
                'steps': [str(s).strip() for s in steps if str(s).strip()][:20],
                'success_criteria': data.get('success_criteria'),
            }
    except Exception:
        pass

    # 2) fallback regex (sem depender de LLM)
    import re
    parts = re.split(r"(?:^|\n|\s)(?:\d+\)|\d+\.|-\s)", txt)
    steps = [re.sub(r"\s+", " ", p).strip(' .;:-') for p in parts if len(re.sub(r"\s+", " ", p).strip()) > 6]
    if len(steps) < 2:
        # split por frases imperativas simples
        sents = re.split(r"[\.;\n]", txt)
        steps = [re.sub(r"\s+", " ", s).strip() for s in sents if len(s.strip()) > 8][:8]

    if len(steps) < 2:
        return None

    dom = (domain or 'general').strip()
    return {
        'name': (name_hint or 'Procedimento aprendido').strip(),
        'goal': f"Executar procedimento observado: {(name_hint or 'tarefa')}",
        'domain': dom,
        'proc_type': _infer_proc_type(dom, txt),
        'preconditions': None,
        'steps': steps[:20],
        'success_criteria': 'Executar passos com resultado útil e reproduzível',
    }


def _select_procedure(context_text: str, domain: str | None = None) -> dict | None:
    ctx = (context_text or '').lower()
    wanted_domain = (domain or '').strip().lower()

    procs = store.list_procedures(limit=80, domain=domain)
    if not procs and wanted_domain:
        # fallback: tenta pool global e filtra por match parcial de domínio
        allp = store.list_procedures(limit=80, domain=None)
        procs = [p for p in allp if wanted_domain in str((p.get('domain') or '')).lower()]
    if not procs:
        return None

    def score(p: dict) -> float:
        name = (p.get('name') or '').lower()
        goal = (p.get('goal') or '').lower()
        d = (p.get('domain') or '').lower()
        ptype = (p.get('proc_type') or 'analysis').lower()
        att = int(p.get('attempts') or 0)
        suc = int(p.get('successes') or 0)
        sr = suc / max(1, att)
        base = float(p.get('avg_score') or 0.0) * 0.5 + sr * 0.3
        overlap = 0.0
        words = set([x for x in ctx.split() if len(x) >= 4][:24])
        for w in words:
            if w in name or w in goal:
                overlap += 0.08
            if d and w in d:
                overlap += 0.12
        if wanted_domain and d and (wanted_domain in d or d in wanted_domain):
            overlap += 0.2

        # preferência por tipo de procedimento conforme contexto
        if any(k in ctx for k in ['python','código','codigo','função','funcao','script']) and ptype == 'code':
            overlap += 0.25
        if any(k in ctx for k in ['jogo','game','xadrez','chess']) and ptype == 'strategy':
            overlap += 0.25
        if any(k in ctx for k in ['buscar','consulta','query','search','api']) and ptype == 'query':
            overlap += 0.25

        return base + min(0.8, overlap)

    ranked = sorted(procs, key=score, reverse=True)
    best = ranked[0]
    if score(best) < 0.05:
        return None
    return best


def _evaluate_procedure_output(output_text: str, success_criteria: str | None = None) -> tuple[float, bool]:
    out = (output_text or '').strip()
    if not out:
        return 0.0, False

    # fallback heurístico
    score = 0.45
    if len(out) > 120:
        score += 0.15
    if 'erro' not in out.lower() and 'failed' not in out.lower():
        score += 0.1
    if success_criteria and any(w in out.lower() for w in str(success_criteria).lower().split()[:6]):
        score += 0.15

    # tenta avaliação LLM quando disponível
    try:
        prompt = f"""Evaluate this procedure output.
Return ONLY JSON: {{"score":0..1, "success":true/false}}.
Success criteria: {success_criteria or 'N/A'}
Output:\n{out[:2000]}"""
        raw = llm.complete(prompt, strategy='cheap', json_mode=True)
        d = json.loads(raw) if raw else {}
        if isinstance(d, dict) and d.get('score') is not None:
            lscore = float(d.get('score') or 0)
            lsuccess = bool(d.get('success'))
            score = (score * 0.4) + (lscore * 0.6)
            return max(0.0, min(1.0, score)), bool(lsuccess or score >= 0.62)
    except Exception:
        pass

    score = max(0.0, min(1.0, score))
    return score, bool(score >= 0.62)


def _execute_procedure_simulation(procedure_id: int, input_text: str | None = None) -> dict:
    p = store.get_procedure(procedure_id)
    if not p:
        return {"ok": False, "error": "procedure not found"}

    try:
        steps = json.loads(p.get('steps_json') or '[]')
    except Exception:
        steps = []

    if not steps:
        return {"ok": False, "error": "procedure has no steps"}

    ptype = (p.get('proc_type') or 'analysis').lower()
    in_txt = (input_text or '').strip()

    # executor fase 3: tipos de procedimento
    if ptype == 'code':
        executed = [f"[code-plan] {s}" for s in steps[:8]]
        skeleton = "def solution(input_data):\n    \"\"\"auto-generated skeleton\"\"\"\n    # TODO: implement steps\n    return input_data\n"
        out = "\n".join(executed) + "\n\n" + skeleton
    elif ptype == 'query':
        executed = [f"[query-plan] {s}" for s in steps[:8]]
        out = "\n".join(executed) + f"\n\nquery_context={in_txt[:180]}"
    elif ptype == 'strategy':
        executed = [f"[strategy-step] {s}" for s in steps[:8]]
        out = "\n".join(executed) + "\n\nnext_move_heuristic: maximize position advantage"
    else:
        executed = [f"[analysis-step] {s}" for s in steps[:8]]
        out = "\n".join(executed)

    score, success = _evaluate_procedure_output(out, success_criteria=p.get('success_criteria'))

    run_id = store.add_procedure_run(
        procedure_id=procedure_id,
        input_text=input_text,
        output_text=out,
        score=score,
        success=success,
        notes=f'simulated execution type={ptype}',
    )

    store.db.add_insight(
        kind='procedure_executed',
        title='Procedimento praticado',
        text=f"Pratiquei '{p.get('name')}' [{ptype}]. Score={score:.2f}, success={success}.",
        priority=3,
    )

    return {"ok": True, "run_id": run_id, "score": score, "success": success, "output": out, "procedure": p.get('name'), "proc_type": ptype}


def _execute_procedure_active(procedure_id: int, input_text: str | None = None, notify: bool = False) -> dict:
    """Executor procedural com efeitos reais locais (e notificação opcional)."""
    p = store.get_procedure(procedure_id)
    if not p:
        return {"ok": False, "error": "procedure not found"}

    try:
        steps = json.loads(p.get('steps_json') or '[]')
    except Exception:
        steps = []
    if not steps:
        return {"ok": False, "error": "procedure has no steps"}

    ptype = (p.get('proc_type') or 'analysis').lower()
    in_txt = (input_text or '').strip()
    PROCEDURE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    artifact_path = None
    out = ""

    if ptype == 'code':
        artifact_path = PROCEDURE_ARTIFACTS_DIR / f"proc_{procedure_id}_{ts}.py"
        code = (
            "def solution(input_data):\n"
            "    \"\"\"Generated by Ultron procedural executor\"\"\"\n"
            "    # Steps:\n"
            + "\n".join([f"    # - {str(s)[:120]}" for s in steps[:10]])
            + "\n    return input_data\n"
        )
        artifact_path.write_text(code)
        out = f"wrote_code_artifact={artifact_path}\nsteps={len(steps[:10])}"
    elif ptype == 'query':
        q = (in_txt or p.get('goal') or p.get('name') or 'general').strip()[:220]
        try:
            kb = search_knowledge(q, top_k=5)
            txt = json.dumps(kb, ensure_ascii=False)[:4000]
        except Exception as e:
            txt = f"query_error: {e}"
        artifact_path = PROCEDURE_ARTIFACTS_DIR / f"proc_{procedure_id}_{ts}.query.txt"
        artifact_path.write_text(txt)
        out = f"query='{q}'\nresult_artifact={artifact_path}"
    elif ptype == 'strategy':
        plan = {
            "procedure": p.get('name'),
            "input": in_txt[:500],
            "steps": [str(s)[:180] for s in steps[:12]],
            "heuristic": "maximize expected utility under constraints",
            "created_at": ts,
        }
        artifact_path = PROCEDURE_ARTIFACTS_DIR / f"proc_{procedure_id}_{ts}.strategy.json"
        artifact_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
        out = f"strategy_plan_artifact={artifact_path}"
    else:
        report = "\n".join([f"- {str(s)[:180]}" for s in steps[:12]])
        artifact_path = PROCEDURE_ARTIFACTS_DIR / f"proc_{procedure_id}_{ts}.analysis.md"
        artifact_path.write_text(f"# Procedural Analysis\n\n## Procedure\n{p.get('name')}\n\n## Input\n{in_txt[:800]}\n\n## Steps\n{report}\n")
        out = f"analysis_artifact={artifact_path}"

    score, success = _evaluate_procedure_output(out, success_criteria=p.get('success_criteria'))
    run_id = store.add_procedure_run(
        procedure_id=procedure_id,
        input_text=input_text,
        output_text=out,
        score=score,
        success=success,
        notes=f'active execution type={ptype}',
    )

    store.db.add_event(
        'procedure_active_executed',
        f"⚙️ active procedure '{p.get('name')}' [{ptype}] => {artifact_path}",
        meta_json=json.dumps({"procedure_id": procedure_id, "proc_type": ptype, "artifact": str(artifact_path), "score": score, "success": success}, ensure_ascii=False),
    )

    if notify:
        store.db.add_event(
            'external_action_executed',
            f"📣 notify_human: Procedimento '{p.get('name')}' executado ativamente (score={score:.2f}).",
            meta_json=json.dumps({"kind": "notify_human", "audit_hash": _compute_audit_hash({"kind":"notify_human","procedure_id":procedure_id,"run_id":run_id})}, ensure_ascii=False),
        )

    _workspace_publish(
        "procedural_executor",
        "procedure.execution",
        {"procedure_id": procedure_id, "name": p.get('name'), "proc_type": ptype, "score": score, "success": success, "artifact": str(artifact_path) if artifact_path else None},
        salience=0.7 if not success else 0.5,
        ttl_sec=1200,
    )

    return {
        "ok": True,
        "run_id": run_id,
        "score": score,
        "success": success,
        "procedure": p.get('name'),
        "proc_type": ptype,
        "artifact": str(artifact_path) if artifact_path else None,
        "output": out,
        "active": True,
    }


async def _run_analogy_transfer(problem_text: str, target_domain: str | None = None) -> dict:
    kb_ctx: list[str] = []
    try:
        kb = await search_knowledge(problem_text[:240], top_k=5)
        if isinstance(kb, list):
            for it in kb[:5]:
                if isinstance(it, dict):
                    kb_ctx.append(str(it.get('content') or it.get('text') or '')[:320])
                else:
                    kb_ctx.append(str(it)[:320])
    except Exception:
        pass

    cand = analogy.propose_analogy(problem_text, target_domain=target_domain, context_snippets=kb_ctx)
    if not cand:
        return {"status": "no_candidate"}

    val = analogy.validate_analogy(cand)
    applied = analogy.apply_analogy(cand, problem_text)
    st = "accepted" if val.get('valid') else "rejected"

    aid = store.add_analogy(
        source_domain=cand.get('source_domain'),
        target_domain=(target_domain or cand.get('target_domain')),
        source_concept=cand.get('source_concept'),
        target_concept=cand.get('target_concept'),
        mapping_json=json.dumps(cand.get('mapping') or {}, ensure_ascii=False),
        inference_rule=applied.get('derived_rule'),
        confidence=float(val.get('confidence') or cand.get('confidence') or 0.5),
        status=st,
        evidence_refs_json=json.dumps(kb_ctx[:3], ensure_ascii=False),
        notes="; ".join(val.get('reasons') or []),
    )

    store.db.add_insight(
        kind='analogy_transfer',
        title='Transferência por analogia',
        text=f"Analogia {st}: {cand.get('source_domain')} -> {(target_domain or cand.get('target_domain'))}. Regra: {applied.get('derived_rule')}",
        priority=4,
        meta_json=json.dumps({"analogy_id": aid, "confidence": val.get('confidence'), "mapping": cand.get('mapping')}, ensure_ascii=False),
    )
    _workspace_publish(
        "analogy",
        "analogy.transfer",
        {"status": st, "analogy_id": aid, "target_domain": (target_domain or cand.get('target_domain')), "confidence": val.get('confidence'), "derived_rule": applied.get('derived_rule')},
        salience=0.8 if st == "accepted" else 0.55,
        ttl_sec=1800,
    )

    return {"status": st, "analogy_id": aid, "candidate": cand, "validation": val, "applied": applied}


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
            jr = await _run_judge_cycle(limit=1, source="action")
            store.db.add_event("action_done", f"🤖 ação #{aid}: auto_resolve_conflicts ({jr.get('attempted')} tentativas, resolved={jr.get('resolved')})")
        elif kind == "curate_memory":
            info = _run_memory_curation(batch_size=30)
            store.db.add_event("action_done", f"🤖 ação #{aid}: curate_memory ({info.get('scanned')} itens)")
        elif kind == "prune_memory":
            n = store.db.prune_low_utility_experiences(limit=200, focus_terms=_goal_focus_terms())
            store.db.add_event("action_done", f"🤖 ação #{aid}: prune_memory ({n} arquivadas)")
        elif kind == "execute_procedure":
            pid = int((meta or {}).get('procedure_id') or 0)
            if pid <= 0:
                store.db.add_event("action_skipped", f"↷ ação #{aid} execute_procedure sem procedure_id")
            else:
                res = _execute_procedure_simulation(pid, input_text=(meta or {}).get('input_text'))
                store.db.add_event("action_done", f"🤖 ação #{aid}: execute_procedure pid={pid} ok={res.get('ok')} score={res.get('score')}")
        elif kind == "execute_procedure_active":
            pid = int((meta or {}).get('procedure_id') or 0)
            if pid <= 0:
                store.db.add_event("action_skipped", f"↷ ação #{aid} execute_procedure_active sem procedure_id")
            else:
                res = _execute_procedure_active(
                    pid,
                    input_text=(meta or {}).get('input_text'),
                    notify=bool((meta or {}).get('notify')),
                )
                store.db.add_event("action_done", f"🤖 ação #{aid}: execute_procedure_active pid={pid} ok={res.get('ok')} score={res.get('score')}")
        elif kind == "generate_analogy_hypothesis":
            ptxt = str((meta or {}).get('problem_text') or text or '')
            td = (meta or {}).get('target_domain')
            res = await _run_analogy_transfer(ptxt, target_domain=td)
            store.db.add_event("action_done", f"🤖 ação #{aid}: generate_analogy_hypothesis status={res.get('status')}")
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

                try:
                    pc = store.db.list_prioritized_conflicts(limit=1)
                    if pc:
                        c0 = pc[0]
                        _enqueue_action_if_new(
                            "generate_analogy_hypothesis",
                            f"(ação) Tentar transferência analógica para '{c0.get('subject')} {c0.get('predicate')}'.",
                            priority=5,
                            meta={
                                "conflict_id": c0.get("id"),
                                "problem_text": f"{c0.get('subject')} {c0.get('predicate')}",
                                "target_domain": c0.get('predicate'),
                            },
                        )
                except Exception as e:
                    logger.debug(f"Analogy planning skipped: {e}")

            # plano (determinístico + ocasional improv)
            try:
                for p in planner.propose_actions(store.db)[:3]:
                    _enqueue_action_if_new(p.kind, p.text, int(p.priority or 0), p.meta)
            except Exception as e:
                logger.debug(f"Planner skipped: {e}")

            # prática procedural (domínios não-declarativos)
            try:
                procs = store.db.list_procedures(limit=5)
                for pr in procs:
                    att = int(pr.get('attempts') or 0)
                    suc = int(pr.get('successes') or 0)
                    if att < 2 or (suc / max(1, att)) < 0.6:
                        _enqueue_action_if_new(
                            "ask_evidence",
                            f"(ação-procedural) Praticar procedimento '{pr.get('name')}' e reportar passos executados + resultado.",
                            priority=4,
                            meta={"procedure_id": pr.get('id')},
                        )

                # seleção automática por contexto recente + execução simulada
                recent_ctx = "\n".join([(e.get('text') or '') for e in store.db.list_experiences(limit=8)])
                sel = _select_procedure(recent_ctx)
                if sel:
                    _enqueue_action_if_new(
                        "execute_procedure_active",
                        f"(ação-procedural) Executar ATIVO procedimento selecionado: {sel.get('name')}",
                        priority=5,
                        meta={"procedure_id": sel.get('id'), "input_text": recent_ctx[:300], "notify": False},
                    )
            except Exception as e:
                logger.debug(f"Procedural planning skipped: {e}")

            # gestão de objetivos (Tarefa 2)
            try:
                goal_info = _refresh_goals_from_context()
                active_goal = goal_info.get("active")
                if active_goal:
                    k, t, pr, mt = _goal_to_action(active_goal)
                    _enqueue_action_if_new(k, t, pr, mt)
            except Exception as e:
                logger.debug(f"Goal planning skipped: {e}")

            # metas persistentes proativas (não dependem de conflito)
            try:
                _enqueue_from_persistent_goal()
            except Exception as e:
                logger.debug(f"Persistent goals skipped: {e}")

            # global workspace: acoplamento frouxo entre módulos
            try:
                ws = _workspace_recent(channels=["metacog.snapshot", "analogy.transfer", "conflict.status"], limit=8)
                for item in ws:
                    ch = item.get("channel")
                    payload = {}
                    try:
                        payload = json.loads(item.get("payload_json") or "{}")
                    except Exception:
                        payload = {}

                    if ch == "metacog.snapshot":
                        dq = float(payload.get("decision_quality") or 0.5)
                        if dq < 0.25:
                            _enqueue_action_if_new(
                                "curate_memory",
                                "(ação-workspace) baixa qualidade decisória detectada; executar curadoria para recuperar sinal.",
                                priority=6,
                            )
                    elif ch == "analogy.transfer" and payload.get("status") == "accepted":
                        _enqueue_action_if_new(
                            "ask_evidence",
                            f"(ação-workspace) Validar em evidência direta a regra analógica: {payload.get('derived_rule')}",
                            priority=5,
                            meta={"analogy_id": payload.get("analogy_id")},
                        )
                    elif ch == "conflict.status" and int(payload.get("needs_human") or 0) > 0:
                        _enqueue_action_if_new(
                            "ask_evidence",
                            "(ação-workspace) Juiz pediu ajuda humana; solicitar evidência objetiva para conflitos críticos.",
                            priority=6,
                        )
            except Exception as e:
                logger.debug(f"Workspace coupling skipped: {e}")

            # metacognição (Etapa D) + auto-modelo global
            try:
                _self_awareness_snapshot()
            except Exception as e:
                logger.debug(f"Metacognition/self-model skipped: {e}")

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


async def judge_loop():
    """Loop dedicado do Juiz para auto-correção contínua."""
    logger.info("Judge loop started")
    await asyncio.sleep(25)
    while True:
        try:
            await _run_judge_cycle(limit=2, source="judge_loop")
        except Exception as e:
            logger.error(f"Judge loop error: {e}")
        await asyncio.sleep(30)


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
    global _autofeeder_task, _autonomy_task, _judge_task
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
    _judge_task = asyncio.create_task(judge_loop())
    logger.info("Autofeeder + Autonomy + Judge tasks created")

@app.on_event("shutdown")
async def shutdown_event():
    global _autofeeder_task, _autonomy_task, _judge_task
    for t in (_autofeeder_task, _autonomy_task, _judge_task):
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
    try:
        curiosity.mark_question_failure(req.question_id)
    except Exception:
        pass
    store.dismiss_question(req.question_id)
    return {"status": "dismissed"}

# --- Graph & Events ---

@app.get("/api/graph/triples")
async def get_triples(since_id: int = 0, limit: int = 500):
    return {"triples": store.get_triples(since_id, limit)}

@app.get("/api/events")
async def get_events(since_id: int = 0, limit: int = 50):
    return {"events": store.get_events(since_id, limit)}

@app.get("/api/insights")
async def get_insights(limit: int = 50, query: str = ""):
    if (query or "").strip():
        return {"insights": store.search_insights(query, limit)}
    return {"insights": store.list_insights(limit)}

@app.post("/api/insights/emit")
async def emit_insight(title: str, text: str, kind: str = "manual", priority: int = 3):
    iid = store.add_insight(kind=kind, title=title, text=text, priority=priority)
    return {"status": "ok", "id": iid}

@app.get("/api/sources")
async def get_sources(limit: int = 50):
    return {"sources": store.get_sources(limit)}

@app.post("/api/sources/rebuild")
async def rebuild_sources(limit: int = 10000):
    added = store.db.rebuild_sources_from_experiences(limit=limit)
    return {"status": "ok", "added": added}

# --- Curiosity ---

@app.post("/api/curiosity/refresh")
async def refresh_curiosity(target_count: int = 5):
    """Trigger adaptive curiosity generation."""
    count = curiosity.refresh_questions(target_count=target_count)
    oq = store.db.list_open_questions_full(limit=20)
    return {"new_questions": count, "open_questions": len(oq)}

@app.get("/api/curiosity/stats")
async def curiosity_stats():
    return {"stats": curiosity.get_stats()}


@app.get("/api/curiosity/queue")
async def curiosity_queue(limit: int = 20):
    items = store.db.list_open_questions_full(limit=limit)
    return {"items": items, "count": len(items)}

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
    info = await _run_judge_cycle(limit=3, source="api")
    return info


@app.get("/api/conflicts-prioritized")
async def prioritized_conflicts(limit: int = 10):
    return {"conflicts": store.db.list_prioritized_conflicts(limit=limit)}


@app.get("/api/judge/status")
async def judge_status():
    open_conf = len(store.db.list_conflicts(status="open", limit=500))
    return {"open_conflicts": open_conf, "retry_cooldown_hours": 1.0}


@app.post("/api/judge/run")
async def judge_run(limit: int = 2):
    return await _run_judge_cycle(limit=limit, source="manual")


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


@app.get("/api/self-awareness/status")
async def self_awareness_status():
    return _self_awareness_snapshot()


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


# --- Adaptabilidade: policy dinâmico + self-patch supervisionado ---

@app.get("/api/policy/runtime")
async def policy_runtime_get():
    from ultronpro.policy import _load_runtime_rules
    return {"rules": _load_runtime_rules()}


@app.post("/api/policy/runtime")
async def policy_runtime_set(rules: Dict[str, Any]):
    from ultronpro.policy import RULES_PATH
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULES_PATH.write_text(json.dumps(rules or {}, ensure_ascii=False, indent=2))
    store.db.add_event("policy_runtime_updated", "🧩 regras dinâmicas de policy atualizadas")
    return {"status": "ok"}


@app.post("/api/selfpatch/prepare")
async def selfpatch_prepare(req: SelfPatchPrepareRequest):
    fp = (req.file_path or "").strip()
    if not _selfpatch_allowed(fp):
        raise HTTPException(403, "file_path not allowed")
    if not (req.reason or "").strip():
        raise HTTPException(400, "reason required")

    p = Path(fp)
    if not p.exists():
        raise HTTPException(404, "file not found")
    txt = p.read_text()
    if req.old_text not in txt:
        raise HTTPException(400, "old_text not found")

    token = secrets.token_urlsafe(18)
    preview = txt.replace(req.old_text, req.new_text, 1)
    _selfpatch_tokens[token] = {
        "file_path": fp,
        "old_text": req.old_text,
        "new_text": req.new_text,
        "reason": req.reason[:200],
        "expires_at": time.time() + 300,
        "diff_hash": hashlib.sha256((req.old_text + "->" + req.new_text).encode("utf-8")).hexdigest(),
    }
    store.db.add_event("selfpatch_prepared", f"🧪 selfpatch prepared for {fp}")
    return {"status": "prepared", "token": token, "diff_hash": _selfpatch_tokens[token]["diff_hash"], "preview_chars": len(preview)}


@app.post("/api/selfpatch/apply")
async def selfpatch_apply(req: SelfPatchApplyRequest):
    t = _selfpatch_tokens.get(req.token)
    if not t:
        raise HTTPException(400, "invalid token")
    if float(t.get("expires_at") or 0) < time.time():
        _selfpatch_tokens.pop(req.token, None)
        raise HTTPException(400, "token expired")

    p = Path(t["file_path"])
    txt = p.read_text()
    if t["old_text"] not in txt:
        raise HTTPException(400, "old_text no longer present")

    backup_dir = Path('/app/data/selfpatch_backups')
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = str(backup_dir / f"{p.name}.{int(time.time())}.bak")
    Path(backup).write_text(txt)

    try:
        p.write_text(txt.replace(t["old_text"], t["new_text"], 1))
        store.db.add_event("selfpatch_applied", f"🛠️ selfpatch applied to {p.name}", meta_json=json.dumps({"diff_hash": t.get("diff_hash"), "reason": t.get("reason")}, ensure_ascii=False))
        _selfpatch_tokens.pop(req.token, None)
        return {"status": "applied", "file": str(p), "backup": backup}
    except PermissionError:
        # fallback: persist proposal for host-side/manual apply
        pending_dir = Path('/app/data/selfpatch_pending')
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_path = pending_dir / f"patch_{int(time.time())}_{p.name}.json"
        pending_path.write_text(json.dumps(t, ensure_ascii=False, indent=2))
        store.db.add_event("selfpatch_pending", f"🧩 selfpatch pending (sem permissão de escrita em {p.name})", meta_json=json.dumps({"pending": str(pending_path), "diff_hash": t.get("diff_hash")}, ensure_ascii=False))
        _selfpatch_tokens.pop(req.token, None)
        return {"status": "pending_manual", "file": str(p), "pending": str(pending_path), "backup": backup}


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

@app.get("/api/goals/persistent")
async def persistent_goals_list():
    data = _persistent_goals_load()
    active = _persistent_goal_active()
    return {"goals": data.get("goals", []), "active": active, "active_id": data.get("active_id")}


@app.post("/api/goals/persistent")
async def persistent_goals_add(req: PersistentGoalRequest):
    t = (req.title or "").strip()
    if not t:
        raise HTTPException(400, "title required")

    data = _persistent_goals_load()
    gid = f"pg_{int(time.time())}_{secrets.token_hex(3)}"
    actions = req.proactive_actions or [
        f"Que evidência devo coletar hoje para avançar: {t}?",
        f"Qual experimento mental simples valida a meta: {t}?",
    ]
    interval_min = max(5, int(req.interval_min or 60))
    active_hours = req.active_hours if (req.active_hours and len(req.active_hours) == 2) else [8, 23]
    g = {
        "id": gid,
        "title": t,
        "description": (req.description or "").strip() or None,
        "proactive_actions": actions[:8],
        "interval_min": interval_min,
        "active_hours": [int(active_hours[0]), int(active_hours[1])],
        "last_run_at": 0,
        "created_at": int(time.time()),
    }
    data.setdefault("goals", []).append(g)
    if not data.get("active_id"):
        data["active_id"] = gid
    _persistent_goals_save(data)
    store.db.add_event("persistent_goal_added", f"🎯 meta persistente criada: {t}")
    return {"status": "ok", "goal": g, "active_id": data.get("active_id")}


@app.post("/api/goals/persistent/{goal_id}/activate")
async def persistent_goals_activate(goal_id: str):
    data = _persistent_goals_load()
    exists = any(g.get("id") == goal_id for g in data.get("goals", []))
    if not exists:
        raise HTTPException(404, "Persistent goal not found")
    data["active_id"] = goal_id
    _persistent_goals_save(data)
    store.db.add_event("persistent_goal_activated", f"🎯 meta persistente ativada: {goal_id}")
    return {"status": "ok", "active_id": goal_id}


@app.post("/api/goals/persistent/{goal_id}/schedule")
async def persistent_goals_schedule(goal_id: str, interval_min: int = 60, start_hour: int = 8, end_hour: int = 23):
    data = _persistent_goals_load()
    found = False
    for g in data.get("goals", []):
        if g.get("id") == goal_id:
            g["interval_min"] = max(5, int(interval_min))
            g["active_hours"] = [max(0, min(23, int(start_hour))), max(0, min(23, int(end_hour)))]
            found = True
            break
    if not found:
        raise HTTPException(404, "Persistent goal not found")
    _persistent_goals_save(data)
    store.db.add_event("persistent_goal_scheduled", f"🗓️ meta persistente agenda atualizada: {goal_id}")
    return {"status": "ok", "goal_id": goal_id}


@app.delete("/api/goals/persistent/{goal_id}")
async def persistent_goals_delete(goal_id: str):
    data = _persistent_goals_load()
    goals0 = data.get("goals", [])
    goals1 = [g for g in goals0 if g.get("id") != goal_id]
    if len(goals1) == len(goals0):
        raise HTTPException(404, "Persistent goal not found")
    data["goals"] = goals1
    if data.get("active_id") == goal_id:
        data["active_id"] = goals1[0].get("id") if goals1 else None
    _persistent_goals_save(data)
    store.db.add_event("persistent_goal_deleted", f"🗑️ meta persistente removida: {goal_id}")
    return {"status": "ok", "active_id": data.get("active_id")}


@app.get("/api/procedures")
async def procedures_list(limit: int = 50, domain: str = ""):
    d = domain.strip() or None
    return {"procedures": store.list_procedures(limit=limit, domain=d)}


@app.post("/api/procedures/learn")
async def procedures_learn(req: ProcedureLearnRequest):
    p = _extract_procedure_from_text(req.observation_text, domain=req.domain, name_hint=req.name)
    if not p:
        raise HTTPException(400, "Could not extract procedure")

    pid = store.add_procedure(
        name=p['name'],
        goal=p.get('goal'),
        steps_json=json.dumps(p.get('steps') or [], ensure_ascii=False),
        domain=p.get('domain'),
        proc_type=p.get('proc_type') or 'analysis',
        preconditions=p.get('preconditions'),
        success_criteria=p.get('success_criteria'),
    )
    store.db.add_insight("procedure_learned", "Nova habilidade procedural", f"Aprendi procedimento: {p['name']} ({p.get('domain')}).", priority=4)
    return {"status": "ok", "procedure_id": pid, "procedure": p}


@app.post("/api/procedures/run-log")
async def procedures_run_log(req: ProcedureRunRequest):
    rid = store.add_procedure_run(
        procedure_id=req.procedure_id,
        input_text=req.input_text,
        output_text=req.output_text,
        score=float(req.score),
        success=bool(req.success),
        notes=req.notes,
    )
    return {"status": "ok", "run_id": rid}


@app.post("/api/procedures/select")
async def procedures_select(req: ProcedureSelectRequest):
    sel = _select_procedure(req.context_text, domain=req.domain)
    return {"selected": sel}


@app.post("/api/procedures/execute")
async def procedures_execute(procedure_id: int, input_text: str = ""):
    return _execute_procedure_simulation(procedure_id, input_text=input_text)


@app.post("/api/procedures/execute-active")
async def procedures_execute_active(procedure_id: int, input_text: str = "", notify: bool = False):
    return _execute_procedure_active(procedure_id, input_text=input_text, notify=notify)


@app.post("/api/analogy/transfer")
async def analogy_transfer(req: AnalogyTransferRequest):
    return await _run_analogy_transfer(req.problem_text, target_domain=req.target_domain)


@app.get("/api/analogies")
async def analogies_list(limit: int = 50, status: str = "", target_domain: str = ""):
    s = status.strip() or None
    td = target_domain.strip() or None
    return {"analogies": store.list_analogies(limit=limit, status=s, target_domain=td)}


@app.post("/api/workspace/publish")
async def workspace_publish(req: WorkspacePublishRequest):
    wid = _workspace_publish(req.module, req.channel, req.payload or {}, salience=float(req.salience), ttl_sec=int(req.ttl_sec))
    return {"status": "ok", "id": wid}


@app.get("/api/workspace/read")
async def workspace_read(limit: int = 30, channels: str = "", include_expired: bool = False):
    chs = [c.strip() for c in channels.split(",") if c.strip()] if channels else None
    return {"items": store.read_workspace(channels=chs, limit=limit, include_expired=include_expired)}


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
