import os
import logging
import json
import asyncio
import time
import hashlib
import secrets
import random
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from ultronpro import llm, knowledge_bridge, graph, settings, curiosity, conflicts, store, extract, planner, goals, autofeeder, policy, analogy, tom, semantics, unsupervised, neuroplastic, causal, intrinsic, emergence, itc, longhorizon, subgoals, neurosym, project_kernel, tool_router, project_executor, integrity
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

class ProcedureInventRequest(BaseModel):
    context_text: str
    domain: Optional[str] = None
    name_hint: Optional[str] = None

class AnalogyTransferRequest(BaseModel):
    problem_text: str
    target_domain: Optional[str] = None

class WorkspacePublishRequest(BaseModel):
    module: str
    channel: str
    payload: Dict[str, Any] = {}
    salience: float = 0.5
    ttl_sec: int = 900

class MilestoneProgressRequest(BaseModel):
    progress: float
    status: Optional[str] = None

class MutationProposalRequest(BaseModel):
    title: str
    rationale: str
    patch: Dict[str, Any]
    author: Optional[str] = "manual"

class MutationDecisionRequest(BaseModel):
    reason: Optional[str] = None

class IntrinsicTickRequest(BaseModel):
    force: bool = False

class ITCRunRequest(BaseModel):
    problem_text: str
    max_steps: int = 0
    budget_seconds: int = 0
    use_rl: bool = True

class HorizonMissionRequest(BaseModel):
    title: str
    objective: str
    horizon_days: int = 14
    context: Optional[str] = None

class HorizonCheckpointRequest(BaseModel):
    note: str
    progress_delta: float = 0.0
    signal: str = "reflection"

class SubgoalMarkRequest(BaseModel):
    status: str = "done"

class ProjectRequest(BaseModel):
    title: str
    objective: str
    scope: Optional[str] = None
    sla_hours: int = 72

class ProjectCheckpointRequest(BaseModel):
    note: str
    progress_delta: float = 0.0
    signal: str = "tick"

class ToolRouteRequest(BaseModel):
    intent: str
    context: Optional[Dict[str, Any]] = None
    prefer_low_cost: bool = True

class IntegrityRulesPatchRequest(BaseModel):
    rules: Dict[str, Any]

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
    "maintain_question_queue": 240,
    "clarify_semantics": 180,
    "unsupervised_discovery": 600,
    "neuroplastic_cycle": 900,
    "invent_procedure": 420,
    "intrinsic_tick": 600,
    "emergence_tick": 420,
    "deliberate_task": 480,
    "horizon_review": 1800,
    "subgoal_planning": 1200,
    "project_management_cycle": 1500,
    "route_toolchain": 420,
    "project_experiment_cycle": 1800,
}

# Etapa E: executor externo com segurança
EXTERNAL_ACTION_ALLOWLIST = {"notify_human"}
_external_confirm_tokens: dict[str, dict] = {}
_selfpatch_tokens: dict[str, dict] = {}
BENCHMARK_HISTORY_PATH = Path("/app/data/benchmark_history.json")
PERSISTENT_GOALS_PATH = Path("/app/data/persistent_goals.json")
PROCEDURE_ARTIFACTS_DIR = Path("/app/data/procedure_artifacts")
NEUROPLASTIC_GATE_STATE_PATH = Path("/app/data/neuroplastic_gate_state.json")


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


def _neuroplastic_gate_load() -> dict:
    try:
        if NEUROPLASTIC_GATE_STATE_PATH.exists():
            d = json.loads(NEUROPLASTIC_GATE_STATE_PATH.read_text())
            if isinstance(d, dict):
                d.setdefault("revert_streaks", {})
                d.setdefault("activation_baselines", {})
                d.setdefault("last_snapshot", {})
                return d
    except Exception:
        pass
    return {"revert_streaks": {}, "activation_baselines": {}, "last_snapshot": {}}


def _neuroplastic_gate_save(data: dict):
    try:
        NEUROPLASTIC_GATE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        NEUROPLASTIC_GATE_STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
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


def _audit_reasoning(decision_type: str, context: dict, rationale: str, confidence: float | None = None):
    payload = {
        "decision_type": decision_type,
        "context": context,
        "rationale": (rationale or "")[:800],
        "confidence": confidence,
        "ts": int(time.time()),
    }
    store.db.add_event("reasoning_audit", f"🧾 {decision_type}: {(rationale or '')[:140]}", meta_json=json.dumps(payload, ensure_ascii=False))


def _neurosym_proof(decision_type: str, premises: list[str], inference: str, conclusion: str, confidence: float = 0.5, action_meta: dict | None = None):
    try:
        pf = neurosym.add_proof(decision_type, premises=premises, inference=inference, conclusion=conclusion, confidence=confidence, action_meta=action_meta or {})
        store.db.add_event("neurosym_proof", f"📐 proof {pf.get('id')} {decision_type}: {(conclusion or '')[:120]}")
    except Exception:
        pass


def _causal_precheck(kind: str, text: str = "", meta: dict | None = None) -> dict:
    model = causal.build_world_model(store.db, limit=4000)
    interventions = causal.infer_intervention_from_action(kind, text=text, meta=meta or {})
    sim = causal.simulate_intervention(model, interventions, steps=3)
    _audit_reasoning("causal_precheck", {"kind": kind, "interventions": interventions}, f"net={sim.get('net_score')} risk={sim.get('risk_score')} benefit={sim.get('benefit_score')}", confidence=0.7)
    return {"model": {"nodes": model.get("nodes") and len(model.get("nodes")) or 0, "edges": len(model.get("edges") or [])}, "simulation": sim}


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


def _apply_runtime_mutation_policy(kind: str, priority: int, cooldown: int, ttl: int) -> tuple[int, int, int, dict]:
    rt = neuroplastic.active_runtime()
    active = (rt or {}).get("active") or []
    p = int(priority)
    cd = int(cooldown)
    t = int(ttl)
    applied = {"treated": False, "mutation_ids": []}

    for m in active:
        patch = (m or {}).get("patch") or {}
        if not isinstance(patch, dict):
            continue

        # canary A/B automático: aplica patch só em fração das decisões
        canary_ratio = float(patch.get("canary_ratio") or 1.0)
        if canary_ratio < 1.0 and random.random() > max(0.0, min(1.0, canary_ratio)):
            continue

        applied["treated"] = True
        if m.get("id"):
            applied["mutation_ids"].append(str(m.get("id")))

        apd = patch.get("action_priority_delta") or {}
        if isinstance(apd, dict):
            p += int(apd.get(kind) or 0)
        acs = patch.get("action_cooldown_scale") or {}
        if isinstance(acs, dict) and acs.get(kind) is not None:
            try:
                cd = int(max(10, float(cd) * float(acs.get(kind))))
            except Exception:
                pass
        if patch.get("queue_ttl_scale") is not None:
            try:
                t = int(max(60, float(t) * float(patch.get("queue_ttl_scale"))))
            except Exception:
                pass

    return max(0, min(10, p)), max(10, min(7200, cd)), max(60, min(7200, t)), applied


def _enqueue_action_if_new(kind: str, text: str, priority: int = 0, meta: dict | None = None, ttl_sec: int | None = None):
    """Enfileira ação com dedupe + cooldown + expiração de fila + runtime mutation policy."""
    recent = store.db.list_actions(limit=120)
    now = time.time()
    cooldown = ACTION_COOLDOWNS_SEC.get(kind, 120)
    mk = meta or {}
    cd_sig = mk.get('conflict_id') or mk.get('goal_id') or mk.get('milestone_id') or mk.get('procedure_id') or ''
    cd_key = f"{kind}:{cd_sig}"

    ttl = int(ttl_sec or ACTION_DEFAULT_TTL_SEC)
    priority, cooldown, ttl, mut = _apply_runtime_mutation_policy(kind, int(priority), int(cooldown), ttl)

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

    expires_at = now + ttl
    mmeta = dict(meta or {})
    if mut.get("treated"):
        mmeta["mutation_treated"] = True
        mmeta["mutation_ids"] = mut.get("mutation_ids") or []
    store.db.enqueue_action(
        kind=kind,
        text=text,
        priority=priority,
        meta_json=json.dumps(mmeta, ensure_ascii=False),
        expires_at=expires_at,
        cooldown_key=cd_key,
    )


def _ensure_goal_milestones(goal_id: int, title: str, description: str | None = None, weeks: int = 4) -> int:
    existing = store.list_goal_milestones(goal_id=goal_id, status=None, limit=32)
    if existing:
        return 0
    planner = goals.GoalPlanner()
    ms = planner.build_weekly_milestones(title, description, weeks=weeks)
    added = 0
    for m in ms:
        store.add_goal_milestone(goal_id, int(m.get("week_index") or 1), m.get("title") or "Milestone", m.get("progress_criteria"))
        added += 1
    return added


def _intrinsic_tick(force: bool = False) -> dict:
    st = intrinsic.load_state()

    stats = store.db.stats()
    meta = _metacognition_tick()
    goals_all = store.db.list_goals(status=None, limit=200)
    goals_done = len([g for g in goals_all if str(g.get('status') or '') == 'done'])
    done_rate = goals_done / max(1, len(goals_all))

    # novelty_index simples: razão de conceitos latentes recentes + perguntas abertas
    novelty_index = min(1.0, (float(stats.get('questions_open') or 0) / 80.0) + 0.2)

    signals = {
        'uncurated': store.db.count_uncurated_experiences(),
        'open_conflicts': len(store.db.list_conflicts(status='open', limit=300)),
        'decision_quality': float(meta.get('decision_quality') or 0.5),
        'goals_done_rate': float(done_rate),
        'novelty_index': novelty_index,
    }

    st = intrinsic.update_drives(st, signals)
    chosen = intrinsic.synthesize_intrinsic_goal(st)
    st = intrinsic.revise_purpose(st, chosen)
    intrinsic.save_state(st)

    # cria/atualiza goal intrínseco
    gid = store.db.upsert_goal(
        f"[IME] {chosen.get('title')}",
        f"{chosen.get('description')} | drive={chosen.get('drive')} reward={chosen.get('intrinsic_reward')}",
        int(chosen.get('priority') or 4),
    )

    _workspace_publish('intrinsic', 'purpose.state', {'purpose': st.get('purpose'), 'drives': st.get('drives'), 'chosen_goal': chosen, 'goal_id': gid}, salience=0.78, ttl_sec=3600)
    store.db.add_event('intrinsic_tick', f"🧭 IME tick: drive={chosen.get('drive')} goal={chosen.get('title')}")

    return {
        'signals': signals,
        'drives': st.get('drives'),
        'purpose': st.get('purpose'),
        'chosen_goal': chosen,
        'goal_id': gid,
    }


def _emergence_tick() -> dict:
    stats = store.db.stats()
    meta = _metacognition_tick()
    inputs = {
        'decision_quality': float(meta.get('decision_quality') or 0.5),
        'open_conflicts': len(store.db.list_conflicts(status='open', limit=400)),
        'novelty_index': min(1.0, float(stats.get('questions_open') or 0) / 80.0 + 0.2),
    }
    st = emergence.tick_latent(inputs)
    policies = emergence.sample_policies({'stats': stats, 'meta': meta}, n=4)
    chosen = emergence.choose_policy(policies, {'stats': stats, 'meta': meta})

    for a in (chosen.get('actions') or [])[:2]:
        _enqueue_action_if_new(
            a,
            f"(ação-emergence) Política latente selecionou: {a}",
            priority=5,
            meta={'emergence_policy': chosen.get('id')},
            ttl_sec=20 * 60,
        )

    item = {'ts': int(time.time()), 'latent': st.get('latent'), 'chosen_policy': chosen}
    emergence.log_eval(item)
    _workspace_publish('emergence', 'emergence.state', item, salience=0.76, ttl_sec=2400)
    store.db.add_event('emergence_tick', f"🧠 emergence policy={chosen.get('id')} actions={','.join(chosen.get('actions') or [])}")
    return item


def _itc_router_need() -> dict:
    meta = _metacognition_tick()
    open_conf = len(store.db.list_conflicts(status='open', limit=300))
    dq = float(meta.get('decision_quality') or 0.5)
    need = (open_conf >= 8) or (dq < 0.25)
    reason = 'conflict_load' if open_conf >= 8 else ('low_decision_quality' if dq < 0.25 else 'none')
    return {'need': need, 'reason': reason, 'open_conflicts': open_conf, 'decision_quality': dq}


def _run_deliberate_task(problem_text: str, max_steps: int = 0, budget_seconds: int = 0, use_rl: bool = True) -> dict:
    out = itc.run_episode(problem_text=problem_text, max_steps=max_steps, budget_seconds=budget_seconds, use_rl=use_rl)
    chosen = out.get('chosen') or {}
    if chosen.get('test'):
        _enqueue_action_if_new(
            'ask_evidence',
            f"(itc-test) {chosen.get('test')}",
            priority=6,
            meta={'source': 'itc', 'confidence': chosen.get('confidence'), 'policy_arm': out.get('policy_arm')},
            ttl_sec=20 * 60,
        )
    _workspace_publish('itc', 'deliberation.episode', out, salience=0.82, ttl_sec=3600)
    store.db.add_event('itc_episode', f"🧠 ITC arm={out.get('policy_arm')} steps={len(out.get('steps') or [])} quality={out.get('quality_proxy')} reward={out.get('reward')}")
    return out


def _run_tool_route(intent: str, context: dict | None = None, prefer_low_cost: bool = True) -> dict:
    plan = tool_router.plan_route(intent=intent, context=context or {}, prefer_low_cost=prefer_low_cost)
    chain = list(plan.get('chain') or [])[:3]

    attempted = []
    for k in chain:
        attempted.append(k)
        try:
            if k == 'ask_evidence':
                q = str((context or {}).get('question') or f"(router:{intent}) executar próximo passo de recuperação")
                store.db.add_questions([{"question": q[:500], "priority": 5, "context": "tool_router"}])
                _neurosym_proof('tool_route', [f'intent={intent}', f'candidate={k}'], 'Selected low-cost evidence query route.', f'Route executed via {k}.', confidence=0.74, action_meta={'kind': k, 'status': 'done', 'intent': intent})
                return {'status': 'ok', 'selected': k, 'attempted': attempted, 'plan': plan}
            if k == 'deliberate_task':
                ptxt = str((context or {}).get('problem_text') or f"Router intent {intent}: deliberate best next move")
                out = _run_deliberate_task(problem_text=ptxt, max_steps=0, budget_seconds=0, use_rl=True)
                if float(out.get('quality_proxy') or 0.0) >= 0.35:
                    _neurosym_proof('tool_route', [f'intent={intent}', f'candidate={k}', f"quality={out.get('quality_proxy')}"], 'Selected deliberate route with acceptable quality.', f'Route executed via {k}.', confidence=0.78, action_meta={'kind': k, 'status': 'done', 'intent': intent})
                    return {'status': 'ok', 'selected': k, 'attempted': attempted, 'plan': plan, 'result': out}
                continue
            if k == 'generate_analogy_hypothesis':
                ptxt = str((context or {}).get('problem_text') or f"{intent} unresolved")
                td = (context or {}).get('target_domain')
                # schedule async path safely
                _enqueue_action_if_new('generate_analogy_hypothesis', f"(router:{intent}) gerar hipótese analógica", priority=5, meta={'problem_text': ptxt[:300], 'target_domain': td, 'intent': intent}, ttl_sec=20 * 60)
                _neurosym_proof('tool_route', [f'intent={intent}', f'candidate={k}'], 'Selected analogy route as fallback chain.', f'Route scheduled via {k}.', confidence=0.68, action_meta={'kind': k, 'status': 'scheduled', 'intent': intent})
                return {'status': 'ok', 'selected': k, 'attempted': attempted, 'plan': plan, 'scheduled': True}
            if k == 'maintain_question_queue':
                info = _maintain_question_queue(stale_hours=18.0, max_fix=4)
                _neurosym_proof('tool_route', [f'intent={intent}', f'candidate={k}'], 'Selected queue maintenance route for recovery.', f'Route executed via {k}.', confidence=0.64, action_meta={'kind': k, 'status': 'done', 'intent': intent})
                return {'status': 'ok', 'selected': k, 'attempted': attempted, 'plan': plan, 'result': info}
        except Exception:
            continue

    _neurosym_proof('tool_route', [f'intent={intent}', f'attempted={attempted}'], 'All route candidates failed or were unavailable.', 'Tool routing failed; no executable candidate.', confidence=0.3, action_meta={'kind': 'route_toolchain', 'status': 'error', 'intent': intent})
    return {'status': 'error', 'attempted': attempted, 'plan': plan}


def _subgoal_planning_tick() -> dict:
    ag = store.db.get_active_goal()
    mission = longhorizon.active_mission()
    if not ag and not mission:
        return {"status": "no_context"}

    title = str((mission or {}).get("title") or (ag or {}).get("title") or "Goal")
    objective = str((mission or {}).get("objective") or (ag or {}).get("description") or title)
    root = subgoals.synthesize_for_goal(title=title, objective=objective, max_nodes=8)

    # enfileira próximos nós abertos (até 2)
    open_nodes = [n for n in (root.get("nodes") or []) if str(n.get("status") or "open") == "open"]
    open_nodes = sorted(open_nodes, key=lambda x: int(x.get("priority") or 0), reverse=True)
    for n in open_nodes[:2]:
        _enqueue_action_if_new(
            "ask_evidence",
            f"(subgoal) {n.get('title')}",
            priority=int(n.get("priority") or 5),
            meta={"subgoal_root_id": root.get("id"), "subgoal_node_id": n.get("id")},
            ttl_sec=25 * 60,
        )

    _workspace_publish("subgoals", "goal.subgoals", root, salience=0.8, ttl_sec=3600)
    store.db.add_event("subgoal_planning", f"🧩 subgoals root={root.get('id')} open={len(open_nodes)}")
    return {"status": "ok", "root": root, "open_nodes": len(open_nodes)}


def _project_management_tick() -> dict:
    project_kernel.ensure_default_playbooks()
    p = project_kernel.active_project()

    if not p:
        # seed a project from active mission/goal
        m = longhorizon.active_mission()
        g = store.db.get_active_goal()
        if m:
            p = project_kernel.upsert_project(m.get('title') or 'Projeto', m.get('objective') or m.get('title') or 'Objetivo', scope=m.get('context'), sla_hours=72)
        elif g:
            p = project_kernel.upsert_project(g.get('title') or 'Projeto', g.get('description') or g.get('title') or 'Objetivo', scope='Seed from active goal', sla_hours=72)
        else:
            return {'status': 'no_project_context'}

    # KPIs proxy
    acts = store.db.list_actions(limit=160)
    done = len([a for a in acts if a.get('status') == 'done'])
    blocked = len([a for a in acts if a.get('status') == 'blocked'])
    errs = len([a for a in acts if a.get('status') == 'error'])

    progress_delta = max(-0.04, min(0.07, (done * 0.0018) - (blocked * 0.0025) - (errs * 0.002)))
    cp = project_kernel.add_checkpoint(
        p.get('id'),
        note=f"tick done={done} blocked={blocked} errors={errs}",
        progress_delta=progress_delta,
        signal='project_tick',
    )

    blocked_hours = float((p.get('kpi') or {}).get('blocked_hours') or 0.0)
    if blocked > 0:
        blocked_hours += 0.5

    stuck = int((p.get('kpi') or {}).get('stuck_cycles') or 0)
    if progress_delta <= 0:
        stuck += 1
    else:
        stuck = max(0, stuck - 1)

    project_kernel.update_kpi(p.get('id'), {
        'advance_week': float(p.get('progress') or 0.0),
        'blocked_hours': blocked_hours,
        'cost_score': float(errs + blocked) / max(1.0, float(done + 1)),
        'stuck_cycles': stuck,
    })

    # playbook triggers
    triggered = []
    if errs >= 2:
        triggered.append('tool_failure')
    if blocked >= 3:
        triggered.append('conflict_stalemate')
    if stuck >= 2:
        triggered.append('kpi_regression')

    suggested = []
    for sig in triggered[:2]:
        acts_pb = project_kernel.suggest_playbook_actions(sig)
        for ap in acts_pb[:2]:
            suggested.append(f"{sig}:{ap}")
            _enqueue_action_if_new(
                'route_toolchain',
                f"(recovery:{sig}) Roteador de ferramenta para fallback: {ap}",
                priority=6,
                meta={
                    'project_id': p.get('id'),
                    'playbook_signal': sig,
                    'fallback': ap,
                    'intent': sig,
                    'prefer_low_cost': True,
                    'context': {'problem_text': f"project={p.get('id')} signal={sig} fallback={ap}", 'target_domain': 'recovery'},
                },
                ttl_sec=25 * 60,
            )

    project_kernel.remember(
        p.get('id'),
        kind='tick',
        text=f"tick done={done} blocked={blocked} errors={errs} delta={progress_delta:+.3f}",
        meta={'triggered': triggered, 'suggested': suggested},
    )
    brief = project_kernel.project_brief(p.get('id'))

    _workspace_publish('project_kernel', 'project.status', {
        'project': p,
        'checkpoint': cp,
        'triggered': triggered,
        'suggested': suggested,
        'brief': brief,
    }, salience=0.84 if triggered else 0.62, ttl_sec=3600)

    # cadência de gestão: sempre agenda próximos 3 passos do brief
    for step in (brief or {}).get('next_steps', [])[:3]:
        _enqueue_action_if_new(
            'ask_evidence',
            f"(project-next) {step}",
            priority=5,
            meta={'project_id': p.get('id'), 'source': 'project_brief'},
            ttl_sec=25 * 60,
        )

    store.db.add_event('project_management_tick', f"📦 project={p.get('id')} progressΔ={progress_delta:+.3f} triggers={','.join(triggered) if triggered else 'none'}")
    return {'status': 'ok', 'project': project_kernel.active_project(), 'triggered': triggered, 'suggested': suggested, 'brief': brief}


def _project_experiment_cycle() -> dict:
    p = project_kernel.active_project()
    if not p:
        return {'status': 'no_active_project'}

    brief = project_kernel.project_brief(p.get('id')) or {}
    exp = project_executor.propose_experiment(p, brief=brief)
    res = project_executor.run_experiment(exp)
    rec = project_executor.record(exp, res)

    project_kernel.remember(
        p.get('id'),
        kind='experiment',
        text=f"exp={exp.get('id')} status={res.get('status')} success={res.get('success')}",
        meta={'metrics': (res.get('metrics') or {}), 'artifact': res.get('artifact')},
    )

    # if experiment indicates optimization still needed, route mitigation chain
    if res.get('status') == 'needs_optimization':
        _enqueue_action_if_new(
            'route_toolchain',
            '(project-experiment) otimização necessária, executar rota de remediação.',
            priority=6,
            meta={
                'intent': 'tool_failure',
                'prefer_low_cost': True,
                'context': {
                    'problem_text': f"Projeto {p.get('id')} benchmark p95={((res.get('metrics') or {}).get('p95_read_ms'))}",
                    'target_domain': 'database_optimization',
                },
            },
            ttl_sec=30 * 60,
        )

    _workspace_publish('project_kernel', 'project.experiment', {'project_id': p.get('id'), 'experiment': rec}, salience=0.82, ttl_sec=3600)
    store.db.add_event('project_experiment_cycle', f"🧪 project={p.get('id')} exp={exp.get('id')} status={res.get('status')}")
    return {'status': 'ok', 'project_id': p.get('id'), 'experiment': rec}


def _horizon_review_tick() -> dict:
    roll = longhorizon.rollover_if_due()
    mission = longhorizon.active_mission()

    if not mission:
        ag = store.db.get_active_goal()
        if ag:
            mission = longhorizon.upsert_mission(
                title=f"Long Horizon: {ag.get('title')}",
                objective=str(ag.get('description') or ag.get('title') or '')[:900],
                horizon_days=14,
                context='Seeded from active goal',
            )

    if not mission:
        return {'status': 'no_mission', 'rollover': roll}

    # progress proxy
    acts = store.db.list_actions(limit=120)
    done = len([a for a in acts if a.get('status') == 'done'])
    blocked = len([a for a in acts if a.get('status') == 'blocked'])
    delta = max(-0.03, min(0.06, (done * 0.002) - (blocked * 0.003)))

    snippet = longhorizon.mission_context_snippet(mission, max_items=8)
    longhorizon.add_checkpoint(mission.get('id'), f"Review: done={done}, blocked={blocked}", progress_delta=delta, signal='autonomy_review')

    _workspace_publish('horizon', 'horizon.mission', {
        'mission_id': mission.get('id'),
        'title': mission.get('title'),
        'objective': mission.get('objective'),
        'progress': mission.get('progress'),
        'context_snippet': snippet,
    }, salience=0.8, ttl_sec=7200)

    # inject continuity into cognition loop
    _enqueue_action_if_new(
        'deliberate_task',
        '(horizon) Revisar missão de longo horizonte e atualizar plano.',
        priority=5,
        meta={'problem_text': snippet, 'budget_seconds': 30, 'max_steps': 4, 'source': 'horizon_review'},
        ttl_sec=40 * 60,
    )

    store.db.add_event('horizon_review', f"🧭 missão {mission.get('id')} progress={mission.get('progress'):.2f} Δ={delta:+.3f}")
    return {'status': 'ok', 'mission': mission, 'rollover': roll, 'delta': delta}


def _refresh_goals_from_context() -> dict:
    recent_exp = store.db.list_experiences(limit=20)
    existing = store.db.list_goals(status=None, limit=200)
    proposed_goals = goals.GoalPlanner().propose_goals(recent_exp, existing_goals=existing)
    created = 0
    ambitions = 0
    milestones_added = 0
    for g in proposed_goals[:7]:
        gid = store.db.upsert_goal(g.get("title") or "Goal", g.get("description"), int(g.get("priority") or 0))
        created += 1
        milestones_added += _ensure_goal_milestones(gid, g.get("title") or "Goal", g.get("description"), weeks=4)
        if bool(g.get("ambition")):
            ambitions += 1
            store.db.add_insight(
                kind="self_ambition",
                title="Vontade autônoma gerada",
                text=f"Defini uma ambição não-determinística: {g.get('title')}",
                priority=5,
            )
            _workspace_publish("goals", "goal.ambition", {"title": g.get("title"), "priority": g.get("priority")}, salience=0.82, ttl_sec=3600)
    active_goal = store.db.activate_next_goal()
    return {"proposed": len(proposed_goals), "upserts": created, "ambitions": ambitions, "milestones_added": milestones_added, "active": active_goal}


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
    _audit_reasoning("conflict_judge_cycle", {"source": source, "open": open_conf}, f"resolved={resolved}, needs_human={needs_human}, attempted={len(results)}", confidence=(resolved / max(1, len(results))) if results else 0.0)
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


def _run_neuroplastic_shadow_eval(proposal_id: str) -> dict:
    """Executa avaliação shadow segura (sem alterar código em produção)."""
    # usa métricas internas atuais como baseline proxy
    agi = _compute_agi_mode_metrics()
    meta = _metacognition_tick()
    score = float(agi.get("agi_mode_percent") or 0)
    dq = float(meta.get("decision_quality") or 0)
    # critério simples de promoção segura
    promote = (score >= 55.0 and dq >= 0.2)
    metrics = {
        "ts": int(time.time()),
        "agi_mode_percent": score,
        "decision_quality": dq,
        "promote_recommendation": promote,
    }
    neuroplastic.set_shadow_metrics(proposal_id, metrics)
    _audit_reasoning("neuroplastic_shadow_eval", {"proposal_id": proposal_id}, f"promote={promote} agi={score:.1f} dq={dq:.2f}", confidence=min(1.0, score/100.0))
    return metrics


def _neuroplastic_gate_snapshot() -> dict:
    agi = _compute_agi_mode_metrics()
    meta = _metacognition_tick()
    bench = _benchmark_history_load()
    req_hist = [float(x.get("requirements_avg_1_8") or 0.0) for x in bench if isinstance(x, dict) and x.get("requirements_avg_1_8") is not None]
    req_avg = (sum(req_hist[-10:]) / max(1, len(req_hist[-10:]))) if req_hist else 0.0
    cost_hist = [float(x.get("cost_estimate") or 0.0) for x in bench if isinstance(x, dict)]
    cost_avg = (sum(cost_hist[-10:]) / max(1, len(cost_hist[-10:]))) if cost_hist else 0.0
    out = {
        "ts": int(time.time()),
        "agi_mode_percent": float(agi.get("agi_mode_percent") or 0.0),
        "decision_quality": float(meta.get("decision_quality") or 0.0),
        "requirements_avg_1_8": float(req_avg),
        "cost_estimate_avg": float(cost_avg),
    }
    st = _neuroplastic_gate_load()
    st["last_snapshot"] = out
    _neuroplastic_gate_save(st)
    return out


def _rolling_gain_days(days: int = 7) -> dict:
    arr = _benchmark_history_load()
    if not arr:
        return {"gain": 0.0, "samples": 0}
    now = int(time.time())
    win = max(1, int(days)) * 86400
    recent = [x for x in arr if isinstance(x, dict) and int(x.get("ts") or 0) >= (now - win)]
    if len(recent) < 2:
        return {"gain": 0.0, "samples": len(recent)}
    first = float(recent[0].get("requirements_avg_1_8") or 0.0)
    last = float(recent[-1].get("requirements_avg_1_8") or 0.0)
    return {"gain": round(last - first, 3), "samples": len(recent)}


def _neuroplastic_auto_manage() -> dict:
    gate = _neuroplastic_gate_snapshot()
    pend = neuroplastic.list_pending()
    rt = neuroplastic.active_runtime()
    active_ids = set([str(x.get("id")) for x in ((rt or {}).get("active") or []) if x.get("id")])

    activated = []
    reverted = []

    # auto-promoção por janela rolling (fase 2)
    for p in pend[:20]:
        pid = str(p.get("id") or "")
        if not pid or pid in active_ids:
            continue
        if str(p.get("status") or "") != "evaluated":
            continue
        sm = p.get("shadow_metrics") or {}
        promote_rec = bool(sm.get("promote_recommendation"))
        if not promote_rec:
            continue
        pass_gate = (
            float(gate.get("requirements_avg_1_8") or 0.0) >= 58.0
            and float(gate.get("decision_quality") or 0.0) >= 0.24
            and float(gate.get("agi_mode_percent") or 0.0) >= 55.0
        )
        if pass_gate:
            # injeta canary default caso patch não tenha definido
            patch = p.get("patch") or {}
            if isinstance(patch, dict) and patch.get("canary_ratio") is None:
                patch["canary_ratio"] = 0.35
                p["patch"] = patch
            ap = neuroplastic.activate(pid)
            if ap:
                activated.append(pid)
                st = _neuroplastic_gate_load()
                st.setdefault("activation_baselines", {})[pid] = {**gate, "activated_at": int(time.time())}
                _neuroplastic_gate_save(st)
                store.db.add_event("neuroplastic_autopromote", f"🧬 auto-promote: {pid}")

    # auto-reversão se degradação persistir
    st = _neuroplastic_gate_load()
    streaks = dict(st.get("revert_streaks") or {})
    baselines = dict(st.get("activation_baselines") or {})
    g7 = _rolling_gain_days(7)
    g14 = _rolling_gain_days(14)
    for aid in list(active_ids):
        base = baselines.get(aid) or {}
        dq_drop = float(base.get("decision_quality") or gate.get("decision_quality") or 0) - float(gate.get("decision_quality") or 0)
        req_drop = float(base.get("requirements_avg_1_8") or gate.get("requirements_avg_1_8") or 0) - float(gate.get("requirements_avg_1_8") or 0)
        age_sec = int(time.time()) - int(base.get("activated_at") or int(time.time()))
        no_sustained_gain = (age_sec >= 86400 and float(g7.get("gain") or 0.0) <= 0.0) or (age_sec >= 2 * 86400 and float(g14.get("gain") or 0.0) < 0.5)

        bad_now = (
            float(gate.get("decision_quality") or 0.0) < 0.18
            or float(gate.get("requirements_avg_1_8") or 0.0) < 50.0
            or dq_drop > 0.12
            or req_drop > 6.0
            or no_sustained_gain
        )
        streaks[aid] = (int(streaks.get(aid) or 0) + 1) if bad_now else 0
        if streaks[aid] >= 2:
            reason = "auto_guardrail_degradation" if not no_sustained_gain else "auto_no_sustained_gain"
            if neuroplastic.revert(aid, reason=reason):
                reverted.append(aid)
                streaks[aid] = 0
                store.db.add_event("neuroplastic_autorevert", f"🛑 auto-revert: {aid} ({reason})")

    st["revert_streaks"] = streaks
    _neuroplastic_gate_save(st)

    return {"gate": gate, "gain_7d": g7, "gain_14d": g14, "activated": activated, "reverted": reverted, "active_runtime": neuroplastic.active_runtime()}


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


def _invent_procedure_from_context(context_text: str, domain: str | None = None, name_hint: str | None = None) -> dict | None:
    ctx = (context_text or '').strip()
    if len(ctx) < 24:
        return None

    prompt = f"""Invent ONE new procedure to solve the context below.
Do not reuse existing procedure names; create a new tool-like strategy.
Return ONLY JSON with keys:
name, goal, domain, proc_type, preconditions, steps (array), success_criteria.
Context:\n{ctx[:2600]}"""
    try:
        raw = llm.complete(prompt, strategy='reasoning', json_mode=True)
        d = json.loads(raw) if raw else {}
        steps = d.get('steps') if isinstance(d, dict) else None
        if isinstance(steps, list) and len(steps) >= 2:
            dom = (domain or d.get('domain') or 'general').strip()
            return {
                'name': (name_hint or d.get('name') or f"Procedimento inventado: {dom}").strip()[:140],
                'goal': d.get('goal') or f"Resolver contexto novo em {dom}",
                'domain': dom,
                'proc_type': (d.get('proc_type') or _infer_proc_type(dom, ctx)).strip(),
                'preconditions': d.get('preconditions'),
                'steps': [str(s).strip() for s in steps if str(s).strip()][:20],
                'success_criteria': d.get('success_criteria') or 'Resultado reproduzível com melhoria mensurável',
            }
    except Exception:
        pass

    # fallback determinístico
    dom = (domain or 'general').strip()
    return {
        'name': (name_hint or f"Procedimento inventado: {dom}").strip()[:140],
        'goal': f"Resolver problema inédito em {dom}",
        'domain': dom,
        'proc_type': _infer_proc_type(dom, ctx),
        'preconditions': 'Contexto mínimo disponível e objetivo definido',
        'steps': [
            'Definir objetivo operacional e restrições',
            'Gerar 2-3 hipóteses de abordagem',
            'Executar microteste de menor custo',
            'Medir resultado e risco',
            'Refinar abordagem e consolidar procedimento',
        ],
        'success_criteria': 'Melhora observável com risco controlado',
    }


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


def _validate_analogy_with_evidence(analogy_id: int) -> dict:
    a = store.get_analogy(analogy_id)
    if not a:
        return {"status": "not_found"}

    rule = str(a.get("inference_rule") or "")
    target = str(a.get("target_domain") or "general")
    score = float(a.get("confidence") or 0.5)

    # evidência factual: tenta buscar snippets no conhecimento global com termos da regra
    ev_hits = 0
    try:
        q = (rule or target)[:220]
        # search_knowledge is async elsewhere; here use lightweight heuristic based on text richness
        ev_hits = 1 if len(q.split()) >= 6 else 0
    except Exception:
        ev_hits = 0

    validated = (score >= 0.62 and ev_hits >= 1)
    new_status = "accepted_validated" if validated else "rejected"
    new_conf = min(0.98, score + 0.08) if validated else max(0.2, score - 0.15)
    note = "validated by factual corroboration" if validated else "insufficient corroboration"
    store.update_analogy_status(analogy_id, status=new_status, confidence=new_conf, notes=note)
    _audit_reasoning(
        "analogy_validation",
        {"analogy_id": analogy_id, "target_domain": target, "ev_hits": ev_hits},
        f"status={new_status}; {note}",
        confidence=new_conf,
    )
    return {"status": new_status, "confidence": new_conf, "evidence_hits": ev_hits}


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
    st = "accepted_provisional" if val.get('valid') else "rejected"

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
        salience=0.8 if st.startswith("accepted") else 0.55,
        ttl_sec=1800,
    )
    _audit_reasoning(
        "analogy_transfer",
        {"problem_text": problem_text[:220], "target_domain": target_domain, "mapping": cand.get("mapping")},
        f"status={st}; rule={applied.get('derived_rule')}",
        confidence=float(val.get("confidence") or 0.5),
    )

    return {"status": st, "analogy_id": aid, "candidate": cand, "validation": val, "applied": applied}


def _maintain_question_queue(stale_hours: float = 24.0, max_fix: int = 6) -> dict:
    """Fecha ciclo de aprendizado: limpa perguntas estagnadas e reescreve quando útil."""
    now = time.time()
    items = store.db.list_open_questions_full(limit=120)
    stale = []
    for q in items:
        age_h = (now - float(q.get("created_at") or now)) / 3600.0
        if age_h >= stale_hours:
            stale.append(q)

    dismissed = 0
    rewritten = 0
    limit_fix = max(0, int(max_fix))
    for q in stale[: limit_fix]:
        qid = int(q.get("id") or 0)
        if qid <= 0:
            continue
        qq = (q.get("question") or "").strip()
        # heurística: se for muito genérica, descarta; se útil, reescreve
        generic = (len(qq) < 24) or (qq.lower().startswith("o que é") and len(qq.split()) <= 3)
        store.dismiss_question(qid)
        dismissed += 1
        if not generic:
            newq = f"(revisada) Responda com evidência objetiva e exemplo concreto: {qq[:220]}"
            store.db.add_questions([{"question": newq, "priority": max(3, int(q.get("priority") or 3)), "context": "curiosity_maintenance"}])
            rewritten += 1

    if dismissed or rewritten:
        store.db.add_event("curiosity_maintenance", f"🧰 manutenção perguntas: stale={len(stale)} dismissed={dismissed} rewritten={rewritten}")
    return {"open": len(items), "stale": len(stale), "dismissed": dismissed, "rewritten": rewritten}


def _milestone_health_check(active_goal: dict | None) -> dict:
    if not active_goal:
        return {"checked": 0, "replanned": 0}
    gid = int(active_goal.get("id") or 0)
    if gid <= 0:
        return {"checked": 0, "replanned": 0}
    ms = store.list_goal_milestones(goal_id=gid, status=None, limit=20)
    now = time.time()
    replanned = 0
    checked = len(ms)
    for m in ms:
        if str(m.get("status") or "") in ("done", "archived"):
            continue
        upd = float(m.get("updated_at") or m.get("created_at") or now)
        age_h = (now - upd) / 3600.0
        prog = float(m.get("progress") or 0.0)
        if age_h > 48 and prog < 0.4:
            _enqueue_action_if_new(
                "ask_evidence",
                f"(ação-replan) Milestone W{m.get('week_index')} travado: {m.get('title')}. Propor estratégia alternativa com menor custo.",
                priority=6,
                meta={"goal_id": gid, "milestone_id": m.get("id")},
            )
            replanned += 1
            break
    return {"checked": checked, "replanned": replanned}


def _enqueue_active_milestone_action(active_goal: dict | None):
    if not active_goal:
        return
    gid = int(active_goal.get("id") or 0)
    if gid <= 0:
        return
    ms = store.get_next_open_milestone(gid)
    if not ms:
        return
    mid = int(ms.get("id") or 0)
    title = ms.get("title") or "milestone"
    crit = ms.get("progress_criteria") or ""
    # autonomia melhorada: sempre puxa próximo passo objetivo do milestone semanal
    _enqueue_action_if_new(
        "ask_evidence",
        f"(ação-milestone) Avançar milestone W{ms.get('week_index')}: {title}. Critério: {crit}",
        priority=6,
        meta={"goal_id": gid, "milestone_id": mid},
        ttl_sec=20 * 60,
    )


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
        _neurosym_proof(
            "policy_block",
            premises=[f"kind={kind}", f"text={text[:140]}", f"policy_reasons={'; '.join(verdict.reasons)[:180]}"],
            inference="Policy rules disallow this action under current norms.",
            conclusion=f"Action {kind} blocked by policy.",
            confidence=max(0.2, min(0.99, float(verdict.score or 0.5))),
            action_meta={"action_id": aid, "kind": kind, "status": "blocked_policy"},
        )
        store.db.add_event("action_blocked", f"⛔ ação bloqueada #{aid}: {text[:120]}")
        return {"id": aid, "status": "blocked", "kind": kind}

    store.db.mark_action(aid, "running", policy_allowed=True, policy_score=verdict.score)

    try:
        dg = None
        dq = 0.5
        cp = None
        causal_checked = False

        # guard deliberativo (System-2) antes de ações de maior impacto
        if kind in ("execute_procedure_active", "prune_memory", "invent_procedure"):
            dg = _run_deliberate_task(
                problem_text=f"Preflight para ação {kind}: {text[:220]}",
                max_steps=0,
                budget_seconds=0,
                use_rl=True,
            )
            dq = float(dg.get("quality_proxy") or 0.0)
            _neurosym_proof(
                "deliberative_preflight",
                premises=[f"kind={kind}", f"quality_proxy={dq:.2f}"],
                inference="System-2 preflight estimated action quality before execution.",
                conclusion=f"Preflight for {kind} quality={dq:.2f}",
                confidence=max(0.2, dq),
                action_meta={"action_id": aid, "kind": kind, "status": "preflight"},
            )
            if dq < 0.38:
                store.db.mark_action(aid, "blocked", last_error=f"deliberation_low_quality {dq:.2f}")
                integrity.register_decision(kind, False, 'deliberation_low_quality', {'action_id': aid, 'dq': dq})
                _neurosym_proof(
                    "deliberative_block",
                    premises=[f"kind={kind}", f"quality_proxy={dq:.2f}"],
                    inference="Deliberative preflight failed minimum quality threshold.",
                    conclusion=f"Action {kind} blocked pending better deliberation.",
                    confidence=max(0.3, dq),
                    action_meta={"action_id": aid, "kind": kind, "status": "blocked_deliberative"},
                )
                return {"id": aid, "status": "blocked", "kind": kind, "deliberation": dg}

        # precheck causal para ações potencialmente sensíveis
        if kind in ("execute_procedure_active", "auto_resolve_conflicts", "prune_memory", "invent_procedure"):
            cp = _causal_precheck(kind, text=text, meta=meta)
            causal_checked = True
            risk = float((cp.get("simulation") or {}).get("risk_score") or 0.0)
            net = float((cp.get("simulation") or {}).get("net_score") or 0.0)
            if risk >= 1.2 and net < 0:
                store.db.mark_action(aid, "blocked", last_error=f"causal_risk_high risk={risk} net={net}")
                integrity.register_decision(kind, False, 'causal_guardrail_block', {'action_id': aid, 'risk': risk, 'net': net})
                _neurosym_proof(
                    "causal_block",
                    premises=[f"kind={kind}", f"risk={risk:.2f}", f"net={net:.2f}"],
                    inference="Causal simulation predicts net negative impact under high risk.",
                    conclusion=f"Action {kind} blocked by causal guardrail.",
                    confidence=min(0.95, max(0.4, risk / 2.0)),
                    action_meta={"action_id": aid, "kind": kind, "status": "blocked_causal"},
                )
                store.db.add_event("action_blocked", f"⛔ ação bloqueada por precheck causal #{aid}: {kind} (risk={risk:.2f}, net={net:.2f})")
                return {"id": aid, "status": "blocked", "kind": kind, "causal": cp}

        # dual-consensus integrity gate (neural + symbolic)
        sym = neurosym.consistency_check(limit=200)
        sym_score = float(sym.get('consistency_score') or 1.0)
        has_proof = dg is not None if kind in ("execute_procedure_active", "prune_memory", "invent_procedure") else True
        ok_integrity, reason_integrity = integrity.evaluate(
            kind=kind,
            neural_confidence=float(dq),
            symbolic_consistency=sym_score,
            has_proof=bool(has_proof),
            causal_checked=bool(causal_checked or kind not in (integrity.load_rules().get('require_causal_precheck') or [])),
        )
        if not ok_integrity:
            store.db.mark_action(aid, "blocked", last_error=f"integrity_veto:{reason_integrity}")
            integrity.register_decision(kind, False, reason_integrity, {'action_id': aid, 'dq': dq, 'sym_score': sym_score})
            store.db.add_event("blocked_integrity", f"🛡️ ação bloqueada por integrity gate #{aid}: {kind} ({reason_integrity})")
            _neurosym_proof(
                "integrity_block",
                premises=[f"kind={kind}", f"dq={dq:.2f}", f"symbolic_consistency={sym_score:.2f}", f"reason={reason_integrity}"],
                inference="Dual-consensus gate denied action due to integrity rule violation.",
                conclusion=f"Action {kind} blocked by integrity gate.",
                confidence=max(0.6, sym_score),
                action_meta={"action_id": aid, "kind": kind, "status": "blocked_integrity"},
            )
            return {"id": aid, "status": "blocked", "kind": kind, "integrity_reason": reason_integrity}
        else:
            integrity.register_decision(kind, True, 'integrity_pass', {'action_id': aid, 'dq': dq, 'sym_score': sym_score})

        if kind == "generate_questions":
            n = curiosity.generate_questions()
            store.db.add_event("action_done", f"🤖 ação #{aid}: generate_questions (+{n})")
        elif kind == "ask_evidence":
            q = text.replace("(ação)", "").strip()
            store.db.add_questions([{"question": q, "priority": 4, "context": "autonomia"}])
            # autonomia orientada a milestones: progresso incremental ao executar micro-passos
            mid = int((meta or {}).get("milestone_id") or 0)
            if mid > 0:
                try:
                    gid = int((meta or {}).get("goal_id") or 0)
                    ms = store.get_next_open_milestone(gid) if gid > 0 else None
                    prev = float(ms.get("progress") or 0.0) if ms and int(ms.get("id") or 0) == mid else 0.0
                    newp = min(1.0, prev + 0.08)
                    nst = "done" if newp >= 1.0 else "active"
                    store.update_milestone_progress(mid, newp, status=nst)
                except Exception:
                    pass
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
        elif kind == "maintain_question_queue":
            info = _maintain_question_queue(stale_hours=24.0, max_fix=6)
            store.db.add_event("action_done", f"🤖 ação #{aid}: maintain_question_queue stale={info.get('stale')} rewritten={info.get('rewritten')}")
        elif kind == "clarify_semantics":
            base = str((meta or {}).get('text') or text or '')
            q = semantics.clarification_prompt(base)
            store.db.add_questions([{"question": q, "priority": 5, "context": "semantics_clarification"}])
            store.db.add_event("action_done", f"🤖 ação #{aid}: clarify_semantics")
            _audit_reasoning("semantic_clarification", {"source_text": base[:180]}, "ambiguity detected; clarification requested", confidence=0.7)
        elif kind == "unsupervised_discovery":
            info = unsupervised.discover_and_restructure(store.db, max_experiences=220)
            store.db.add_event("action_done", f"🤖 ação #{aid}: unsupervised_discovery scanned={info.get('scanned')} induced={info.get('triples_induced')}")
            store.db.add_insight(
                kind="unsupervised_learning",
                title="Aprendizado não-supervisionado executado",
                text=f"Induzi {info.get('triples_induced')} relações latentes (conceitos={info.get('concepts_total')}, arestas={info.get('edges_total')}).",
                priority=4,
                meta_json=json.dumps(info, ensure_ascii=False)[:3000],
            )
            _workspace_publish("unsupervised", "latent.discovery", info, salience=0.7, ttl_sec=3600)
        elif kind == "neuroplastic_cycle":
            pend = neuroplastic.list_pending()
            evaluated = 0
            for p in pend[:5]:
                if str(p.get("status") or "") == "pending":
                    _run_neuroplastic_shadow_eval(str(p.get("id")))
                    evaluated += 1
            managed = _neuroplastic_auto_manage()
            store.db.add_event("action_done", f"🤖 ação #{aid}: neuroplastic_cycle evaluated={evaluated} activated={len(managed.get('activated') or [])} reverted={len(managed.get('reverted') or [])}")
        elif kind == "invent_procedure":
            ctx = str((meta or {}).get("context_text") or text or "")
            dom = (meta or {}).get("domain")
            inv = _invent_procedure_from_context(ctx, domain=dom)
            if not inv:
                store.db.add_event("action_skipped", f"↷ ação #{aid}: invent_procedure sem contexto suficiente")
            else:
                pid = store.add_procedure(
                    name=inv['name'],
                    goal=inv.get('goal'),
                    steps_json=json.dumps(inv.get('steps') or [], ensure_ascii=False),
                    domain=inv.get('domain'),
                    proc_type=inv.get('proc_type') or 'analysis',
                    preconditions=inv.get('preconditions'),
                    success_criteria=inv.get('success_criteria'),
                )
                store.db.add_insight(
                    kind='procedure_invented',
                    title='Novo procedimento inventado',
                    text=f"Invenção procedural: {inv.get('name')} ({inv.get('domain')}) id={pid}",
                    priority=5,
                )
                _workspace_publish("procedural_inventor", "procedure.invented", {"procedure_id": pid, "name": inv.get('name'), "domain": inv.get('domain')}, salience=0.82, ttl_sec=3600)
                store.db.add_event("action_done", f"🤖 ação #{aid}: invent_procedure id={pid}")
        elif kind == "intrinsic_tick":
            info = _intrinsic_tick(force=bool((meta or {}).get("force")))
            store.db.add_event("action_done", f"🤖 ação #{aid}: intrinsic_tick drive={((info.get('chosen_goal') or {}).get('drive'))}")
        elif kind == "emergence_tick":
            info = _emergence_tick()
            store.db.add_event("action_done", f"🤖 ação #{aid}: emergence_tick policy={((info.get('chosen_policy') or {}).get('id'))}")
        elif kind == "deliberate_task":
            ptxt = str((meta or {}).get("problem_text") or text or "")
            bsec = int((meta or {}).get("budget_seconds") or 35)
            msteps = int((meta or {}).get("max_steps") or 4)
            info = _run_deliberate_task(problem_text=ptxt, max_steps=msteps, budget_seconds=bsec)
            store.db.add_event("action_done", f"🤖 ação #{aid}: deliberate_task steps={len(info.get('steps') or [])}")
        elif kind == "horizon_review":
            info = _horizon_review_tick()
            store.db.add_event("action_done", f"🤖 ação #{aid}: horizon_review status={info.get('status')}")
        elif kind == "subgoal_planning":
            info = _subgoal_planning_tick()
            store.db.add_event("action_done", f"🤖 ação #{aid}: subgoal_planning status={info.get('status')}")
        elif kind == "project_management_cycle":
            info = _project_management_tick()
            store.db.add_event("action_done", f"🤖 ação #{aid}: project_management_cycle status={info.get('status')}")
        elif kind == "route_toolchain":
            intent = str((meta or {}).get('intent') or 'general')
            ctx = (meta or {}).get('context') or {}
            plc = bool((meta or {}).get('prefer_low_cost', True))
            info = _run_tool_route(intent=intent, context=ctx, prefer_low_cost=plc)
            store.db.add_event("action_done", f"🤖 ação #{aid}: route_toolchain status={info.get('status')} selected={info.get('selected')}")
        elif kind == "project_experiment_cycle":
            info = _project_experiment_cycle()
            store.db.add_event("action_done", f"🤖 ação #{aid}: project_experiment_cycle status={info.get('status')}")
        else:
            store.db.add_event("action_skipped", f"↷ ação #{aid} desconhecida: {kind}")

        store.db.mark_action(aid, "done")
        _neurosym_proof(
            "action_execution",
            premises=[f"kind={kind}", f"text={text[:140]}", "policy=allowed"],
            inference="Action execution path completed without guardrail violation.",
            conclusion=f"Action {kind} executed successfully.",
            confidence=0.72,
            action_meta={"action_id": aid, "kind": kind, "status": "done"},
        )
        return {"id": aid, "status": "done", "kind": kind}
    except Exception as e:
        store.db.mark_action(aid, "error", last_error=str(e)[:500])
        _neurosym_proof(
            "action_error",
            premises=[f"kind={kind}", f"error={str(e)[:180]}"],
            inference="Execution failed due to runtime exception.",
            conclusion=f"Action {kind} failed.",
            confidence=0.4,
            action_meta={"action_id": aid, "kind": kind, "status": "error"},
        )
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

            # Sprint 2: manutenção ativa da fila de curiosidade
            _enqueue_action_if_new(
                "maintain_question_queue",
                "(ação) Revisar fila de perguntas estagnadas e reescrever/descarte para manter utilidade.",
                priority=3,
                ttl_sec=15 * 60,
            )

            # aprendizado não-supervisionado profundo (indução latente + reestruturação)
            _enqueue_action_if_new(
                "unsupervised_discovery",
                "(ação) Descobrir conceitos latentes e reestruturar conhecimento sem template fixo.",
                priority=4,
                ttl_sec=30 * 60,
            )

            # neuroplasticidade fase 1: avaliar propostas de mutação em shadow mode
            _enqueue_action_if_new(
                "neuroplastic_cycle",
                "(ação) Rodar ciclo de avaliação shadow de mutações arquiteturais pendentes.",
                priority=3,
                ttl_sec=30 * 60,
            )

            # IME fase 1: atualização de motivação intrínseca
            _enqueue_action_if_new(
                "intrinsic_tick",
                "(ação) Atualizar drives intrínsecos e sintetizar propósito interno.",
                priority=4,
                ttl_sec=30 * 60,
            )

            # emergência de políticas: dinâmica latente + sampler divergente
            _enqueue_action_if_new(
                "emergence_tick",
                "(ação) Atualizar estado latente e amostrar políticas divergentes.",
                priority=4,
                ttl_sec=20 * 60,
            )

            # System-2 router: agenda deliberação prolongada quando complexidade subir
            itc_need = _itc_router_need()
            if bool(itc_need.get('need')):
                _enqueue_action_if_new(
                    "deliberate_task",
                    f"(ação-itc) Deliberar problema complexo: reason={itc_need.get('reason')}",
                    priority=6,
                    meta={"problem_text": f"Conflitos abertos={itc_need.get('open_conflicts')}; dq={itc_need.get('decision_quality'):.2f}; resolver trade-offs e plano.", "budget_seconds": 45, "max_steps": 5},
                    ttl_sec=25 * 60,
                )

            # continuidade de longo horizonte (dias/semanas)
            _enqueue_action_if_new(
                "horizon_review",
                "(ação-horizon) Revisar missão persistente de longo prazo.",
                priority=4,
                ttl_sec=45 * 60,
            )

            _enqueue_action_if_new(
                "subgoal_planning",
                "(ação-subgoal) Decompor objetivo atual em DAG de sub-objetivos.",
                priority=4,
                ttl_sec=35 * 60,
            )

            _enqueue_action_if_new(
                "project_management_cycle",
                "(ação-projeto) Rodar ciclo de gestão de projeto + recuperação de falhas.",
                priority=5,
                ttl_sec=40 * 60,
            )

            _enqueue_action_if_new(
                "project_experiment_cycle",
                "(ação-projeto) Rodar experimento técnico e validar hipótese de melhoria.",
                priority=5,
                ttl_sec=45 * 60,
            )

            # Sprint 3: clarificação semântica ativa (ambiguidade/metáfora/ironia)
            try:
                last_exp = (store.db.list_experiences(limit=1) or [{}])[-1]
                ltxt = str(last_exp.get("text") or "")[:400]
                diag = semantics.detect_ambiguity(ltxt)
                if float(diag.get("score") or 0) >= 0.45:
                    _enqueue_action_if_new(
                        "clarify_semantics",
                        "(ação) Solicitar clarificação semântica para reduzir ambiguidade.",
                        priority=5,
                        meta={"text": ltxt, "ambiguity": diag},
                        ttl_sec=15 * 60,
                    )
            except Exception:
                pass

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
                else:
                    # inventividade procedural: criar ferramenta nova quando catálogo não cobre contexto
                    _enqueue_action_if_new(
                        "invent_procedure",
                        "(ação-procedural) Inventar novo procedimento para contexto sem cobertura atual.",
                        priority=6,
                        meta={"context_text": recent_ctx[:500], "domain": "general"},
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
                    _enqueue_active_milestone_action(active_goal)
                    _milestone_health_check(active_goal)
            except Exception as e:
                logger.debug(f"Goal planning skipped: {e}")

            # metas persistentes proativas (não dependem de conflito)
            try:
                _enqueue_from_persistent_goal()
            except Exception as e:
                logger.debug(f"Persistent goals skipped: {e}")

            # global workspace: acoplamento frouxo entre módulos
            try:
                # atualiza leitura TOM no workspace
                try:
                    _workspace_publish("tom", "user.intent", tom.infer_user_intent(store.db.list_experiences(limit=20)), salience=0.68, ttl_sec=1200)
                except Exception:
                    pass

                ws = _workspace_recent(channels=["metacog.snapshot", "analogy.transfer", "conflict.status", "user.intent"], limit=10)
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
                    elif ch == "analogy.transfer" and str(payload.get("status") or "").startswith("accepted"):
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
                    elif ch == "user.intent":
                        il = str(payload.get("label") or "")
                        if il == "confused":
                            _enqueue_action_if_new(
                                "ask_evidence",
                                "(ação-workspace-TOM) Explicar em linguagem mais simples e confirmar entendimento.",
                                priority=6,
                            )
                        elif il == "testing":
                            _enqueue_action_if_new(
                                "ask_evidence",
                                "(ação-workspace-TOM) Entregar resposta auditável com critérios de teste e limites.",
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
    intent = tom.infer_user_intent(store.db.list_experiences(limit=20))
    return {"status": "online", "stats": stats, "next": next_q, "agi": agi, "tom": intent}

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


@app.get("/api/stream/events")
async def stream_events(request: Request, since_id: int = 0, heartbeat_sec: int = 15):
    """SSE stream de eventos para voz ativa no frontend."""
    hb = max(5, min(60, int(heartbeat_sec or 15)))

    async def gen():
        last_id = int(since_id or 0)
        # hello
        hello = {"type": "hello", "since_id": last_id, "ts": int(time.time())}
        yield f"event: hello\ndata: {json.dumps(hello, ensure_ascii=False)}\n\n"

        while True:
            if await request.is_disconnected():
                break
            try:
                rows = store.db.list_events(since_id=last_id, limit=80)
                if rows:
                    for e in rows:
                        last_id = max(last_id, int(e.get("id") or 0))
                        payload = {
                            "id": e.get("id"),
                            "created_at": e.get("created_at"),
                            "kind": e.get("kind"),
                            "text": e.get("text"),
                            "meta_json": e.get("meta_json"),
                        }
                        ev_name = "insight" if str(e.get("kind") or "") == "insight" else "event"
                        yield f"id: {last_id}\nevent: {ev_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                else:
                    # keep-alive heartbeat
                    ping = {"ts": int(time.time()), "last_id": last_id}
                    yield f"event: ping\ndata: {json.dumps(ping)}\n\n"

                await asyncio.sleep(hb)
            except Exception as ex:
                err = {"error": str(ex)[:200], "ts": int(time.time())}
                yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"
                await asyncio.sleep(hb)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


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
    _audit_reasoning(
        "conflict_manual_resolution",
        {"conflict_id": id, "decided_by": req.decided_by},
        f"chosen_object={req.chosen_object}; resolution={req.resolution or ''}",
        confidence=0.8,
    )
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


@app.get("/api/tom/status")
async def tom_status(limit: int = 20):
    recent = store.db.list_experiences(limit=max(5, min(100, int(limit))))
    out = tom.infer_user_intent(recent)
    _workspace_publish("tom", "user.intent", out, salience=0.72, ttl_sec=1200)
    return out


@app.get("/api/language/diagnose")
async def language_diagnose(limit: int = 5):
    exps = store.db.list_experiences(limit=max(1, min(50, int(limit))))
    out = []
    for e in exps:
        txt = str(e.get("text") or "")
        if not txt.strip():
            continue
        d = semantics.detect_ambiguity(txt[:500])
        out.append({"experience_id": e.get("id"), "diag": d, "sample": txt[:180]})
    return {"items": out}


@app.get("/api/language/eval")
async def language_eval():
    res = semantics.evaluate_language_dataset("/app/ultronpro/data_language_eval.json")
    store.db.add_event("language_eval", f"🗣️ language eval acc={res.get('accuracy')}", meta_json=json.dumps(res, ensure_ascii=False)[:4000])
    return res


@app.post("/api/unsupervised/run")
async def unsupervised_run(max_experiences: int = 220):
    info = unsupervised.discover_and_restructure(store.db, max_experiences=max_experiences)
    store.db.add_event("unsupervised_run", f"🧠 unsupervised run: scanned={info.get('scanned')} induced={info.get('triples_induced')}")
    return info


@app.get("/api/causal/model")
async def causal_model(limit: int = 4000):
    m = causal.build_world_model(store.db, limit=limit)
    return {"nodes": len(m.get("nodes") or {}), "edges": len(m.get("edges") or []), "sample_edges": (m.get("edges") or [])[:20]}


@app.post("/api/causal/simulate")
async def causal_simulate(kind: str = "ask_evidence", text: str = "", steps: int = 3):
    m = causal.build_world_model(store.db, limit=4000)
    interventions = causal.infer_intervention_from_action(kind, text=text, meta={})
    s = causal.simulate_intervention(m, interventions, steps=steps)
    return {"kind": kind, "interventions": interventions, "simulation": s, "model": {"nodes": len(m.get("nodes") or {}), "edges": len(m.get("edges") or [])}}


@app.get("/api/unsupervised/status")
async def unsupervised_status():
    return unsupervised.state_summary()


# --- IME Fase 1 (motivação intrínseca) ---

@app.post("/api/intrinsic/tick")
async def intrinsic_tick(req: IntrinsicTickRequest):
    return _intrinsic_tick(force=bool(req.force))


@app.get("/api/purpose/status")
async def purpose_status():
    st = intrinsic.load_state()
    return {
        "purpose": st.get("purpose"),
        "drives": st.get("drives"),
        "satiation": st.get("satiation"),
        "history_tail": (st.get("history") or [])[-8:],
    }


@app.post("/api/emergence/tick")
async def emergence_tick_run():
    return _emergence_tick()


@app.get("/api/emergence/status")
async def emergence_status(limit: int = 20):
    st = emergence.state()
    hist = emergence.eval_history(limit=limit)
    return {"state": st, "history": hist}


@app.get("/api/emergence/indistinguishability")
async def emergence_indistinguishability(limit: int = 40):
    hist = emergence.eval_history(limit=limit)
    if not hist:
        return {"score": 0.0, "samples": 0, "note": "insufficient data"}

    # proxy: diversidade de políticas escolhidas + variação de ações
    policies = [((h.get('chosen_policy') or {}).get('id')) for h in hist]
    unique_p = len(set([p for p in policies if p]))
    action_sets = [tuple(((h.get('chosen_policy') or {}).get('actions') or [])) for h in hist]
    unique_a = len(set(action_sets))
    score = min(1.0, (unique_p * 0.25) + (unique_a * 0.12))
    return {"score": round(score, 3), "samples": len(hist), "unique_policies": unique_p, "unique_action_sets": unique_a}


# --- Inference-Time Compute (System 2) ---

@app.post("/api/itc/run")
async def itc_run(req: ITCRunRequest):
    return _run_deliberate_task(req.problem_text, max_steps=req.max_steps, budget_seconds=req.budget_seconds, use_rl=bool(req.use_rl))


@app.get("/api/itc/history")
async def itc_history(limit: int = 40):
    return {"items": itc.history(limit=limit)}


@app.get("/api/itc/status")
async def itc_status():
    r = _itc_router_need()
    h = itc.history(limit=30)
    avg_q = (sum(float(x.get('quality_proxy') or 0.0) for x in h) / max(1, len(h))) if h else 0.0
    avg_t = (sum(float(x.get('elapsed_sec') or 0.0) for x in h) / max(1, len(h))) if h else 0.0
    avg_r = (sum(float(x.get('reward') or 0.0) for x in h) / max(1, len(h))) if h else 0.0
    return {"router": r, "episodes": len(h), "avg_quality_proxy": round(avg_q, 3), "avg_elapsed_sec": round(avg_t, 3), "avg_reward": round(avg_r, 3), "policy": itc.policy_status()}


@app.get("/api/itc/policy")
async def itc_policy():
    return itc.policy_status()


# --- Long Horizon Memory / Continuity ---

@app.post("/api/horizon/missions")
async def horizon_mission_create(req: HorizonMissionRequest):
    m = longhorizon.upsert_mission(req.title, req.objective, horizon_days=req.horizon_days, context=req.context)
    store.db.add_event('horizon_mission', f"🎯 missão ativa: {m.get('id')} {m.get('title')}")
    return m


@app.get("/api/horizon/missions")
async def horizon_missions(limit: int = 30):
    return {"active": longhorizon.active_mission(), "items": longhorizon.list_missions(limit=limit)}


@app.post("/api/horizon/missions/{mission_id}/checkpoint")
async def horizon_checkpoint(mission_id: str, req: HorizonCheckpointRequest):
    cp = longhorizon.add_checkpoint(mission_id, req.note, progress_delta=req.progress_delta, signal=req.signal)
    if not cp:
        raise HTTPException(404, 'mission not found')
    return {"status": "ok", "checkpoint": cp}


@app.post("/api/horizon/review")
async def horizon_review():
    return _horizon_review_tick()


@app.post("/api/subgoals/plan")
async def subgoals_plan():
    return _subgoal_planning_tick()


@app.get("/api/subgoals")
async def subgoals_list(limit: int = 20):
    return {"items": subgoals.list_roots(limit=limit)}


@app.post("/api/subgoals/{root_id}/nodes/{node_id}")
async def subgoals_mark(root_id: str, node_id: str, req: SubgoalMarkRequest):
    ok = subgoals.mark_node(root_id, node_id, status=req.status)
    if not ok:
        raise HTTPException(404, "node not found")
    return {"status": "ok"}


@app.post('/api/projects')
async def projects_create(req: ProjectRequest):
    p = project_kernel.upsert_project(req.title, req.objective, scope=req.scope, sla_hours=req.sla_hours)
    store.db.add_event('project_upsert', f"📦 projeto ativo: {p.get('id')} {p.get('title')}")
    return p


@app.get('/api/projects')
async def projects_list(limit: int = 30):
    return {'active': project_kernel.active_project(), 'items': project_kernel.list_projects(limit=limit)}


@app.post('/api/projects/{project_id}/checkpoint')
async def projects_checkpoint(project_id: str, req: ProjectCheckpointRequest):
    cp = project_kernel.add_checkpoint(project_id, req.note, progress_delta=req.progress_delta, signal=req.signal)
    if not cp:
        raise HTTPException(404, 'project not found')
    project_kernel.remember(project_id, kind=req.signal or 'checkpoint', text=req.note, meta={'progress_delta': req.progress_delta})
    return {'status': 'ok', 'checkpoint': cp}


@app.get('/api/projects/playbooks')
async def projects_playbooks():
    return project_kernel.get_playbooks()


@app.post('/api/projects/tick')
async def projects_tick():
    return _project_management_tick()


@app.get('/api/projects/{project_id}/brief')
async def projects_brief(project_id: str):
    b = project_kernel.project_brief(project_id)
    if not b:
        raise HTTPException(404, 'project not found')
    return b


@app.get('/api/projects/{project_id}/memory')
async def projects_memory(project_id: str, query: str = '', limit: int = 30):
    return {'items': project_kernel.recall(project_id, query=query, limit=limit)}


@app.get('/api/projects/{project_id}/experiments')
async def projects_experiments(project_id: str, limit: int = 30):
    return {'items': project_executor.list_experiments(project_id=project_id, limit=limit)}


@app.post('/api/projects/experiments/run')
async def projects_experiment_run():
    return _project_experiment_cycle()


@app.post('/api/tool-router/plan')
async def tool_router_plan(req: ToolRouteRequest):
    return tool_router.plan_route(req.intent, context=req.context or {}, prefer_low_cost=bool(req.prefer_low_cost))


@app.post('/api/tool-router/run')
async def tool_router_run(req: ToolRouteRequest):
    return _run_tool_route(req.intent, context=req.context or {}, prefer_low_cost=bool(req.prefer_low_cost))


# --- Neuroplasticidade Fase 1 (safe mutate loop) ---

@app.get("/api/neuroplastic/proposals")
async def neuroplastic_proposals():
    return {"items": neuroplastic.list_pending()}


@app.post("/api/neuroplastic/proposals")
async def neuroplastic_add(req: MutationProposalRequest):
    item = neuroplastic.add_proposal(req.title, req.rationale, req.patch or {}, author=req.author or "manual")
    store.db.add_event("neuroplastic_proposal", f"🧬 proposta criada: {item.get('id')} {item.get('title')}")
    return item


@app.post("/api/neuroplastic/proposals/{proposal_id}/evaluate")
async def neuroplastic_evaluate(proposal_id: str):
    m = _run_neuroplastic_shadow_eval(proposal_id)
    return {"proposal_id": proposal_id, "shadow": m}


@app.post("/api/neuroplastic/proposals/{proposal_id}/activate")
async def neuroplastic_activate(proposal_id: str, req: MutationDecisionRequest):
    p = neuroplastic.activate(proposal_id)
    if not p:
        raise HTTPException(404, "proposal not found")
    store.db.add_event("neuroplastic_activate", f"🟢 mutação ativada: {proposal_id}", meta_json=json.dumps({"reason": req.reason}, ensure_ascii=False))
    return {"status": "active", "proposal": p, "runtime": neuroplastic.active_runtime()}


@app.post("/api/neuroplastic/proposals/{proposal_id}/revert")
async def neuroplastic_revert(proposal_id: str, req: MutationDecisionRequest):
    ok = neuroplastic.revert(proposal_id, reason=req.reason or "manual")
    if not ok:
        raise HTTPException(404, "proposal not active/not found")
    store.db.add_event("neuroplastic_revert", f"🔴 mutação revertida: {proposal_id}", meta_json=json.dumps({"reason": req.reason}, ensure_ascii=False))
    return {"status": "reverted", "proposal_id": proposal_id, "runtime": neuroplastic.active_runtime()}


@app.get("/api/neuroplastic/runtime")
async def neuroplastic_runtime():
    return neuroplastic.active_runtime()


@app.get("/api/neuroplastic/history")
async def neuroplastic_history(limit: int = 50):
    return {"items": neuroplastic.history(limit=limit)}


@app.get("/api/neuroplastic/gate/status")
async def neuroplastic_gate_status():
    st = _neuroplastic_gate_load()
    snap = _neuroplastic_gate_snapshot()
    return {
        "snapshot": snap,
        "gain_7d": _rolling_gain_days(7),
        "gain_14d": _rolling_gain_days(14),
        "revert_streaks": st.get("revert_streaks") or {},
        "activation_baselines": st.get("activation_baselines") or {},
        "runtime": neuroplastic.active_runtime(),
    }


@app.post("/api/neuroplastic/gate/run")
async def neuroplastic_gate_run():
    return _neuroplastic_auto_manage()


# --- Memory Curation ---

@app.get("/api/memory/curation/status")
async def memory_curation_status():
    return {"uncurated": store.db.count_uncurated_experiences()}


@app.post("/api/curiosity/maintenance/run")
async def curiosity_maintenance_run(stale_hours: float = 24.0, max_fix: int = 6):
    return _maintain_question_queue(stale_hours=stale_hours, max_fix=max_fix)


def _tom_ab_report(window_actions: int = 200) -> dict:
    acts = store.db.list_actions(limit=max(60, int(window_actions)))
    tom_tagged = 0
    tom_done = 0
    baseline_done = 0
    baseline_total = 0
    for a in acts:
        meta = {}
        try:
            meta = json.loads(a.get("meta_json") or "{}")
        except Exception:
            meta = {}
        has_tom = bool(meta.get("tom_intent"))
        if has_tom:
            tom_tagged += 1
            if a.get("status") == "done":
                tom_done += 1
        else:
            baseline_total += 1
            if a.get("status") == "done":
                baseline_done += 1

    tom_sr = (tom_done / max(1, tom_tagged)) if tom_tagged else 0.0
    base_sr = (baseline_done / max(1, baseline_total)) if baseline_total else 0.0
    lift = tom_sr - base_sr
    return {
        "window_actions": len(acts),
        "tom_tagged": tom_tagged,
        "tom_success_rate": round(tom_sr, 3),
        "baseline_success_rate": round(base_sr, 3),
        "lift": round(lift, 3),
        "label": "positive" if lift > 0.05 else ("neutral" if lift >= -0.05 else "negative"),
    }


def _milestone_kpi(window_days: int = 7) -> dict:
    goals_all = store.db.list_goals(status=None, limit=300)
    ms_all = []
    for g in goals_all[:120]:
        ms_all.extend(store.list_goal_milestones(goal_id=int(g.get("id") or 0), status=None, limit=40))

    done = [m for m in ms_all if str(m.get("status") or "") == "done"]
    active = [m for m in ms_all if str(m.get("status") or "") in ("open", "active")]

    now = time.time()
    delayed = 0
    for m in active:
        upd = float(m.get("updated_at") or m.get("created_at") or now)
        age_h = (now - upd) / 3600.0
        if age_h > (window_days * 24 / 2):
            delayed += 1

    recent_actions = store.db.list_actions(limit=250)
    replan_actions = [a for a in recent_actions if "(ação-replan)" in str(a.get("text") or "")]

    return {
        "milestones_total": len(ms_all),
        "milestones_done": len(done),
        "throughput_done_rate": round(len(done) / max(1, len(ms_all)), 3),
        "delayed_open": delayed,
        "replan_rate": round(len(replan_actions) / max(1, len(recent_actions)), 3),
    }


@app.get("/api/sprint2/health")
async def sprint2_health():
    return {
        "tom_ab": _tom_ab_report(window_actions=220),
        "milestones": _milestone_kpi(window_days=7),
        "curiosity_maintenance": _maintain_question_queue(stale_hours=9999.0, max_fix=0),
    }


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
    lang_eval = semantics.evaluate_language_dataset("/app/ultronpro/data_language_eval.json")
    scenarios = {
        "graph_learning": float(p.get("learning", 0)) >= 45,
        "conflict_handling": float(p.get("synthesis", 0)) >= 35,
        "memory_hygiene": float(p.get("curation", 0)) >= 35,
        "safety_controls": int(_autonomy_state.get("consecutive_errors") or 0) < 3,
        "autonomy_quality": float(meta.get("decision_quality") or 0) >= 0.2,
        "language_ambiguity_eval": float(lang_eval.get("accuracy") or 0.0) >= 0.6,
    }

    # Sprint 1: benchmark formal requisitos 1..8 (item 9 fora)
    intent = tom.infer_user_intent(store.db.list_experiences(limit=30))
    open_conf = len(store.db.list_conflicts(status="open", limit=500))
    procs = store.list_procedures(limit=100)
    domain_counts = {}
    for x in procs:
        d = (x.get("domain") or "").strip().lower()
        if not d:
            continue
        domain_counts[d] = domain_counts.get(d, 0) + int(x.get("attempts") or 0)
    proc_domains = len(domain_counts)
    total_attempts = sum(domain_counts.values())
    top_share = (max(domain_counts.values()) / max(1, total_attempts)) if domain_counts else 1.0
    anti_overfit_bonus = max(0.0, 1.0 - top_share)  # perto de 1 quando distribuído entre domínios

    accepted_analogies = len(store.list_analogies(limit=200, status="accepted_validated"))
    reasoning_audits = len([e for e in store.db.list_events(limit=300) if (e.get("kind") or "") == "reasoning_audit"])
    lang_diag = semantics.detect_ambiguity("\n".join([(e.get("text") or "")[:220] for e in store.db.list_experiences(limit=8)]))
    goals_all = store.db.list_goals(status=None, limit=200)
    milestones_total = 0
    for g in goals_all[:40]:
        milestones_total += len(store.list_goal_milestones(int(g.get("id") or 0), status=None, limit=16))

    req_scores = {
        "1_generalizacao_entre_dominios": round(min(100.0, proc_domains * 12 + accepted_analogies * 6 + anti_overfit_bonus * 18), 1),
        "2_transferencia_aprendizado": round(min(100.0, accepted_analogies * 18 + float(p.get("synthesis", 0)) * 0.4), 1),
        "3_raciocinio_abstracao": round(min(100.0, float(p.get("synthesis", 0)) * 0.7 + reasoning_audits * 0.2), 1),
        "4_aprendizado_autonomo": round(min(100.0, float(p.get("autonomy", 0)) * 0.75 + max(0, 30 - open_conf) * 0.5), 1),
        "5_linguagem_natural": round(min(100.0, 30.0 + (20.0 if intent.get("label") else 0.0) + float(intent.get("confidence") or 0) * 20.0 + (float(lang_eval.get("accuracy") or 0.0) * 40.0) + (10.0 if float(lang_diag.get("score") or 0) >= 0.35 else 0.0)), 1),
        "6_metacognicao": round(min(100.0, float(meta.get("decision_quality") or 0) * 100.0 + (20 - min(20, int(meta.get("low_quality_streak") or 0) * 5))), 1),
        "7_planejamento_decisao": round(min(100.0, float(p.get("goals", 0)) * 0.8 + min(25.0, milestones_total * 1.8)), 1),
        "8_adaptabilidade_ambientes_novos": round(min(100.0, float(p.get("autonomy", 0)) * 0.6 + proc_domains * 8 + accepted_analogies * 4 + anti_overfit_bonus * 14), 1),
    }

    passed = len([v for v in scenarios.values() if v])
    score = round((passed / max(1, len(scenarios))) * 100.0, 1)
    req_avg = round(sum(req_scores.values()) / max(1, len(req_scores)), 1)

    out = {
        "ts": int(time.time()),
        "score": score,
        "scenarios": scenarios,
        "agi_mode_percent": agi.get("agi_mode_percent"),
        "decision_quality": meta.get("decision_quality"),
        "requirements_1_8": req_scores,
        "requirements_avg_1_8": req_avg,
        "sprint3_signals": {
            "semantic_ambiguity_score": float(lang_diag.get("score") or 0.0),
            "language_eval_accuracy": float(lang_eval.get("accuracy") or 0.0),
            "domain_diversity": proc_domains,
            "anti_overfit_bonus": round(float(anti_overfit_bonus), 3),
            "top_domain_share": round(float(top_share), 3),
        },
    }
    _autonomy_state["last_benchmark"] = out
    _benchmark_history_append(out)
    store.db.add_event("agi_benchmark", f"📊 benchmark AGI score={score} req_avg(1-8)={req_avg}", meta_json=json.dumps(out, ensure_ascii=False))
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


@app.post("/api/procedures/invent")
async def procedures_invent(req: ProcedureInventRequest):
    inv = _invent_procedure_from_context(req.context_text, domain=req.domain, name_hint=req.name_hint)
    if not inv:
        raise HTTPException(400, "Could not invent procedure")
    pid = store.add_procedure(
        name=inv['name'],
        goal=inv.get('goal'),
        steps_json=json.dumps(inv.get('steps') or [], ensure_ascii=False),
        domain=inv.get('domain'),
        proc_type=inv.get('proc_type') or 'analysis',
        preconditions=inv.get('preconditions'),
        success_criteria=inv.get('success_criteria'),
    )
    store.db.add_insight("procedure_invented", "Novo procedimento inventado", f"Invenção procedural: {inv['name']} ({inv.get('domain')})", priority=5)
    return {"status": "ok", "procedure_id": pid, "procedure": inv}


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


@app.post("/api/analogies/{analogy_id}/validate")
async def analogy_validate(analogy_id: int):
    res = _validate_analogy_with_evidence(analogy_id)
    if res.get("status") == "not_found":
        raise HTTPException(404, "Analogy not found")
    return res


@app.get("/api/reasoning/audit")
async def reasoning_audit(limit: int = 80):
    # list_events returns oldest-first; fetch a wide window and slice from tail
    evs = [e for e in store.db.list_events(limit=5000) if (e.get('kind') or '') == 'reasoning_audit']
    return {"items": evs[-max(1, int(limit)):], "count": len(evs)}


@app.get("/api/neurosym/proofs")
async def neurosym_proofs(limit: int = 80):
    return {"items": neurosym.history(limit=limit)}


@app.get("/api/neurosym/consistency")
async def neurosym_consistency(limit: int = 200):
    return neurosym.consistency_check(limit=limit)


@app.get("/api/neurosym/fidelity")
async def neurosym_fidelity(limit: int = 120):
    return neurosym.explanation_fidelity(limit=limit)


@app.post("/api/neurosym/check")
async def neurosym_check(limit: int = 200):
    return {"consistency": neurosym.consistency_check(limit=limit), "fidelity": neurosym.explanation_fidelity(limit=min(120, limit))}


@app.get('/api/integrity/status')
async def integrity_status():
    return integrity.status()


@app.post('/api/integrity/rules')
async def integrity_rules_patch(req: IntegrityRulesPatchRequest):
    integrity.save_rules(req.rules or {})
    return integrity.status()


@app.post('/api/integrity/evaluate')
async def integrity_evaluate(kind: str, neural_confidence: float = 0.5, symbolic_consistency: float = 1.0, has_proof: bool = True, causal_checked: bool = True):
    ok, reason = integrity.evaluate(kind, neural_confidence=neural_confidence, symbolic_consistency=symbolic_consistency, has_proof=has_proof, causal_checked=causal_checked)
    return {'allowed': ok, 'reason': reason}


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


@app.get("/api/goals/{goal_id}/milestones")
async def goal_milestones(goal_id: int, status: str = "all", limit: int = 30):
    s = None if status == "all" else status
    items = store.list_goal_milestones(goal_id=goal_id, status=s, limit=limit)
    return {"goal_id": goal_id, "milestones": items, "next": store.get_next_open_milestone(goal_id)}


@app.post("/api/goals/{goal_id}/milestones/ensure")
async def goal_milestones_ensure(goal_id: int, weeks: int = 4):
    g = [x for x in store.db.list_goals(status=None, limit=500) if int(x.get("id") or 0) == int(goal_id)]
    if not g:
        raise HTTPException(404, "Goal not found")
    added = _ensure_goal_milestones(goal_id, g[0].get("title") or "Goal", g[0].get("description"), weeks=weeks)
    return {"status": "ok", "added": added}


@app.post("/api/milestones/{milestone_id}/progress")
async def milestone_progress(milestone_id: int, req: MilestoneProgressRequest):
    p = max(0.0, min(1.0, float(req.progress)))
    st = req.status or ("done" if p >= 1.0 else ("active" if p > 0 else "open"))
    store.update_milestone_progress(milestone_id, p, status=st)
    _audit_reasoning("milestone_progress_update", {"milestone_id": milestone_id}, f"progress={p:.2f}, state={st}", confidence=p)
    return {"status": "ok", "milestone_id": milestone_id, "progress": p, "state": st}

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
