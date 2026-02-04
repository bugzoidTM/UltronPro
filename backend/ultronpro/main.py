from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi import UploadFile, File

from ultronpro.store import Store
from ultronpro.curiosity import CuriosityProcessor
from ultronpro.extract import extract_triples, extract_norms
from ultronpro.media import save_upload
from ultronpro.perception import image_basic_facts
from ultronpro.federated import export_bundle, import_bundle
from ultronpro.signing import sign_bundle, verify_bundle
from ultronpro.policy import evaluate_action
from ultronpro.planner import propose_actions
import json


store = Store("/app/data/ultronpro.sqlite")
curiosity = CuriosityProcessor()


async def _autonomous_loop():
    """Fase 1: autonomia mínima.

    - mantém 3 perguntas abertas (curiosidade)
    - procura contradições e cria perguntas de síntese

    Não depende de clique humano (roda sozinho).
    """
    while True:
        try:
            st = store.stats()

            # 1) Batch absorption: process new experiences into triples
            for e in store.list_unprocessed_experiences(limit=10):
                txt = e.get('text') or ''
                triples = extract_triples(txt)
                norms = []
                if (e.get('modality') or '').lower() == 'law':
                    norms = extract_norms(txt)

                for (s, p, o, conf) in (triples[:10] + norms[:10]):
                    store.add_or_reinforce_triple(s, p, o, confidence=conf, experience_id=e.get('id'), note="from_experience")
                store.mark_experience_processed(int(e.get('id')))

            # 2) Maintain curiosity backlog (3 open questions)
            if st.get('questions_open', 0) < 3:
                exps = store.list_experiences(limit=40)
                open_q = store.list_open_questions(limit=50)
                store.add_questions(curiosity.propose(exps, open_q))

            # 3) Synthesis questions on contradictions + persisted doubt + source governance
            for c in store.find_contradictions(min_conf=0.6):
                info = store.upsert_conflict(c)
                if info:
                    cid = int(info['id'])
                    # IMPORTANT: only penalize sources when the conflict is NEW or CHANGED.
                    if bool(info.get('is_new')) or bool(info.get('has_new_variant')):
                        store.register_contradiction(c)

                    if store.should_prompt_conflict(
                        cid,
                        is_new=bool(info.get('is_new')),
                        has_new_variant=bool(info.get('has_new_variant')),
                        cooldown_hours=12.0,
                    ):
                        store.add_synthesis_question_if_needed(c, conflict_id=cid)

            # 4) Planner/Executor: propose internal actions and run them through policy gate
            act = store.next_action()
            if not act:
                for pa in propose_actions(store):
                    store.enqueue_action(pa.kind, pa.text, priority=pa.priority, meta_json=json.dumps(pa.meta or {}))
                    break
                act = store.next_action()

            if act:
                try:
                    store.mark_action(int(act['id']), 'running')
                    norms = store.list_norms(limit=200)
                    v = evaluate_action(act.get('text') or '', norms)
                    if not v.allowed:
                        store.mark_action(int(act['id']), 'blocked', policy_allowed=False, policy_score=v.score, last_error='policy_block')
                    else:
                        # Execute only internal safe actions: turn into a question
                        if act.get('kind') in ('ask_evidence','clarify_laws'):
                            store.add_questions([{"question": act.get('text') or 'Ação', "context": 'executor', "priority": 4}])
                        elif act.get('kind') == 'generate_questions':
                            exps = store.list_experiences(limit=40)
                            open_q = store.list_open_questions(limit=50)
                            store.add_questions(curiosity.propose(exps, open_q))
                        store.mark_action(int(act['id']), 'done', policy_allowed=True, policy_score=v.score)
                except Exception as e:
                    store.mark_action(int(act['id']), 'error', last_error=str(e)[:500])
        except Exception:
            pass
        await asyncio.sleep(20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_autonomous_loop())
    yield
    task.cancel()


app = FastAPI(title="UltronPRO", version="0.1.0", lifespan=lifespan)


class Ingest(BaseModel):
    user_id: str | None = None
    source_id: str | None = None
    modality: str = Field(default="text")
    text: str | None = Field(default=None, max_length=20000)
    title: str | None = Field(default=None, max_length=200)


class Answer(BaseModel):
    question_id: int
    answer: str = Field(..., min_length=1, max_length=20000)


class Dismiss(BaseModel):
    question_id: int


@app.get("/api/status")
async def status():
    return {"stats": store.stats(), "next": store.next_question()}


@app.post("/api/ingest")
async def ingest(req: Ingest):
    if req.source_id:
        store.ensure_source(req.source_id, kind="manual", label=req.source_id)

    # If this is a 'law', persist it explicitly as legislation (primary law scaffolding).
    if (req.modality or '').lower() == 'law' and req.text:
        store.add_law(req.text, title=req.title, source_id=req.source_id)

    eid = store.add_experience(req.user_id, req.text, source_id=req.source_id, modality=req.modality)

    # Auto-curiosidade: após ingerir, mantenha pelo menos 3 perguntas abertas.
    st = store.stats()
    if st.get('questions_open', 0) < 3:
        exps = store.list_experiences(limit=40)
        open_q = store.list_open_questions(limit=50)
        store.add_questions(curiosity.propose(exps, open_q))

    return {"success": True, "experience_id": eid}


@app.post("/api/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    user_id: str | None = None,
    source_id: str | None = None,
):
    if source_id:
        store.ensure_source(source_id, kind="upload", label=source_id)

    mime = file.content_type or "application/octet-stream"
    data = await file.read()

    suffix = ""
    if mime.startswith("image/"):
        suffix = ".img"
    elif mime.startswith("audio/"):
        suffix = ".audio"
    elif mime.startswith("text/"):
        suffix = ".txt"

    path = save_upload(data, "/app/data/uploads", suffix=suffix)

    modality = "file"
    text: str | None = None

    if mime.startswith("image/"):
        modality = "image"
        try:
            facts = image_basic_facts(path)
            text = f"IMAGE_FACTS {facts}"
        except Exception:
            text = "IMAGE_UPLOAD (facts_unavailable)"
    elif mime.startswith("audio/"):
        modality = "audio"
        text = f"AUDIO_UPLOAD mime={mime} bytes={len(data)}"
    elif mime.startswith("text/"):
        modality = "text"
        try:
            text = data.decode("utf-8", errors="replace")[:20000]
        except Exception:
            text = None

    eid = store.add_experience(
        user_id=user_id,
        text=text,
        source_id=source_id,
        modality=modality,
        blob_path=path,
        mime=mime,
    )
    return {"success": True, "experience_id": eid, "modality": modality, "mime": mime}


@app.post("/api/curiosity/refresh")
async def refresh():
    exps = store.list_experiences(limit=40)
    open_q = store.list_open_questions(limit=50)
    proposed = curiosity.propose(exps, open_q)
    store.add_questions(proposed)
    return {"success": True, "proposed": proposed, "next": store.next_question()}


@app.get("/api/next")
async def next_q():
    q = store.next_question()
    if not q:
        raise HTTPException(status_code=404, detail="No open questions")
    return q


@app.post("/api/answer")
async def answer(req: Answer):
    store.answer_question(req.question_id, req.answer)

    # Assimilação mínima: extrair triplas do texto (heurístico) e registrar como conhecimento.
    triples = extract_triples(req.answer)
    for (s, p, o, conf) in triples[:10]:
        store.add_or_reinforce_triple(s, p, o, confidence=conf, note=f"from_answer:q={req.question_id}")

    return {"success": True, "triples_added": len(triples)}


@app.post("/api/dismiss")
async def dismiss(req: Dismiss):
    store.dismiss_question(req.question_id)
    return {"success": True}


@app.post("/api/reset/questions")
async def reset_questions():
    store.reset_questions()
    return {"success": True}


@app.get("/api/laws")
async def list_laws(status: str = 'active', limit: int = 50):
    return {"success": True, "laws": store.list_laws(status=status, limit=limit)}


@app.get("/api/norms")
async def list_norms(limit: int = 200):
    return {"success": True, "norms": store.list_norms(limit=limit)}


@app.get("/api/graph/triples")
async def graph_triples(since_id: int = 0, limit: int = 500):
    return {"success": True, "triples": store.list_triples_since(since_id=since_id, limit=limit)}


@app.get("/api/graph/conflicts")
async def graph_conflicts(limit: int = 25):
    conflicts = store.list_conflicts(status='open', limit=limit)
    # include variants for coloring nodes
    enriched = []
    for c in conflicts:
        cid = int(c.get('id'))
        full = store.get_conflict(cid) or c
        enriched.append(full)
    return {"success": True, "conflicts": enriched}


@app.get("/api/actions")
async def actions(limit: int = 50):
    return {"success": True, "actions": store.list_actions(limit=limit)}


@app.post("/api/laws/{law_id}/archive")
async def archive_law(law_id: int):
    store.archive_law(law_id)
    return {"success": True}


class MigrateLawsReq(BaseModel):
    limit: int = Field(default=200, ge=1, le=5000)


@app.post("/api/laws/migrate")
async def migrate_laws(req: MigrateLawsReq):
    res = store.migrate_text_experiences_to_laws(limit=req.limit)
    return {"success": True, **res}


class PolicyCheckReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


@app.post("/api/policy/check")
async def policy_check(req: PolicyCheckReq):
    norms = store.list_norms(limit=200)
    v = evaluate_action(req.text, norms)
    return {
        "success": True,
        "allowed": v.allowed,
        "score": v.score,
        "reasons": v.reasons,
        "matched_norms": v.matched_norms,
    }


@app.get("/api/conflicts")
async def list_conflicts(status: str = 'open', limit: int = 50):
    return {"success": True, "conflicts": store.list_conflicts(status=status, limit=limit)}


@app.get("/api/conflicts/{conflict_id}")
async def get_conflict(conflict_id: int):
    c = store.get_conflict(conflict_id)
    if not c:
        raise HTTPException(status_code=404, detail="Conflict not found")
    return {"success": True, "conflict": c}


class ResolveConflict(BaseModel):
    resolution: str | None = Field(default=None, max_length=2000)
    chosen_object: str | None = Field(default=None, max_length=500)
    decided_by: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)


@app.post("/api/conflicts/{conflict_id}/resolve")
async def resolve_conflict(conflict_id: int, req: ResolveConflict):
    store.resolve_conflict(
        conflict_id,
        resolution=req.resolution,
        chosen_object=req.chosen_object,
        decided_by=req.decided_by,
        notes=req.notes,
    )
    return {"success": True}


@app.post("/api/conflicts/{conflict_id}/archive")
async def archive_conflict(conflict_id: int):
    store.archive_conflict(conflict_id)
    return {"success": True}


class ImportBundle(BaseModel):
    bundle: dict
    source: str | None = None
    signature: str | None = None


def _bundle_key() -> bytes | None:
    # Optional key for signed bundles.
    # For Swarm, mount a secret at /run/secrets/ultronpro_bundle_key.
    import os
    p = "/run/secrets/ultronpro_bundle_key"
    if os.path.exists(p):
        try:
            return open(p, "rb").read().strip() or None
        except Exception:
            return None
    env = os.getenv("ULTRONPRO_BUNDLE_KEY")
    return env.encode("utf-8") if env else None


@app.get("/api/federated/export")
async def federated_export(since_experience_id: int | None = None):
    return export_bundle(store, since_id=since_experience_id)


@app.get("/api/federated/export_signed")
async def federated_export_signed(since_experience_id: int | None = None):
    bundle = export_bundle(store, since_id=since_experience_id)
    key = _bundle_key()
    if not key:
        return {"success": False, "error": "Signing key not configured", "bundle": bundle}
    sig = sign_bundle(bundle, key)
    return {"success": True, "bundle": bundle, "signature": sig}


@app.post("/api/federated/import")
async def federated_import(req: ImportBundle):
    # If key exists, require signature.
    key = _bundle_key()
    if key:
        if not req.signature:
            raise HTTPException(status_code=400, detail="Missing signature")
        if not verify_bundle(req.bundle, req.signature, key):
            raise HTTPException(status_code=400, detail="Invalid signature")

    res = import_bundle(store, req.bundle, source=req.source or "federated")
    return {"success": True, **res}


# --- UI (single-page static)
app.mount("/static", StaticFiles(directory="/app/ui"), name="static")


@app.get("/", response_class=HTMLResponse)
async def ui():
    with open("/app/ui/index.html", "r", encoding="utf-8") as f:
        return f.read()
