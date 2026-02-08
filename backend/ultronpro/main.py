import os
import logging
import json
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ultronpro import llm, knowledge_bridge, graph, settings, curiosity, conflicts, store, extract, planner, autofeeder
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

# --- Startup ---
_autofeeder_task = None

async def autofeeder_loop():
    """Background task que busca conhecimento de fontes públicas."""
    logger.info("Autofeeder started")
    await asyncio.sleep(30)  # Wait 30s before first fetch
    
    while True:
        try:
            result = autofeeder.fetch_next()
            if result:
                # Ingest the fetched content (sem LLM, só armazena)
                exp_id = store.add_experience(
                    text=result.text,
                    source_id=result.source_id,
                    modality=result.modality
                )
                logger.info(f"Autofeeder: Ingested from {result.source_id} (exp_id={exp_id})")
        except Exception as e:
            logger.error(f"Autofeeder error: {e}")
        
        # Wait 60 seconds before next attempt (cooldowns are handled internally)
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    global _autofeeder_task
    logger.info("Starting UltronPRO...")
    store.init_db()
    graph.init()
    # Ensure settings are loaded/initialized
    s = settings.load_settings()
    logger.info(f"Loaded settings. LightRAG URL: {s.get('lightrag_url')}")
    
    # Start background autofeeder
    _autofeeder_task = asyncio.create_task(autofeeder_loop())
    logger.info("Autofeeder task created")

@app.on_event("shutdown")
async def shutdown_event():
    global _autofeeder_task
    if _autofeeder_task:
        _autofeeder_task.cancel()
        try:
            await _autofeeder_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutdown complete")

# --- API Endpoints ---

@app.get("/api/status")
async def get_status():
    """System status and next question."""
    stats = store.get_stats()
    next_q = curiosity.get_next_question()
    return {"status": "online", "stats": stats, "next": next_q}

@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """Ingest raw text/experience."""
    # 1. Store raw experience
    exp_id = store.add_experience(req.text, req.source_id, req.modality)
    
    # 2. Extract Triples (LLM)
    triples = extract.extract_triples(req.text)
    
    # 3. Update Graph & Detect Conflicts
    added = 0
    for t in triples:
        if graph.add_triple(t, source_id=f"exp_{exp_id}"):
            added += 1
            
    # 4. Push to LightRAG (Disabled by user request - knowledge flows FROM LightRAG only)
    # await ingest_knowledge(req.text, source=req.source_id or "user")

    return {"status": "ok", "experience_id": exp_id, "triples_extracted": len(triples), "triples_added": added}

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
    return {"resolved": len([r for r in results if r['resolved']]), "results": results}

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
