import httpx
import os
import logging
from typing import List, Dict, Optional

LIGHTRAG_URL = "https://lightrag.nutef.com/api"
LIGHTRAG_KEY = os.getenv("LIGHTRAG_API_KEY")

logger = logging.getLogger("uvicorn")

async def search_knowledge(query: str, top_k: int = 5) -> List[Dict]:
    """Search LightRAG knowledge base."""
    if not LIGHTRAG_KEY:
        logger.warning("LIGHTRAG_API_KEY not configured. Skipping knowledge search.")
        return []
        
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{LIGHTRAG_URL}/search",
                headers={"X-API-Key": LIGHTRAG_KEY},
                json={"query": query, "limit": top_k},
                timeout=10.0
            )
            resp.raise_for_status()
            data = resp.json()
            # Adapt LightRAG response format to UltronPro expectations
            return [{"text": r.get("content"), "source_id": "lightrag", "score": r.get("score")} for r in data.get("results", [])]
            
    except Exception as e:
        logger.error(f"LightRAG Search Error: {e}")
        return []

async def ingest_knowledge(text: str, source: str = "ultronpro") -> bool:
    """Push new knowledge to LightRAG."""
    if not LIGHTRAG_KEY:
        return False
        
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{LIGHTRAG_URL}/ingest",
                headers={"X-API-Key": LIGHTRAG_KEY},
                json={"text": text, "metadata": {"source": source}},
                timeout=30.0
            )
            return resp.status_code == 200
    except Exception as e:
        logger.error(f"LightRAG Ingest Error: {e}")
        return False
