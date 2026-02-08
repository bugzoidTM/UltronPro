import httpx
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("uvicorn")

def _get_lightrag_config():
    """Load LightRAG config from settings."""
    from ultronpro import settings
    s = settings.load_settings()
    return s.get("lightrag_url"), s.get("lightrag_api_key")

async def search_knowledge(query: str, top_k: int = 5) -> List[Dict]:
    """Search LightRAG knowledge base."""
    url, key = _get_lightrag_config()
    if not url or not key:
        logger.warning("LightRAG not configured. Skipping knowledge search.")
        return []
        
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{url}/search",
                headers={"X-API-Key": key},
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

async def fetch_random_documents(limit: int = 1) -> List[Dict]:
    """Fetch random documents from LightRAG."""
    url, key = _get_lightrag_config()
    if not url or not key:
        return []
        
    try:
        async with httpx.AsyncClient() as client:
            # List all documents
            resp = await client.get(
                f"{url.replace('/api', '')}/documents",
                headers={"X-API-Key": key},
                timeout=10.0
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Get processed documents
            processed = data.get("statuses", {}).get("processed", [])
            if not processed:
                return []
            
            # Pick random ones
            import random
            selected = random.sample(processed, min(limit, len(processed)))
            
            results = []
            for doc in selected:
                # Fetch full document content
                doc_id = doc.get("id")
                if doc_id:
                    try:
                        doc_resp = await client.get(
                            f"{url.replace('/api', '')}/documents/{doc_id}",
                            headers={"X-API-Key": key},
                            timeout=5.0
                        )
                        if doc_resp.status_code == 200:
                            doc_data = doc_resp.json()
                            content = doc_data.get("content", "")
                            if len(content) > 100:  # Only if has substantial content
                                results.append({
                                    "id": doc_id,
                                    "content": content[:2000],  # Limit to 2000 chars
                                    "summary": doc.get("content_summary", "")[:200]
                                })
                    except Exception:
                        pass
            
            return results
            
    except Exception as e:
        logger.error(f"LightRAG Fetch Error: {e}")
        return []

async def ingest_knowledge(text: str, source: str = "ultronpro") -> bool:
    """Push new knowledge to LightRAG (DISABLED - unidirectional flow FROM LightRAG only)."""
    # Disabled by user request - knowledge flows FROM LightRAG to UltronPro, not the reverse
    return False
