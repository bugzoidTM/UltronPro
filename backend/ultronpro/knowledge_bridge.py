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
    """Search LightRAG knowledge base via /query endpoint."""
    url, key = _get_lightrag_config()
    if not url or not key:
        logger.warning("LightRAG not configured. Skipping knowledge search.")
        return []

    base = url.rstrip("/")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base}/query",
                headers={"X-API-Key": key},
                json={
                    "query": query,
                    "mode": "mix",
                    "top_k": int(top_k),
                    "only_need_context": True,
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()

            # LightRAG usually returns context blob; adapt into list for UI
            context = data.get("context") or data.get("response") or ""
            if not context:
                return []

            text = str(context)
            return [{"text": text[:2500], "source_id": "lightrag", "score": 0.7, "type": "experience"}]

    except Exception as e:
        logger.error(f"LightRAG Search Error: {e}")
        return []

async def fetch_random_documents(limit: int = 1) -> List[Dict]:
    """Fetch random documents from LightRAG.

    Compatível com instâncias onde /documents/{id} não existe.
    """
    url, key = _get_lightrag_config()
    if not url or not key:
        return []

    base = url.replace('/api', '')

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base}/documents",
                headers={"X-API-Key": key},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

            processed = data.get("statuses", {}).get("processed", [])
            if not processed:
                return []

            import random
            selected = random.sample(processed, min(limit, len(processed)))

            results = []
            for doc in selected:
                doc_id = doc.get("id")
                if not doc_id:
                    continue

                # 1) Tenta endpoint de detalhe (se existir)
                content = ""
                try:
                    doc_resp = await client.get(
                        f"{base}/documents/{doc_id}",
                        headers={"X-API-Key": key},
                        timeout=5.0,
                    )
                    if doc_resp.status_code == 200:
                        doc_data = doc_resp.json()
                        content = (doc_data.get("content") or "").strip()
                except Exception:
                    pass

                # 2) Fallback: usa summary do próprio listing
                if not content:
                    content = (doc.get("content_summary") or "").strip()

                if len(content) > 40:
                    results.append(
                        {
                            "id": doc_id,
                            "content": content[:2000],
                            "summary": (doc.get("content_summary") or content)[:200],
                        }
                    )

            return results

    except Exception as e:
        logger.error(f"LightRAG Fetch Error: {e}")
        return []

async def ingest_knowledge(text: str, source: str = "ultronpro") -> bool:
    """Push new knowledge to LightRAG (DISABLED - unidirectional flow FROM LightRAG only)."""
    # Disabled by user request - knowledge flows FROM LightRAG to UltronPro, not the reverse
    return False
