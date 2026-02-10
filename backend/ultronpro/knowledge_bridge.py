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

def _extract_context_any(data) -> str:
    """Tolerante a mudanças de schema JSON."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "\n".join([_extract_context_any(x) for x in data[:6] if _extract_context_any(x)])
    if isinstance(data, dict):
        # chaves comuns em variações de API
        for k in ("context", "response", "result", "answer", "text", "content"):
            v = data.get(k)
            if v:
                return _extract_context_any(v)
        # fallback: serializa campos curtos
        parts = []
        for k, v in list(data.items())[:8]:
            sv = _extract_context_any(v)
            if sv:
                parts.append(f"{k}: {sv}")
        return "\n".join(parts)
    return str(data)


async def search_knowledge(query: str, top_k: int = 5) -> List[Dict]:
    """Search LightRAG knowledge base with adaptive endpoint/schema handling."""
    url, key = _get_lightrag_config()
    if not url or not key:
        logger.warning("LightRAG not configured. Skipping knowledge search.")
        return []

    base = url.rstrip("/")
    payload = {
        "query": query,
        "mode": "mix",
        "top_k": int(top_k),
        "only_need_context": True,
    }

    endpoints = [f"{base}/query", f"{base}/query/data"]

    try:
        async with httpx.AsyncClient() as client:
            last_err = None
            for ep in endpoints:
                try:
                    resp = await client.post(
                        ep,
                        headers={"X-API-Key": key},
                        json=payload,
                        timeout=20.0,
                    )
                    if resp.status_code >= 400:
                        last_err = f"{ep} -> {resp.status_code}"
                        continue

                    data = resp.json()
                    context = _extract_context_any(data)
                    if not context:
                        continue

                    text = str(context)
                    return [{"text": text[:2500], "source_id": "lightrag", "score": 0.7, "type": "experience"}]
                except Exception as e:
                    last_err = str(e)
                    continue

            if last_err:
                logger.error(f"LightRAG Search Error: {last_err}")
            return []

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
