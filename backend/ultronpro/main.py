# ... existing code ...
from fastapi import FastAPI
from ultronpro.knowledge_bridge import search_knowledge
import logging

app = FastAPI()
logger = logging.getLogger("uvicorn")

@app.get("/api/search")
async def search(query: str):
    """Semantic search integration."""
    try:
        results = await search_knowledge(query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Search API Error: {e}")
        return {"error": str(e), "results": []}

# ... rest of main.py ...
