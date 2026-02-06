import httpx
import json
import logging

OLLAMA_URL = "http://ollama:11434/api/generate"
logger = logging.getLogger("uvicorn")

def complete(prompt: str, model: str = "mistral", system: str = None, json_mode: bool = False) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096}
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"
        
    try:
        r = httpx.post(OLLAMA_URL, json=payload, timeout=120.0)
        if r.status_code == 200:
            return r.json().get("response", "")
        logger.error(f"Ollama Error {r.status_code}: {r.text}")
        return ""
    except Exception as e:
        logger.error(f"Ollama Exception: {e}")
        return ""
