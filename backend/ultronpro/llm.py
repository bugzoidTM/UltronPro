import os
import logging
from typing import Dict, Optional, Any
import httpx
from ultronpro.settings import get_api_key
from ultronpro import persona

# --- API Models ---
from pydantic import BaseModel

class SettingsUpdate(BaseModel):
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    lightrag_api_key: Optional[str] = None

logger = logging.getLogger("uvicorn")

# LLM Routing Strategy
MODELS = {
    "default": {"provider": "openai", "model": "gpt-4o-mini"},
    "cheap": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "reasoning": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
    "creative": {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
    "deep": {"provider": "deepseek", "model": "deepseek-reasoner"},
    "local": {"provider": "ollama", "model": os.getenv('OLLAMA_MODEL', 'llama3.2:1b')}
}

class LLMRouter:
    def __init__(self):
        # Clients cache
        self.clients = {}

    def _get_client(self, provider: str) -> Any:
        if provider in self.clients:
            return self.clients[provider]
        
        client = None
        try:
            key = get_api_key(provider)
            
            if provider == "openai":
                from openai import OpenAI
                if key: client = OpenAI(api_key=key)
            elif provider == "anthropic":
                from anthropic import Anthropic
                if key: client = Anthropic(api_key=key)
            elif provider == "groq":
                from groq import Groq
                if key: client = Groq(api_key=key)
            elif provider == "deepseek":
                from openai import OpenAI
                if key: client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
            elif provider == "ollama":
                # local HTTP endpoint; no API key required
                client = {"base_url": os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')}
        except Exception as e:
            logger.error(f"Failed to init {provider}: {e}")

        if client:
            self.clients[provider] = client
            
        return client

    def complete(self, prompt: str, strategy: str = "default", system: str = None, json_mode: bool = False) -> str:
        # runtime personality injection (dynamic system prompt + few-shot exemplars)
        try:
            system = persona.build_system_prompt(system)
        except Exception:
            pass

        config = MODELS.get(strategy, MODELS["default"])
        provider = config["provider"]
        model = config["model"]
        
        # 1. Try preferred provider
        client = self._get_client(provider)
        
        # 2. Fallback to OpenAI if missing
        if not client:
            if provider != "openai":
                logger.warning(f"Provider {provider} missing, falling back to OpenAI.")
                provider = "openai"
                model = MODELS["default"]["model"]
                client = self._get_client("openai")

        # 3. Fallback to local Ollama if cloud providers unavailable
        if not client:
            provider = 'ollama'
            model = MODELS['local']['model']
            client = self._get_client('ollama')

        if not client:
            logger.error("No LLM clients available. Please configure API Keys in Settings.")
            return ""

        try:
            if provider == "anthropic":
                return self._call_anthropic(client, model, prompt, system, json_mode)
            if provider == 'ollama':
                return self._call_ollama(client, model, prompt, system, json_mode)
            return self._call_openai_compat(client, model, prompt, system, json_mode)
        except Exception as e:
            logger.error(f"LLM Call Error ({provider}/{model}): {e}")
            # final fallback: ollama local
            try:
                ocli = self._get_client('ollama')
                if ocli:
                    return self._call_ollama(ocli, MODELS['local']['model'], prompt, system, json_mode)
            except Exception:
                pass
            return ""

    def _call_openai_compat(self, client, model, prompt, system, json_mode):
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {"model": model, "messages": messages, "temperature": 0.3}
        if json_mode and "groq" not in str(client.base_url):
            kwargs["response_format"] = {"type": "json_object"}

        res = client.chat.completions.create(**kwargs)
        return res.choices[0].message.content

    def _call_anthropic(self, client, model, prompt, system, json_mode):
        kwargs = {
            "model": model, 
            "max_tokens": 4096, 
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        if system: kwargs["system"] = system
        if json_mode: kwargs["messages"][0]["content"] += "\nReturn ONLY valid JSON."
            
        res = client.messages.create(**kwargs)
        return res.content[0].text

    def _call_ollama(self, client, model, prompt, system, json_mode):
        base = (client or {}).get('base_url') or os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
        body_prompt = prompt
        if system:
            body_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}"
        if json_mode:
            body_prompt += "\n\nReturn ONLY valid JSON."
        payload = {
            'model': model,
            'prompt': body_prompt,
            'stream': False,
            'options': {'temperature': 0.2}
        }
        with httpx.Client(timeout=25.0) as hc:
            r = hc.post(base.rstrip('/') + '/api/generate', json=payload)
            r.raise_for_status()
            data = r.json()
            return str(data.get('response') or '')

router = LLMRouter()
def complete(prompt: str, strategy: str = "default", system: str = None, json_mode: bool = False) -> str:
    return router.complete(prompt, strategy, system, json_mode)
