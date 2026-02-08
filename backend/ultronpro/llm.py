import os
import logging
from openai import OpenAI
from groq import Groq
from anthropic import Anthropic

# LLM Routing Strategy
MODELS = {
    "default": {"provider": "openai", "model": "gpt-4o-mini"},
    "cheap": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "reasoning": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
    "creative": {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
    "deep": {"provider": "deepseek", "model": "deepseek-reasoner"}
}

logger = logging.getLogger("uvicorn")

class LLMRouter:
    def __init__(self):
        self.clients = {}

    def _get_client(self, provider: str):
        if provider in self.clients:
            return self.clients[provider]
        
        # Lazy init logic with environment keys
        try:
            if provider == "openai":
                key = os.getenv("OPENAI_API_KEY")
                if key: self.clients[provider] = OpenAI(api_key=key)
            elif provider == "anthropic":
                key = os.getenv("ANTHROPIC_API_KEY")
                if key: self.clients[provider] = Anthropic(api_key=key)
            elif provider == "groq":
                key = os.getenv("GROQ_API_KEY")
                if key: self.clients[provider] = Groq(api_key=key)
            elif provider == "deepseek":
                key = os.getenv("DEEPSEEK_API_KEY")
                if key: self.clients[provider] = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        except Exception as e:
            logger.error(f"Failed to init {provider}: {e}")

        return self.clients.get(provider)

    def complete(self, prompt: str, strategy: str = "default", system: str = None, json_mode: bool = False) -> str:
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

        if not client:
            logger.error("No LLM clients available. Set API keys.")
            return ""

        try:
            if provider == "anthropic":
                return self._call_anthropic(client, model, prompt, system, json_mode)
            else:
                return self._call_openai_compat(client, model, prompt, system, json_mode)
        except Exception as e:
            logger.error(f"LLM Call Error ({provider}/{model}): {e}")
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

router = LLMRouter()
def complete(prompt: str, strategy: str = "default", system: str = None, json_mode: bool = False) -> str:
    return router.complete(prompt, strategy, system, json_mode)
