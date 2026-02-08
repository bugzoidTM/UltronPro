import json
import os
import logging
from typing import Dict

SETTINGS_FILE = "/app/data/settings.json"
logger = logging.getLogger("uvicorn")

# Default settings with discovered LightRAG key
DEFAULT_SETTINGS = {
    "openai_api_key": "",
    "anthropic_api_key": "",
    "groq_api_key": "",
    "deepseek_api_key": "",
    "lightrag_api_key": "b9714901186877fe01b1d7cd81ad65d9",  # Auto-discovered
    "lightrag_url": "http://lightrag2_lightrag2.1.ccxtpz7umbbwfjb1ew0ji8lux:9621/api" # Internal Docker network address
}

def load_settings() -> Dict[str, str]:
    """Load settings from persistent storage, merging with defaults."""
    settings = DEFAULT_SETTINGS.copy()
    
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                settings.update(saved)
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            
    return settings

def save_settings(new_settings: Dict[str, str]) -> bool:
    """Save settings to persistent storage."""
    try:
        current = load_settings()
        # Update only known keys to prevent garbage
        for k in DEFAULT_SETTINGS.keys():
            if k in new_settings:
                current[k] = new_settings[k]
        
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(current, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        return False

def get_api_key(provider: str) -> str:
    """Helper to get specific API key."""
    s = load_settings()
    key_map = {
        "openai": "openai_api_key",
        "anthropic": "anthropic_api_key",
        "groq": "groq_api_key",
        "deepseek": "deepseek_api_key",
        "lightrag": "lightrag_api_key"
    }
    return s.get(key_map.get(provider, ""), "")
