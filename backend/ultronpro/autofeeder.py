"""Auto-feeder: busca conhecimento de fontes públicas sem intervenção humana.

Fontes implementadas:
- Wikipedia random article summary (PT-BR)
- Useless Facts API
- Quotes via quotable.io

Cada fonte tem cooldown para não sobrecarregar.
"""
from __future__ import annotations

import time
import random
import urllib.request
import urllib.error
import json
from dataclasses import dataclass, field
from typing import Any

# Cooldowns in seconds
COOLDOWN_WIKIPEDIA = 300  # 5 min
COOLDOWN_FACTS = 180      # 3 min
COOLDOWN_QUOTES = 240     # 4 min

_last_fetch: dict[str, float] = {}


@dataclass
class FeedResult:
    source_id: str
    modality: str
    text: str
    title: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _get_json(url: str, timeout: float = 10.0) -> dict | list | None:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "UltronPRO/0.1 (educational AGI project)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def fetch_wikipedia_random(lang: str = "pt") -> FeedResult | None:
    """Fetch a random Wikipedia article summary."""
    key = f"wikipedia_{lang}"
    if time.time() - _last_fetch.get(key, 0) < COOLDOWN_WIKIPEDIA:
        return None

    # Get random article title
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"
    data = _get_json(url)
    if not data or not isinstance(data, dict):
        return None

    title = data.get("title") or "Artigo"
    extract = data.get("extract") or ""
    if len(extract) < 50:
        return None

    _last_fetch[key] = time.time()
    return FeedResult(
        source_id=f"wikipedia:{lang}",
        modality="text",
        title=title,
        text=f"{title}\n\n{extract[:2000]}",
        meta={"url": data.get("content_urls", {}).get("desktop", {}).get("page")},
    )


def fetch_useless_fact() -> FeedResult | None:
    """Fetch a random useless fact."""
    key = "uselessfacts"
    if time.time() - _last_fetch.get(key, 0) < COOLDOWN_FACTS:
        return None

    url = "https://uselessfacts.jsph.pl/api/v2/facts/random?language=en"
    data = _get_json(url)
    if not data or not isinstance(data, dict):
        return None

    text = data.get("text") or ""
    if len(text) < 10:
        return None

    _last_fetch[key] = time.time()
    return FeedResult(
        source_id="uselessfacts.jsph.pl",
        modality="text",
        title="Random Fact",
        text=text[:1000],
        meta={"source": data.get("source")},
    )


def fetch_quote() -> FeedResult | None:
    """Fetch a random quote."""
    key = "quotable"
    if time.time() - _last_fetch.get(key, 0) < COOLDOWN_QUOTES:
        return None

    url = "https://api.quotable.io/quotes/random"
    data = _get_json(url)
    if not data or not isinstance(data, list) or len(data) == 0:
        return None

    q = data[0]
    content = q.get("content") or ""
    author = q.get("author") or "Desconhecido"
    if len(content) < 10:
        return None

    _last_fetch[key] = time.time()
    return FeedResult(
        source_id=f"quotable:{author}",
        modality="text",
        title=f"Quote: {author}",
        text=f'"{content}" — {author}',
        meta={"tags": q.get("tags", [])},
    )


def fetch_next() -> FeedResult | None:
    """Try to fetch from any available source (round-robin with cooldown)."""
    fetchers = [
        fetch_wikipedia_random,
        fetch_useless_fact,
        fetch_quote,
    ]
    random.shuffle(fetchers)
    for fn in fetchers:
        try:
            result = fn()
            if result:
                return result
        except Exception:
            pass
    return None
