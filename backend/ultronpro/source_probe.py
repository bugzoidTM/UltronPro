from __future__ import annotations

import re
import html
from typing import Any

import httpx


def _clean_html(raw: str) -> str:
    s = raw or ''
    s = re.sub(r'(?is)<script[^>]*>.*?</script>', ' ', s)
    s = re.sub(r'(?is)<style[^>]*>.*?</style>', ' ', s)
    s = re.sub(r'(?is)<noscript[^>]*>.*?</noscript>', ' ', s)
    s = re.sub(r'(?is)<svg[^>]*>.*?</svg>', ' ', s)
    s = re.sub(r'(?is)<[^>]+>', ' ', s)
    s = html.unescape(s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def fetch_clean_text(url: str, max_chars: int = 8000, timeout_sec: float = 12.0) -> dict[str, Any]:
    u = (url or '').strip()
    if not (u.startswith('http://') or u.startswith('https://')):
        return {'ok': False, 'error': 'invalid_url'}

    lim = max(500, min(int(max_chars or 8000), 50000))

    try:
        with httpx.Client(timeout=timeout_sec, follow_redirects=True, headers={'User-Agent': 'UltronPRO/0.1 source-probe'}) as client:
            r = client.get(u)
            ct = (r.headers.get('content-type') or '').lower()
            body = r.text if 'text' in ct or 'html' in ct or not ct else ''
            title = None
            m = re.search(r'(?is)<title[^>]*>(.*?)</title>', body)
            if m:
                title = _clean_html(m.group(1))[:240]
            text = _clean_html(body)
            if len(text) > lim:
                text = text[:lim]
            return {
                'ok': True,
                'url': str(r.url),
                'status_code': int(r.status_code),
                'content_type': ct,
                'title': title,
                'text': text,
                'text_chars': len(text),
            }
    except Exception as e:
        return {'ok': False, 'error': str(e)[:300], 'url': u}
