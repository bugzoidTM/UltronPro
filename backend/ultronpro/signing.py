from __future__ import annotations

import hmac
import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_bundle(bundle: dict[str, Any], key: bytes) -> str:
    mac = hmac.new(key, canonical_json(bundle), hashlib.sha256)
    return mac.hexdigest()


def verify_bundle(bundle: dict[str, Any], signature: str, key: bytes) -> bool:
    try:
        expected = sign_bundle(bundle, key)
        return hmac.compare_digest(expected, signature)
    except Exception:
        return False
