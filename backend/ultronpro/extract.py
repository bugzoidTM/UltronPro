from __future__ import annotations

import re
from typing import Iterable


# Very small, deterministic extractor. We keep it simple on purpose.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "X não é Y" (explicit negation)
    (re.compile(r"^\s*([^\n\r]{1,80}?)\s+n[ãa]o\s+é\s+([^\n\r]{1,120}?)\s*\.?$", re.IGNORECASE), "não_é"),
    # "X é Y"
    (re.compile(r"^\s*([^\n\r]{1,80}?)\s+é\s+([^\n\r]{1,120}?)\s*\.?$", re.IGNORECASE), "é"),
    # "X tem Y"
    (re.compile(r"^\s*([^\n\r]{1,80}?)\s+tem\s+([^\n\r]{1,120}?)\s*\.?$", re.IGNORECASE), "tem"),
    # "X significa Y"
    (re.compile(r"^\s*([^\n\r]{1,80}?)\s+significa\s+([^\n\r]{1,120}?)\s*\.?$", re.IGNORECASE), "significa"),
]


def extract_triples(text: str) -> list[tuple[str, str, str, float]]:
    """Return list of (subject,predicate,object,confidence)."""
    out: list[tuple[str, str, str, float]] = []
    for raw_line in re.split(r"[\n\r]+", text or ""):
        line = raw_line.strip()
        if not line:
            continue
        # Avoid super long lines
        if len(line) > 260:
            continue
        for pat, pred in _PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            subj = m.group(1).strip(" \t\"'")
            obj = m.group(2).strip(" \t\"'")
            if subj and obj:
                out.append((subj, pred, obj, 0.65))
            break
    return out
