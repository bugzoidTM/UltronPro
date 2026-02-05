from __future__ import annotations

import re
from typing import Iterable


# Very small, deterministic extractor. We keep it simple on purpose.
# Now relaxed to find patterns inside sentences, not just full-line matches.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "X não é Y" (explicit negation)
    (re.compile(r"([A-ZÀ-ÿ][^\n\r]{1,50}?)\s+n[ãa]o\s+é\s+([^\n\r]{1,100}?)[\.,;]", re.IGNORECASE), "não_é"),
    # "X é Y"
    (re.compile(r"([A-ZÀ-ÿ][^\n\r]{1,50}?)\s+é\s+([^\n\r]{1,100}?)[\.,;]", re.IGNORECASE), "é"),
    # "X são Y"
    (re.compile(r"([A-ZÀ-ÿ][^\n\r]{1,50}?)\s+são\s+([^\n\r]{1,100}?)[\.,;]", re.IGNORECASE), "são"),
    # "X tem Y"
    (re.compile(r"([A-ZÀ-ÿ][^\n\r]{1,50}?)\s+tem\s+([^\n\r]{1,100}?)[\.,;]", re.IGNORECASE), "tem"),
    # "X constitui Y"
    (re.compile(r"([A-ZÀ-ÿ][^\n\r]{1,50}?)\s+constitui-?se?\s+([^\n\r]{1,100}?)[\.,;]", re.IGNORECASE), "constitui"),
    # "X significa Y"
    (re.compile(r"([A-ZÀ-ÿ][^\n\r]{1,50}?)\s+significa\s+([^\n\r]{1,100}?)[\.,;]", re.IGNORECASE), "significa"),
]


def extract_norms(text: str) -> list[tuple[str, str, str, float]]:
    """Extract normative 'laws' into a simple triple form.

    This is intentionally heuristic and Portuguese-biased.
    Output triples use subject='AGI' predicate='deve'.
    """
    out: list[tuple[str, str, str, float]] = []
    if not text:
        return out
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        ll = l.lower()
        if ll.startswith('lei ') or ll.startswith('lei:') or ll.startswith('lei-'):
            continue
        # common patterns: "Você deve ...", "Busque ...", "Não ...", "Valorize ..."
        if ll.startswith('você deve '):
            out.append(('AGI', 'deve', l[len('Você deve '):].strip(), 0.75))
        elif ll.startswith('busque '):
            out.append(('AGI', 'deve', l.strip(), 0.7))
        elif ll.startswith('valorize '):
            out.append(('AGI', 'deve', l.strip(), 0.7))
        elif ll.startswith('reconheça '):
            out.append(('AGI', 'deve', l.strip(), 0.7))
        elif ll.startswith('interprete '):
            out.append(('AGI', 'deve', l.strip(), 0.7))
        elif ll.startswith('não '):
            out.append(('AGI', 'não_deve', l[len('Não '):].strip(), 0.7))
    return out


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
