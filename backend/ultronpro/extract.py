from __future__ import annotations
import re
import json
from ultronpro import llm

# Regex fallback (just in case)
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"([A-ZÀ-ÿ][^\n\r]{1,50}?)\s+é\s+([^\n\r]{1,100}?)[\.,;]", re.IGNORECASE), "é"),
]

def extract_norms(text: str) -> list[tuple[str, str, str, float]]:
    """Extract norms using LLM."""
    if not text: return []
    prompt = f"""Extract normative rules/laws from the text.
Return JSON array of objects with keys: "rule" (text of the rule).
Text: {text[:4000]}"""
    
    res = llm.complete(prompt, json_mode=True)
    out = []
    try:
        data = json.loads(res)
        if isinstance(data, dict):
            # diverse formats handling
            if 'rules' in data: data = data['rules']
            elif 'norms' in data: data = data['norms']
        
        if isinstance(data, list):
            for i in data:
                txt = i if isinstance(i, str) else i.get('rule') or i.get('text')
                if txt:
                    out.append(('AGI', 'deve', txt, 0.8))
    except:
        pass
    return out

def extract_triples(text: str) -> list[tuple[str, str, str, float]]:
    """Extract triples using LLM (Mistral)."""
    if not text or len(text) < 10:
        return []

    # Optimization: if text is very short, use regex to save GPU
    if len(text) < 50:
        # minimal regex logic
        return []

    prompt = f"""Extract key facts from the text as triples (Subject, Predicate, Object).
Focus on relationships, definitions, and causality.
Return ONLY a JSON array of objects with keys "s", "p", "o".
Output in Portuguese.
Text: {text[:3000]}"""

    res = llm.complete(prompt, json_mode=True)
    out = []
    try:
        data = json.loads(res)
        if isinstance(data, dict):
            if 'triples' in data: data = data['triples']
            elif 'facts' in data: data = data['facts']
            
        if isinstance(data, list):
            for i in data:
                s = i.get('s') or i.get('subject')
                p = i.get('p') or i.get('predicate')
                o = i.get('o') or i.get('object')
                if s and p and o:
                    out.append((str(s), str(p), str(o), 0.85))
    except Exception:
        pass
    
    return out
