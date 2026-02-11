from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import time

STATE_PATH = Path('/app/data/unsupervised_state.json')


def _tokens(text: str) -> list[str]:
    t = re.sub(r"[^\w\sà-ÿÀ-Ÿ-]", " ", (text or '').lower())
    ws = [w.strip('-') for w in re.split(r"\s+", t) if len(w.strip('-')) >= 4]
    stop = {
        'para','como','com','sem','sobre','entre','essa','esse','isso','uma','mais','menos','from','that','this',
        'quando','onde','quais','qual','depois','antes','porque','porquê','então','entao','muito','pouco','todo','toda'
    }
    return [w for w in ws if w not in stop][:120]


def _load_state() -> dict[str, Any]:
    try:
        if STATE_PATH.exists():
            d = json.loads(STATE_PATH.read_text())
            if isinstance(d, dict):
                d.setdefault('concepts', {})
                d.setdefault('edges', {})
                d.setdefault('last_run_at', 0)
                return d
    except Exception:
        pass
    return {'concepts': {}, 'edges': {}, 'last_run_at': 0}


def _save_state(st: dict[str, Any]):
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2))
    except Exception:
        pass


def state_summary() -> dict[str, Any]:
    st = _load_state()
    concepts = st.get('concepts') or {}
    edges = st.get('edges') or {}
    top_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        'scanned': 0,
        'concepts_total': len(concepts),
        'edges_total': len(edges),
        'top_concepts': [{'term': c, 'freq': f} for c, f in top_concepts],
        'top_edges': [{'pair': k, 'weight': w} for k, w in top_edges],
        'triples_induced': 0,
        'last_run_at': st.get('last_run_at') or 0,
    }


def discover_and_restructure(db, max_experiences: int = 200) -> dict[str, Any]:
    """Aprendizado não-supervisionado leve: induz conceitos latentes e relações sem template fixo."""
    st = _load_state()
    exps = db.list_experiences(limit=max(20, int(max_experiences)))

    cooc = {}
    concepts = st.get('concepts') or {}
    edges = st.get('edges') or {}

    scanned = 0
    for e in exps:
        txt = (e.get('text') or '').strip()
        if len(txt) < 40:
            continue
        scanned += 1
        toks = _tokens(txt)
        uniq = list(dict.fromkeys(toks))[:40]
        for t in uniq:
            concepts[t] = int(concepts.get(t) or 0) + 1
        for i in range(len(uniq)):
            for j in range(i + 1, min(len(uniq), i + 8)):
                a, b = sorted([uniq[i], uniq[j]])
                k = f"{a}|{b}"
                cooc[k] = int(cooc.get(k) or 0) + 1

    # merge cooc into persistent edges
    for k, v in cooc.items():
        edges[k] = int(edges.get(k) or 0) + int(v)

    # select latent concepts by frequency (no predefined ontology)
    top_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:30]
    top_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)[:40]

    new_triples = 0
    for c, freq in top_concepts[:12]:
        if freq >= 3:
            try:
                db.add_or_reinforce_triple('LATENT', 'concept', c, confidence=min(0.95, 0.45 + freq * 0.02))
                new_triples += 1
            except Exception:
                pass

    for k, w in top_edges[:20]:
        if w < 3:
            continue
        a, b = k.split('|', 1)
        conf = min(0.92, 0.4 + (w * 0.03))
        try:
            db.add_or_reinforce_triple(a, 'latent_related_to', b, confidence=conf)
            new_triples += 1
        except Exception:
            pass

    st['concepts'] = dict(sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:1500])
    st['edges'] = dict(sorted(edges.items(), key=lambda x: x[1], reverse=True)[:3000])
    st['last_run_at'] = int(time.time())
    _save_state(st)

    return {
        'scanned': scanned,
        'concepts_total': len(st['concepts']),
        'edges_total': len(st['edges']),
        'top_concepts': [{'term': c, 'freq': f} for c, f in top_concepts[:10]],
        'top_edges': [{'pair': k, 'weight': w} for k, w in top_edges[:10]],
        'triples_induced': new_triples,
        'last_run_at': st['last_run_at'],
    }
