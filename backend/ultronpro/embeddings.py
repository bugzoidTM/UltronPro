"""Embeddings module using sentence-transformers (all-MiniLM-L6-v2).

Provides semantic search over experiences and triples.
Model is loaded lazily on first use to avoid startup delay.
"""
from __future__ import annotations

import json
import numpy as np
from typing import Any

_model = None
_model_name = "all-MiniLM-L6-v2"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name)
    return _model


def embed_text(text: str) -> list[float]:
    """Return embedding vector for a single text."""
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True)
    return vec.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts."""
    if not texts:
        return []
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    denom = (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    if denom == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / denom)


def search_similar(
    query_vec: list[float],
    candidates: list[dict[str, Any]],
    vec_key: str = "embedding",
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[tuple[dict[str, Any], float]]:
    """Search candidates by cosine similarity to query vector.
    
    Returns list of (candidate, score) tuples sorted by score descending.
    """
    results = []
    for c in candidates:
        vec = c.get(vec_key)
        if not vec:
            continue
        score = cosine_similarity(query_vec, vec)
        if score >= min_score:
            results.append((c, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def embedding_to_json(vec: list[float]) -> str:
    """Serialize embedding to JSON for storage."""
    return json.dumps(vec)


def embedding_from_json(s: str | None) -> list[float] | None:
    """Deserialize embedding from JSON."""
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None
