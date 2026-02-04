from __future__ import annotations

from typing import Any


def export_bundle(store, since_id: int | None = None, limit: int = 200) -> dict[str, Any]:
    """Export a batch of learning artifacts.

    MVP: no signatures, no encryption. Intended for future federated batch sync.
    """
    import sqlite3

    with sqlite3.connect(str(store.path)) as c:
        c.row_factory = sqlite3.Row
        if since_id:
            exps = c.execute(
                "SELECT id, created_at, user_id, text FROM experiences WHERE id > ? ORDER BY id ASC LIMIT ?",
                (since_id, limit),
            ).fetchall()
        else:
            exps = c.execute(
                "SELECT id, created_at, user_id, text FROM experiences ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()[::-1]

        tris = c.execute(
            "SELECT id, created_at, updated_at, subject, predicate, object, confidence, support_count, contradict_count FROM triples ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()[::-1]

    return {
        "schema": 1,
        "experiences": [dict(r) for r in exps],
        "triples": [dict(r) for r in tris],
    }


def import_bundle(store, bundle: dict[str, Any], source: str = "federated") -> dict[str, Any]:
    """Import a batch.

    MVP strategy:
    - experiences: append as new experiences (provenance tagged)
    - triples: reinforce triple with a small confidence bump
    """
    exps = bundle.get("experiences") or []
    tris = bundle.get("triples") or []

    exp_ids = []
    for e in exps[:500]:
        txt = (e.get("text") or "").strip()
        if not txt:
            continue
        exp_ids.append(store.add_experience(user_id=source, text=txt))

    for t in tris[:1000]:
        s = (t.get("subject") or "").strip()
        p = (t.get("predicate") or "").strip()
        o = (t.get("object") or "").strip()
        if not (s and p and o):
            continue
        store.add_or_reinforce_triple(s, p, o, confidence=float(t.get("confidence") or 0.5), note=f"import:{source}")

    return {"experiences_imported": len(exp_ids), "triples_imported": len(tris)}
