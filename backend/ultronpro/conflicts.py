from __future__ import annotations

from typing import Any

from ultronpro import store
from ultronpro.conflict_resolver import ConflictResolver


_resolver: ConflictResolver | None = None


def _get_resolver() -> ConflictResolver:
    global _resolver
    if _resolver is None:
        _resolver = ConflictResolver(store.db)
    return _resolver


def list_conflicts(status: str = "open", limit: int = 20) -> list[dict[str, Any]]:
    return store.db.list_conflicts(status=status, limit=limit)


def get_conflict(conflict_id: int) -> dict[str, Any] | None:
    return store.db.get_conflict(int(conflict_id))


def resolve_manual(
    conflict_id: int,
    chosen_object: str,
    decided_by: str | None = None,
    resolution: str | None = None,
) -> bool:
    if not chosen_object:
        return False
    store.db.resolve_conflict(
        int(conflict_id),
        resolution=resolution,
        chosen_object=chosen_object,
        decided_by=decided_by,
    )
    return True


def archive(conflict_id: int):
    store.db.archive_conflict(int(conflict_id))


async def auto_resolve_all(limit: int = 3) -> list[dict[str, Any]]:
    resolver = _get_resolver()
    return resolver.resolve_pending(max_conflicts=max(1, int(limit)))
