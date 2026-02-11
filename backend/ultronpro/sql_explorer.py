from __future__ import annotations

import os
import re
import sqlite3
from typing import Any

DB_PATH = os.getenv('ULTRONPRO_DB_PATH', '/app/data/ultron.db')

_DENY_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|vacuum|reindex|analyze|truncate|grant|revoke|begin|commit|rollback)\b",
    re.IGNORECASE,
)


def _connect_ro() -> sqlite3.Connection:
    uri = f"file:{DB_PATH}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    return con


def _validate_query(query: str) -> str:
    q = (query or '').strip()
    if not q:
        raise ValueError('empty_query')
    if q.count(';') > 1:
        raise ValueError('multiple_statements_not_allowed')

    lower = q.lower().lstrip()
    if not (lower.startswith('select') or lower.startswith('pragma') or lower.startswith('explain') or lower.startswith('with')):
        raise ValueError('only_select_pragma_explain_with_allowed')
    if _DENY_RE.search(q):
        raise ValueError('mutation_sql_not_allowed')
    return q


def list_tables() -> dict[str, Any]:
    with _connect_ro() as con:
        rows = con.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table','view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
    return {
        'ok': True,
        'db_path': DB_PATH,
        'tables': [dict(r) for r in rows],
    }


def describe_table(name: str) -> dict[str, Any]:
    t = (name or '').strip()
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', t):
        return {'ok': False, 'error': 'invalid_table_name'}

    with _connect_ro() as con:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=? LIMIT 1",
            (t,),
        ).fetchone()
        if not exists:
            return {'ok': False, 'error': 'table_not_found', 'table': t}

        cols = con.execute(f'PRAGMA table_info("{t}")').fetchall()
        sample = con.execute(f'SELECT * FROM "{t}" LIMIT 5').fetchall()

    return {
        'ok': True,
        'table': t,
        'columns': [dict(c) for c in cols],
        'sample_rows': [dict(r) for r in sample],
    }


def execute_sql(query: str, limit: int = 200) -> dict[str, Any]:
    q = _validate_query(query)
    lim = max(1, min(int(limit or 200), 1000))

    sql = q
    if re.match(r'^\s*select\b', q, re.IGNORECASE) and ' limit ' not in q.lower():
        sql = q.rstrip().rstrip(';') + f' LIMIT {lim}'

    with _connect_ro() as con:
        cur = con.execute(sql)
        rows = cur.fetchmany(lim)
        cols = [d[0] for d in (cur.description or [])]

    return {
        'ok': True,
        'limit': lim,
        'columns': cols,
        'rows': [dict(r) if isinstance(r, sqlite3.Row) else r for r in rows],
        'row_count': len(rows),
        'applied_sql': sql,
    }
