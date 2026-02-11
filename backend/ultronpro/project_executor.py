from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import sqlite3
import statistics
import time

EXPERIMENTS_PATH = Path('/app/data/project_experiments.json')
DB_PATH = os.getenv('ULTRONPRO_DB_PATH', '/app/data/ultron.db')


def _load() -> list[dict[str, Any]]:
    try:
        if EXPERIMENTS_PATH.exists():
            d = json.loads(EXPERIMENTS_PATH.read_text())
            if isinstance(d, list):
                return d
    except Exception:
        pass
    return []


def _save(items: list[dict[str, Any]]):
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_PATH.write_text(json.dumps(items[-400:], ensure_ascii=False, indent=2))


def list_experiments(project_id: str | None = None, limit: int = 40) -> list[dict[str, Any]]:
    arr = _load()
    if project_id:
        arr = [x for x in arr if str(x.get('project_id') or '') == str(project_id)]
    return arr[-max(1, int(limit)):]


def propose_experiment(project: dict[str, Any], brief: dict[str, Any] | None = None) -> dict[str, Any]:
    pid = str(project.get('id') or '')
    ts = int(time.time())
    blockers = (brief or {}).get('blockers') or []
    return {
        'id': f'exp_{ts}_{len(_load())+1}',
        'project_id': pid,
        'created_at': ts,
        'status': 'planned',
        'kind': 'sqlite_benchmark',
        'hypothesis': 'Pequenas otimizações de consulta e índices reduzem latência de leitura.',
        'target_metric': 'p95_read_ms',
        'target_value': 20.0,
        'blocker_hint': blockers[:2],
        'steps': [
            'Coletar baseline de latência em consultas de leitura comuns',
            'Inspecionar índice e query plan',
            'Gerar script de otimização sugerida',
            'Reavaliar latência e decidir promoção',
        ],
    }


def _safe_benchmark_sqlite() -> dict[str, Any]:
    start = time.time()
    if not Path(DB_PATH).exists():
        return {'ok': False, 'error': f'db_not_found:{DB_PATH}'}

    lat = []
    plans = []
    tables = []
    indexes = []

    with sqlite3.connect(DB_PATH, timeout=10) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 25").fetchall()
        tables = [str(r[0]) for r in rows]

        idx = c.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index' ORDER BY tbl_name, name LIMIT 60").fetchall()
        indexes = [{'name': str(r[0]), 'table': str(r[1])} for r in idx]

        # representative read query
        q = "SELECT id, subject, predicate, object FROM triples ORDER BY id DESC LIMIT 50"
        for _ in range(30):
            t0 = time.perf_counter()
            c.execute(q).fetchall()
            lat.append((time.perf_counter() - t0) * 1000.0)

        try:
            p = c.execute(f"EXPLAIN QUERY PLAN {q}").fetchall()
            plans = [dict(x) for x in p]
        except Exception:
            plans = []

    p50 = statistics.median(lat) if lat else 0.0
    p95 = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)] if lat else 0.0

    return {
        'ok': True,
        'elapsed_sec': round(time.time() - start, 3),
        'samples': len(lat),
        'p50_read_ms': round(float(p50), 3),
        'p95_read_ms': round(float(p95), 3),
        'tables': tables,
        'indexes': indexes[:25],
        'query_plan': plans[:20],
    }


def _suggest_sql_artifact(metrics: dict[str, Any], project_id: str) -> str:
    out_dir = Path('/app/data/procedure_artifacts')
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    fp = out_dir / f'project_{project_id}_{ts}_db_optimize.sql'
    sql = """
-- Generated optimization suggestions (safe review first)
PRAGMA optimize;
ANALYZE;
-- Consider index review for frequent filters/order patterns
-- CREATE INDEX IF NOT EXISTS idx_triples_sp ON triples(subject, predicate);
""".strip() + "\n"
    fp.write_text(sql)
    return str(fp)


def run_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    kind = str(exp.get('kind') or '')
    if kind != 'sqlite_benchmark':
        return {'ok': False, 'error': 'unsupported_experiment_kind'}

    m = _safe_benchmark_sqlite()
    if not m.get('ok'):
        return {'ok': False, 'metrics': m, 'status': 'failed'}

    target = float(exp.get('target_value') or 20.0)
    p95 = float(m.get('p95_read_ms') or 9999.0)
    success = p95 <= target

    artifact = _suggest_sql_artifact(m, str(exp.get('project_id') or 'unknown'))
    return {
        'ok': True,
        'status': 'success' if success else 'needs_optimization',
        'success': success,
        'metrics': m,
        'artifact': artifact,
        'recommendations': [
            'apply pragma optimize/analyze in maintenance window',
            'review indexes for subject/predicate filters',
            're-run benchmark after changes',
        ] if not success else ['keep current profile and monitor p95 over time'],
    }


def record(experiment: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    item = dict(experiment)
    item['updated_at'] = int(time.time())
    item['status'] = result.get('status') or ('done' if result.get('ok') else 'failed')
    item['result'] = result

    arr = _load()
    arr.append(item)
    _save(arr)
    return item
