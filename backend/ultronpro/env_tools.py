from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import subprocess
import time

SANDBOX_ROOT = Path('/app/data/sandbox')
HISTORY_PATH = SANDBOX_ROOT / 'history.json'


def _safe_name(name: str) -> str:
    s = ''.join(ch for ch in (name or '') if ch.isalnum() or ch in ('_', '-', '.')).strip('.')
    return s or f'file_{int(time.time())}.txt'


def _load_history() -> list[dict[str, Any]]:
    try:
        if HISTORY_PATH.exists():
            d = json.loads(HISTORY_PATH.read_text())
            if isinstance(d, list):
                return d
    except Exception:
        pass
    return []


def _save_history(items: list[dict[str, Any]]):
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(items[-400:], ensure_ascii=False, indent=2))


def write_file(rel_path: str, content: str) -> dict[str, Any]:
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    p = SANDBOX_ROOT / _safe_name(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content or '')
    return {'ok': True, 'path': str(p), 'bytes': len(content or '')}


def read_file(rel_path: str) -> dict[str, Any]:
    p = SANDBOX_ROOT / _safe_name(rel_path)
    if not p.exists():
        return {'ok': False, 'error': 'not_found'}
    txt = p.read_text()
    return {'ok': True, 'path': str(p), 'content': txt[:12000]}


def list_files(limit: int = 100) -> dict[str, Any]:
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    out = []
    for p in SANDBOX_ROOT.rglob('*'):
        if p.is_file() and p.name != 'history.json':
            out.append({'path': str(p), 'size': p.stat().st_size, 'mtime': int(p.stat().st_mtime)})
    out.sort(key=lambda x: x['mtime'], reverse=True)
    return {'items': out[:max(1, int(limit))]}


def run_python(code: str | None = None, file_path: str | None = None, timeout_sec: int = 15) -> dict[str, Any]:
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    run_dir = SANDBOX_ROOT / f'run_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)

    target = None
    if file_path:
        fp = SANDBOX_ROOT / _safe_name(file_path)
        if not fp.exists():
            return {'ok': False, 'error': 'file_not_found', 'path': str(fp)}
        target = fp
    else:
        target = run_dir / 'main.py'
        target.write_text(code or '')

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    # discourage user-site and env leakage
    env['PYTHONNOUSERSITE'] = '1'

    started = time.time()
    try:
        proc = subprocess.run(
            ['python3', '-I', str(target)],
            cwd=str(run_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, min(120, int(timeout_sec))),
        )
        out = {
            'ok': proc.returncode == 0,
            'returncode': proc.returncode,
            'stdout': (proc.stdout or '')[:12000],
            'stderr': (proc.stderr or '')[:12000],
            'elapsed_sec': round(time.time() - started, 3),
            'target': str(target),
        }
    except subprocess.TimeoutExpired as e:
        out = {
            'ok': False,
            'returncode': -9,
            'stdout': (e.stdout or '')[:12000] if isinstance(e.stdout, str) else '',
            'stderr': ((e.stderr or '') if isinstance(e.stderr, str) else '')[:12000],
            'elapsed_sec': round(time.time() - started, 3),
            'target': str(target),
            'error': 'timeout',
        }

    h = _load_history()
    h.append({'ts': ts, 'kind': 'python_run', 'target': out.get('target'), 'ok': out.get('ok'), 'returncode': out.get('returncode'), 'elapsed_sec': out.get('elapsed_sec')})
    _save_history(h)
    return out


def history(limit: int = 50) -> dict[str, Any]:
    arr = _load_history()
    return {'items': arr[-max(1, int(limit)):]}