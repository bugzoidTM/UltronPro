from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time
import hashlib

PATH = Path('/app/data/identity_daily.json')


def _load() -> dict[str, Any]:
    if PATH.exists():
        try:
            d = json.loads(PATH.read_text(encoding='utf-8'))
            if isinstance(d, dict):
                d.setdefault('entries', [])
                d.setdefault('pending_promises', [])
                d.setdefault('last_review_day', '')
                return d
        except Exception:
            pass
    return {'updated_at': int(time.time()), 'entries': [], 'pending_promises': [], 'last_review_day': ''}


def _save(d: dict[str, Any]) -> None:
    d['updated_at'] = int(time.time())
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')


def add_promise(text: str, source: str = 'system') -> dict[str, Any]:
    d = _load()
    p = {
        'id': f"prm_{int(time.time()*1000)}",
        'ts': int(time.time()),
        'text': str(text or '')[:220],
        'source': str(source or 'system')[:40],
        'status': 'pending',
    }
    arr = list(d.get('pending_promises') or [])
    arr.append(p)
    d['pending_promises'] = arr[-200:]
    _save(d)
    return p


def _day_key(ts: float | None = None) -> str:
    t = time.localtime(ts or time.time())
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"


def due_daily_review(hour_local: int = 23, now_ts: float | None = None) -> bool:
    d = _load()
    now = now_ts or time.time()
    k = _day_key(now)
    tm = time.localtime(now)
    if tm.tm_hour != int(hour_local):
        return False
    if str(d.get('last_review_day') or '') == k:
        return False
    return True


def run_daily_review(completed_hints: list[str] | None = None, failed_hints: list[str] | None = None, protocol_update: str = '') -> dict[str, Any]:
    d = _load()
    pending = list(d.get('pending_promises') or [])
    completed_hints = [str(x).lower() for x in (completed_hints or [])]
    failed_hints = [str(x).lower() for x in (failed_hints or [])]

    done = []
    failed = []
    carry = []
    for p in pending:
        txt = str(p.get('text') or '').lower()
        if any(h and h in txt for h in completed_hints):
            p['status'] = 'done'
            done.append(p)
        elif any(h and h in txt for h in failed_hints):
            p['status'] = 'failed'
            failed.append(p)
        else:
            carry.append(p)

    day = _day_key()
    checksum_src = '|'.join([str(x.get('id')) + ':' + str(x.get('status')) for x in (done + failed + carry)]) + '|' + str(protocol_update)
    checksum = hashlib.sha256(checksum_src.encode('utf-8')).hexdigest()[:16]

    entry = {
        'id': f"idr_{int(time.time()*1000)}",
        'day': day,
        'ts': int(time.time()),
        'promises_done': done[-30:],
        'promises_failed': failed[-30:],
        'promises_carry': carry[-60:],
        'protocol_update': str(protocol_update or '')[:400],
        'checksum': checksum,
    }

    arr = list(d.get('entries') or [])
    arr.append(entry)
    d['entries'] = arr[-400:]
    d['pending_promises'] = carry[-200:]
    d['last_review_day'] = day
    _save(d)
    return {'ok': True, 'entry': entry, 'path': str(PATH)}


def status(limit: int = 20) -> dict[str, Any]:
    d = _load()
    return {
        'ok': True,
        'pending_promises': (d.get('pending_promises') or [])[-60:],
        'entries': (d.get('entries') or [])[-max(1, int(limit)):],
        'last_review_day': d.get('last_review_day'),
        'path': str(PATH),
    }
