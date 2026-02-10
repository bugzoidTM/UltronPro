#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path
import sys

REPO = Path('/root/.openclaw/workspace/UltronPro')
SERVICE = 'ultronpro_ultronpro'
CONTAINER_FILTER = 'ultronpro_ultronpro'


def sh(cmd: str, check=True):
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\nstdout={p.stdout}\nstderr={p.stderr}")
    return p.stdout.strip()


def get_container_id() -> str:
    return sh(f"docker ps --filter name={CONTAINER_FILTER} -q | head -n1")


def list_pending(cid: str):
    out = sh(f"docker exec {cid} sh -lc 'ls -1 /app/data/selfpatch_pending/*.json 2>/dev/null || true'", check=False)
    if not out.strip():
        return []
    return [x.strip() for x in out.splitlines() if x.strip()]


def map_target(file_path: str) -> Path:
    if file_path.startswith('/app/ultronpro/'):
        rel = file_path[len('/app/ultronpro/'):]
        return REPO / 'backend' / 'ultronpro' / rel
    if file_path.startswith('/app/ui/'):
        rel = file_path[len('/app/ui/'):]
        return REPO / 'backend' / 'ui' / rel
    raise ValueError(f'unsupported file_path: {file_path}')


def apply_patch_one(cjson: dict):
    fp = cjson['file_path']
    old = cjson['old_text']
    new = cjson['new_text']

    target = map_target(fp)
    if not target.exists():
        raise RuntimeError(f'target file missing: {target}')

    txt = target.read_text()
    if old not in txt:
        raise RuntimeError(f'old_text not found in {target}')

    bak = target.with_suffix(target.suffix + '.bak.hostpatch')
    bak.write_text(txt)
    target.write_text(txt.replace(old, new, 1))
    return str(target)


def smoke_compile(changed_files):
    py_files = [f for f in changed_files if f.endswith('.py')]
    if not py_files:
        return
    cmd = 'python3 -m py_compile ' + ' '.join(py_files)
    sh(cmd)


def deploy():
    sh(f"docker build -t ultronpro_backend:local -f {REPO}/backend/Dockerfile {REPO}/backend")
    sh(f"docker service update --force {SERVICE}")


def remove_pending(cid: str, path: str):
    sh(f"docker exec {cid} sh -lc 'rm -f {path}'", check=False)


def main():
    cid = get_container_id()
    if not cid:
        print('No running ultronpro container found.')
        return 0

    pendings = list_pending(cid)
    if not pendings:
        print('No pending selfpatch files.')
        return 0

    changed = []
    applied = 0
    for p in pendings:
        raw = sh(f"docker exec {cid} sh -lc 'cat {p}'")
        data = json.loads(raw)
        try:
            f = apply_patch_one(data)
            changed.append(f)
            remove_pending(cid, p)
            applied += 1
            print(f'Applied pending patch -> {f}')
        except Exception as e:
            print(f'Failed pending patch {p}: {e}')

    if not changed:
        print('No patch applied.')
        return 1

    smoke_compile(changed)
    deploy()
    print(f'Applied {applied} patch(es), smoke passed, deploy triggered.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f'ERROR: {e}')
        raise SystemExit(1)
