from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


def _ts() -> float:
    return time.time()


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        # timeout reduces "database is locked" under concurrent loop + API usage
        c = sqlite3.connect(str(self.path), timeout=30)
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=5000")
        except Exception:
            pass
        return c

    def _init(self):
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sources(
                  id TEXT PRIMARY KEY,
                  created_at REAL NOT NULL,
                  updated_at REAL,
                  kind TEXT,
                  label TEXT,
                  trust REAL NOT NULL DEFAULT 0.5,
                  support_count INTEGER NOT NULL DEFAULT 0,
                  contradict_count INTEGER NOT NULL DEFAULT 0,
                  notes TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  processed_at REAL,
                  user_id TEXT,
                  source_id TEXT,
                  modality TEXT NOT NULL DEFAULT 'text',
                  text TEXT,
                  blob_path TEXT,
                  mime TEXT,
                  embedding_json TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS laws(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  updated_at REAL,
                  status TEXT NOT NULL, -- active|archived
                  core INTEGER NOT NULL DEFAULT 1,
                  title TEXT,
                  text TEXT NOT NULL,
                  source_id TEXT,
                  source_experience_id INTEGER
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS events(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  kind TEXT NOT NULL,
                  text TEXT NOT NULL,
                  meta_json TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS insights(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  kind TEXT NOT NULL,
                  priority INTEGER NOT NULL DEFAULT 3,
                  title TEXT NOT NULL,
                  text TEXT NOT NULL,
                  conflict_id INTEGER,
                  source_id TEXT,
                  meta_json TEXT
                )
                """
            )
            # migrations for events are unnecessary (new table)
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS actions(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  updated_at REAL,
                  status TEXT NOT NULL, -- queued|running|done|blocked|error|expired
                  kind TEXT NOT NULL,
                  text TEXT NOT NULL,
                  priority INTEGER NOT NULL DEFAULT 0,
                  policy_allowed INTEGER,
                  policy_score REAL,
                  last_error TEXT,
                  meta_json TEXT,
                  expires_at REAL,
                  cooldown_key TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS goals(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  status TEXT NOT NULL, -- open|active|done|archived
                  title TEXT NOT NULL,
                  description TEXT,
                  priority INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_milestones(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  goal_id INTEGER NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL,
                  week_index INTEGER NOT NULL DEFAULT 1,
                  title TEXT NOT NULL,
                  progress_criteria TEXT,
                  status TEXT NOT NULL DEFAULT 'open',
                  progress REAL NOT NULL DEFAULT 0.0,
                  FOREIGN KEY(goal_id) REFERENCES goals(id)
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS conflicts(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  updated_at REAL,
                  status TEXT NOT NULL, -- open|resolved|archived
                  subject TEXT NOT NULL,
                  predicate TEXT NOT NULL,
                  key TEXT NOT NULL, -- subject\u241Fp\u241F for uniqueness
                  first_seen_at REAL NOT NULL,
                  last_seen_at REAL NOT NULL,
                  seen_count INTEGER NOT NULL DEFAULT 1,
                  last_summary TEXT,
                  last_question_at REAL,
                  question_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS conflict_variants(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conflict_id INTEGER NOT NULL,
                  triple_id INTEGER,
                  object TEXT,
                  confidence REAL,
                  first_seen_at REAL NOT NULL,
                  last_seen_at REAL NOT NULL,
                  seen_count INTEGER NOT NULL DEFAULT 1,
                  FOREIGN KEY(conflict_id) REFERENCES conflicts(id)
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS conflict_resolutions(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conflict_id INTEGER NOT NULL,
                  created_at REAL NOT NULL,
                  decided_by TEXT,
                  chosen_object TEXT,
                  resolution_text TEXT,
                  notes TEXT,
                  FOREIGN KEY(conflict_id) REFERENCES conflicts(id)
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS questions(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  status TEXT NOT NULL, -- open|answered|dismissed
                  question TEXT NOT NULL,
                  context TEXT,
                  priority INTEGER NOT NULL DEFAULT 0,
                  answered_at REAL,
                  answer TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS triples(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  updated_at REAL,
                  subject TEXT NOT NULL,
                  predicate TEXT NOT NULL,
                  object TEXT NOT NULL,
                  confidence REAL NOT NULL DEFAULT 0.5,
                  support_count INTEGER NOT NULL DEFAULT 1,
                  contradict_count INTEGER NOT NULL DEFAULT 0,
                  embedding_json TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS triple_evidence(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  triple_id INTEGER NOT NULL,
                  experience_id INTEGER,
                  note TEXT,
                  created_at REAL NOT NULL,
                  FOREIGN KEY(triple_id) REFERENCES triples(id)
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS procedures(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  updated_at REAL,
                  status TEXT NOT NULL DEFAULT 'active',
                  domain TEXT,
                  proc_type TEXT NOT NULL DEFAULT 'analysis',
                  name TEXT NOT NULL,
                  goal TEXT,
                  preconditions TEXT,
                  steps_json TEXT NOT NULL,
                  success_criteria TEXT,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  successes INTEGER NOT NULL DEFAULT 0,
                  avg_score REAL NOT NULL DEFAULT 0.0,
                  source_experience_id INTEGER
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS procedure_runs(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  procedure_id INTEGER NOT NULL,
                  created_at REAL NOT NULL,
                  input_text TEXT,
                  output_text TEXT,
                  score REAL,
                  success INTEGER,
                  notes TEXT,
                  FOREIGN KEY(procedure_id) REFERENCES procedures(id)
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS analogies(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  updated_at REAL,
                  status TEXT NOT NULL DEFAULT 'hypothesis',
                  source_domain TEXT,
                  target_domain TEXT,
                  source_concept TEXT,
                  target_concept TEXT,
                  mapping_json TEXT,
                  inference_rule TEXT,
                  confidence REAL NOT NULL DEFAULT 0.5,
                  evidence_refs_json TEXT,
                  notes TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS global_workspace(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at REAL NOT NULL,
                  module TEXT NOT NULL,
                  channel TEXT NOT NULL,
                  payload_json TEXT,
                  salience REAL NOT NULL DEFAULT 0.5,
                  ttl_sec INTEGER NOT NULL DEFAULT 900,
                  expires_at REAL,
                  consumed_by_json TEXT
                )
                """
            )

            # lightweight migrations (add columns if upgrading)

            # migrations for laws
            if c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='laws'").fetchone():
                law_cols = {r[1] for r in c.execute("PRAGMA table_info(laws)").fetchall()}
                if "core" not in law_cols:
                    try:
                        c.execute("ALTER TABLE laws ADD COLUMN core INTEGER NOT NULL DEFAULT 1")
                    except Exception:
                        pass
                if "source_experience_id" not in law_cols:
                    try:
                        c.execute("ALTER TABLE laws ADD COLUMN source_experience_id INTEGER")
                    except Exception:
                        pass

            # migrations for sources
            src_cols = {r[1] for r in c.execute("PRAGMA table_info(sources)").fetchall()}
            for col, ddl in [
                ("updated_at", "ALTER TABLE sources ADD COLUMN updated_at REAL"),
                ("support_count", "ALTER TABLE sources ADD COLUMN support_count INTEGER NOT NULL DEFAULT 0"),
                ("contradict_count", "ALTER TABLE sources ADD COLUMN contradict_count INTEGER NOT NULL DEFAULT 0"),
            ]:
                if col not in src_cols:
                    try:
                        c.execute(ddl)
                    except Exception:
                        pass

            # migrations for experiences
            exp_cols = {r[1] for r in c.execute("PRAGMA table_info(experiences)").fetchall()}
            for col, ddl in [
                ("processed_at", "ALTER TABLE experiences ADD COLUMN processed_at REAL"),
                ("source_id", "ALTER TABLE experiences ADD COLUMN source_id TEXT"),
                ("modality", "ALTER TABLE experiences ADD COLUMN modality TEXT NOT NULL DEFAULT 'text'"),
                ("blob_path", "ALTER TABLE experiences ADD COLUMN blob_path TEXT"),
                ("mime", "ALTER TABLE experiences ADD COLUMN mime TEXT"),
                ("embedding_json", "ALTER TABLE experiences ADD COLUMN embedding_json TEXT"),
                ("curated_at", "ALTER TABLE experiences ADD COLUMN curated_at REAL"),
                ("archived_at", "ALTER TABLE experiences ADD COLUMN archived_at REAL"),
                ("utility_score", "ALTER TABLE experiences ADD COLUMN utility_score REAL"),
            ]:
                if col not in exp_cols:
                    try:
                        c.execute(ddl)
                    except Exception:
                        pass

            # migrations for actions
            act_cols = {r[1] for r in c.execute("PRAGMA table_info(actions)").fetchall()}
            for col, ddl in [
                ("expires_at", "ALTER TABLE actions ADD COLUMN expires_at REAL"),
                ("cooldown_key", "ALTER TABLE actions ADD COLUMN cooldown_key TEXT"),
            ]:
                if col not in act_cols:
                    try:
                        c.execute(ddl)
                    except Exception:
                        pass

            # migrations for procedures
            if c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='procedures'").fetchone():
                pcols = {r[1] for r in c.execute("PRAGMA table_info(procedures)").fetchall()}
                if "proc_type" not in pcols:
                    try:
                        c.execute("ALTER TABLE procedures ADD COLUMN proc_type TEXT NOT NULL DEFAULT 'analysis'")
                    except Exception:
                        pass

            # migrations for conflicts
            if c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conflicts'").fetchone():
                conf_cols = {r[1] for r in c.execute("PRAGMA table_info(conflicts)").fetchall()}
                for col, ddl in [
                    ("last_question_at", "ALTER TABLE conflicts ADD COLUMN last_question_at REAL"),
                    ("question_count", "ALTER TABLE conflicts ADD COLUMN question_count INTEGER NOT NULL DEFAULT 0"),
                ]:
                    if col not in conf_cols:
                        try:
                            c.execute(ddl)
                        except Exception:
                            pass

            cols = {r[1] for r in c.execute("PRAGMA table_info(triples)").fetchall()}
            if "updated_at" not in cols:
                try:
                    c.execute("ALTER TABLE triples ADD COLUMN updated_at REAL")
                except Exception:
                    pass
            if "support_count" not in cols:
                try:
                    c.execute("ALTER TABLE triples ADD COLUMN support_count INTEGER NOT NULL DEFAULT 1")
                except Exception:
                    pass
            if "contradict_count" not in cols:
                try:
                    c.execute("ALTER TABLE triples ADD COLUMN contradict_count INTEGER NOT NULL DEFAULT 0")
                except Exception:
                    pass
            if "embedding_json" not in cols:
                try:
                    c.execute("ALTER TABLE triples ADD COLUMN embedding_json TEXT")
                except Exception:
                    pass

    # --- experiences
    def _ensure_source_conn(
        self,
        c: sqlite3.Connection,
        source_id: str,
        kind: str | None = None,
        label: str | None = None,
        trust: float | None = None,
    ):
        row = c.execute("SELECT id FROM sources WHERE id=?", (source_id,)).fetchone()
        if row:
            return
        c.execute(
            "INSERT INTO sources(id, created_at, updated_at, kind, label, trust, support_count, contradict_count, notes) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                source_id,
                _ts(),
                _ts(),
                kind,
                label,
                float(trust) if trust is not None else 0.5,
                0,
                0,
                None,
            ),
        )

    def ensure_source(
        self,
        source_id: str,
        kind: str | None = None,
        label: str | None = None,
        trust: float | None = None,
    ):
        with self._conn() as c:
            self._ensure_source_conn(c, source_id, kind=kind, label=label, trust=trust)

    def _recompute_source_trust(self, c: sqlite3.Connection, source_id: str):
        row = c.execute(
            "SELECT support_count, contradict_count FROM sources WHERE id=?",
            (source_id,),
        ).fetchone()
        if not row:
            return
        sup = int(row[0] or 0)
        con = int(row[1] or 0)
        # Jeffreys-ish prior: (sup+1)/(sup+con+2)
        trust = (sup + 1.0) / (sup + con + 2.0)
        trust = max(0.05, min(0.95, trust))
        c.execute(
            "UPDATE sources SET trust=?, updated_at=? WHERE id=?",
            (trust, _ts(), source_id),
        )

    def source_bump_support(self, source_id: str, n: int = 1):
        if not source_id:
            return
        with self._conn() as c:
            self.ensure_source(source_id)
            c.execute(
                "UPDATE sources SET support_count = support_count + ?, updated_at=? WHERE id=?",
                (int(n), _ts(), source_id),
            )
            self._recompute_source_trust(c, source_id)

    def source_bump_contradict(self, source_id: str, n: int = 1):
        if not source_id:
            return
        with self._conn() as c:
            self.ensure_source(source_id)
            c.execute(
                "UPDATE sources SET contradict_count = contradict_count + ?, updated_at=? WHERE id=?",
                (int(n), _ts(), source_id),
            )
            self._recompute_source_trust(c, source_id)

    def list_sources(self, limit: int = 100, order_by: str = "trust") -> list[dict[str, Any]]:
        """Lista todas as fontes com suas estatísticas de confiança."""
        order = "trust DESC" if order_by == "trust" else "updated_at DESC"
        with self._conn() as c:
            rows = c.execute(
                f"""
                SELECT id, created_at, updated_at, kind, label, trust, 
                       support_count, contradict_count, notes
                FROM sources
                ORDER BY {order}, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        """Busca uma fonte específica."""
        with self._conn() as c:
            row = c.execute(
                """
                SELECT id, created_at, updated_at, kind, label, trust, 
                       support_count, contradict_count, notes
                FROM sources
                WHERE id=?
                """,
                (source_id,),
            ).fetchone()
        return dict(row) if row else None

    def add_experience(
        self,
        user_id: str | None,
        text: str | None,
        source_id: str | None = None,
        modality: str = "text",
        blob_path: str | None = None,
        mime: str | None = None,
        embedding_json: str | None = None,
    ) -> int:
        with self._conn() as c:
            if source_id:
                kind = "lightrag" if str(source_id).startswith("lightrag:") else "feed"
                self._ensure_source_conn(c, str(source_id), kind=kind, label=str(source_id))
            cur = c.execute(
                "INSERT INTO experiences(created_at, processed_at, user_id, source_id, modality, text, blob_path, mime, embedding_json) VALUES(?,?,?,?,?,?,?,?,?)",
                (_ts(), None, user_id, source_id, modality, text, blob_path, mime, embedding_json),
            )
            return int(cur.lastrowid)

    def rebuild_sources_from_experiences(self, limit: int = 5000) -> int:
        """Backfill de fontes a partir das experiências antigas."""
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT DISTINCT source_id
                FROM experiences
                WHERE source_id IS NOT NULL AND trim(source_id) <> ''
                ORDER BY source_id
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            added = 0
            for r in rows:
                sid = str(r[0])
                ex = c.execute("SELECT id FROM sources WHERE id=?", (sid,)).fetchone()
                if ex:
                    continue
                kind = "lightrag" if sid.startswith("lightrag:") else "feed"
                self._ensure_source_conn(c, sid, kind=kind, label=sid)
                added += 1
            return added

    def update_experience_embedding(self, eid: int, embedding_json: str):
        with self._conn() as c:
            c.execute("UPDATE experiences SET embedding_json=? WHERE id=?", (embedding_json, int(eid)))

    def list_experiences(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, created_at, processed_at, curated_at, user_id, source_id, modality, text, blob_path, mime, embedding_json FROM experiences ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows][::-1]

    def count_uncurated_experiences(self) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM experiences WHERE archived_at IS NULL AND curated_at IS NULL AND text IS NOT NULL AND length(text) > 40"
            ).fetchone()
        return int(row[0] or 0)

    def list_uncurated_experiences(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, created_at, source_id, modality, text
                FROM experiences
                WHERE archived_at IS NULL
                  AND curated_at IS NULL
                  AND text IS NOT NULL
                  AND length(text) > 40
                ORDER BY id ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_archived_experiences(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) FROM experiences WHERE archived_at IS NOT NULL").fetchone()
        return int(row[0] or 0)

    def count_distilled_experiences(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) FROM experiences WHERE modality='distilled'").fetchone()
        return int(row[0] or 0)

    def mark_experiences_curated(self, ids: list[int]):
        if not ids:
            return
        now = _ts()
        with self._conn() as c:
            for eid in ids:
                c.execute("UPDATE experiences SET curated_at=? WHERE id=?", (now, int(eid)))

    def prune_low_utility_experiences(self, limit: int = 200, focus_terms: list[str] | None = None) -> int:
        """Arquiva ruído textual antigo e curto para reduzir carga cognitiva.

        Se focus_terms vier, aumenta utilidade de itens relacionados ao objetivo ativo.
        """
        now = _ts()
        terms = [t.lower().strip() for t in (focus_terms or []) if t and len(t.strip()) >= 3][:12]

        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, text, source_id, modality
                FROM experiences
                WHERE archived_at IS NULL
                  AND text IS NOT NULL
                  AND modality IN ('text','answer','file')
                ORDER BY id ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

            pruned = 0
            for r in rows:
                eid = int(r[0])
                txt = (r[1] or '').strip()
                sid = (r[2] or '')
                mod = (r[3] or 'text')
                txtl = txt.lower()

                # heurística simples de utilidade
                utility = 0.5
                if len(txt) < 80:
                    utility -= 0.25
                if sid.startswith('uselessfacts') or sid.startswith('quotable'):
                    utility -= 0.25
                if mod == 'answer' and len(txt) < 40:
                    utility -= 0.2

                # bônus se relacionado ao objetivo ativo
                if terms and any(t in txtl for t in terms):
                    utility += 0.25

                utility = max(0.0, min(1.0, utility))
                if utility <= 0.25:
                    c.execute(
                        "UPDATE experiences SET archived_at=?, utility_score=? WHERE id=?",
                        (now, utility, eid),
                    )
                    pruned += 1
                else:
                    c.execute("UPDATE experiences SET utility_score=COALESCE(utility_score, ?) WHERE id=?", (utility, eid))

            return pruned

    def list_experiences_with_embeddings(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, created_at, user_id, source_id, modality, text, embedding_json FROM experiences WHERE embedding_json IS NOT NULL ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_experiences_without_embeddings(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, text FROM experiences WHERE embedding_json IS NULL AND text IS NOT NULL AND length(text) > 10 ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_unprocessed_experiences(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, created_at, processed_at, user_id, source_id, modality, text, blob_path, mime FROM experiences WHERE processed_at IS NULL ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_experience_processed(self, eid: int):
        with self._conn() as c:
            c.execute("UPDATE experiences SET processed_at=? WHERE id=?", (_ts(), int(eid)))

    # --- laws
    def add_law(
        self,
        text: str,
        title: str | None = None,
        source_id: str | None = None,
        source_experience_id: int | None = None,
    ) -> int:
        text = (text or '').strip()
        if not text:
            raise ValueError('empty law')
        now = _ts()
        with self._conn() as c:
            # de-dupe: if identical text already exists, return its id
            row = c.execute(
                "SELECT id FROM laws WHERE text=? AND status='active' ORDER BY id DESC LIMIT 1",
                (text,),
            ).fetchone()
            if row:
                return int(row[0])

            cur = c.execute(
                "INSERT INTO laws(created_at,updated_at,status,core,title,text,source_id,source_experience_id) VALUES(?,?,?,?,?,?,?,?)",
                (now, now, 'active', 1, title, text, source_id, source_experience_id),
            )
            lid = int(cur.lastrowid)
            try:
                c.execute(
                    "INSERT INTO events(created_at, kind, text, meta_json) VALUES(?,?,?,?)",
                    (now, 'law_new', f"+1 lei: {(title or 'Lei').strip()}", None),
                )
            except Exception:
                pass
            return lid

    def migrate_text_experiences_to_laws(self, limit: int = 200) -> dict[str, Any]:
        """Heuristically promote old text experiences into 'law' modality + laws table.

        Also marks them for reprocessing by setting processed_at=NULL.
        Runs in a single connection/transaction to reduce lock contention.
        """
        with self._conn() as c:
            try:
                c.execute("BEGIN IMMEDIATE")
            except Exception:
                pass

            rows = c.execute(
                """
                SELECT id, text, source_id, modality
                FROM experiences
                WHERE (modality IS NULL OR modality='text')
                  AND text IS NOT NULL
                ORDER BY id ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

            promoted = 0
            skipped = 0
            for r in rows:
                eid = int(r[0])
                text = (r[1] or '').strip()
                source_id = r[2]
                if len(text) < 80:
                    skipped += 1
                    continue

                t0 = text.lstrip()[:200].lower()
                # law-ish heuristics (PT)
                lawish = (
                    "lei" in t0
                    or "você deve" in t0
                    or t0.startswith("busque ")
                    or t0.startswith("valorize ")
                    or t0.startswith("reconheça ")
                    or t0.startswith("interprete ")
                    or t0.startswith("não ")
                    or "não " in t0
                    or "autonomia" in t0
                    or "não causar dano" in t0
                )
                if not lawish:
                    skipped += 1
                    continue

                # best-effort title: first line
                first_line = text.splitlines()[0].strip() if text.splitlines() else None
                title = None
                if first_line and len(first_line) <= 80:
                    title = first_line

                # insert/merge into laws (avoid nested connections)
                now = _ts()
                row2 = c.execute(
                    "SELECT id FROM laws WHERE text=? AND status='active' ORDER BY id DESC LIMIT 1",
                    (text,),
                ).fetchone()
                if not row2:
                    c.execute(
                        "INSERT INTO laws(created_at,updated_at,status,title,text,source_id,source_experience_id) VALUES(?,?,?,?,?,?,?)",
                        (now, now, 'active', title, text, source_id, eid),
                    )

                # promote modality and reprocess
                c.execute(
                    "UPDATE experiences SET modality='law', processed_at=NULL WHERE id=?",
                    (eid,),
                )
                promoted += 1

            return {"promoted": promoted, "skipped": skipped, "scanned": len(rows)}

    def list_laws(self, status: str = 'active', limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, created_at, updated_at, status, title, text, source_id FROM laws WHERE status=? ORDER BY id DESC LIMIT ?",
                (status, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows][::-1]

    def archive_law(self, law_id: int):
        now = _ts()
        with self._conn() as c:
            c.execute("UPDATE laws SET status='archived', updated_at=? WHERE id=?", (now, int(law_id)))

    # --- events
    def add_event(self, kind: str, text: str, meta_json: str | None = None) -> int:
        now = _ts()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO events(created_at, kind, text, meta_json) VALUES(?,?,?,?)",
                (now, kind, (text or '')[:2000], meta_json),
            )
            return int(cur.lastrowid)

    def list_events(self, since_id: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, created_at, kind, text, meta_json
                FROM events
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (int(since_id), int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- insights
    def add_insight(
        self,
        kind: str,
        title: str,
        text: str,
        priority: int = 3,
        conflict_id: int | None = None,
        source_id: str | None = None,
        meta_json: str | None = None,
    ) -> int:
        now = _ts()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO insights(created_at, kind, priority, title, text, conflict_id, source_id, meta_json) VALUES(?,?,?,?,?,?,?,?)",
                (now, kind, int(priority), (title or "")[:180], (text or "")[:2500], conflict_id, source_id, meta_json),
            )
            # mirror in events so overview feed can show it immediately
            c.execute(
                "INSERT INTO events(created_at, kind, text, meta_json) VALUES(?,?,?,?)",
                (now, "insight", f"💡 {title}: {(text or '')[:180]}", meta_json),
            )
            return int(cur.lastrowid)

    def list_insights(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, created_at, kind, priority, title, text, conflict_id, source_id, meta_json
                FROM insights
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows][::-1]

    def search_insights(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        like = f"%{(query or '').strip()}%"
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, created_at, kind, priority, title, text, conflict_id, source_id, meta_json
                FROM insights
                WHERE title LIKE ? OR text LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (like, like, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows][::-1]

    # --- actions
    def enqueue_action(
        self,
        kind: str,
        text: str,
        priority: int = 0,
        meta_json: str | None = None,
        expires_at: float | None = None,
        cooldown_key: str | None = None,
    ) -> int:
        now = _ts()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO actions(created_at,updated_at,status,kind,text,priority,meta_json,expires_at,cooldown_key) VALUES(?,?,?,?,?,?,?,?,?)",
                (now, now, 'queued', kind, text, int(priority or 0), meta_json, expires_at, cooldown_key),
            )
            return int(cur.lastrowid)

    def next_action(self) -> dict[str, Any] | None:
        now = _ts()
        with self._conn() as c:
            row = c.execute(
                """
                SELECT id, created_at, updated_at, status, kind, text, priority, policy_allowed, policy_score, last_error, meta_json, expires_at, cooldown_key
                FROM actions
                WHERE status='queued'
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY priority DESC, id ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
        return dict(row) if row else None

    def mark_action(self, action_id: int, status: str, *, policy_allowed: bool | None = None, policy_score: float | None = None, last_error: str | None = None):
        now = _ts()
        with self._conn() as c:
            c.execute(
                """
                UPDATE actions
                SET status=?, updated_at=?, policy_allowed=COALESCE(?,policy_allowed), policy_score=COALESCE(?,policy_score), last_error=COALESCE(?,last_error)
                WHERE id=?
                """,
                (
                    status,
                    now,
                    (1 if policy_allowed else 0) if policy_allowed is not None else None,
                    float(policy_score) if policy_score is not None else None,
                    last_error,
                    int(action_id),
                ),
            )

    def list_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, created_at, updated_at, status, kind, text, priority, policy_allowed, policy_score, last_error, meta_json, expires_at, cooldown_key
                FROM actions
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows][::-1]

    def expire_queued_actions(self) -> int:
        now = _ts()
        with self._conn() as c:
            cur = c.execute(
                "UPDATE actions SET status='expired', updated_at=? WHERE status='queued' AND expires_at IS NOT NULL AND expires_at<=?",
                (now, now),
            )
            return int(cur.rowcount or 0)

    # --- goals
    def upsert_goal(self, title: str, description: str | None = None, priority: int = 0) -> int:
        t = (title or "").strip()
        if not t:
            raise ValueError("empty goal title")

        with self._conn() as c:
            row = c.execute(
                """
                SELECT id FROM goals
                WHERE title=? AND status IN ('open','active')
                ORDER BY id DESC LIMIT 1
                """,
                (t,),
            ).fetchone()
            if row:
                gid = int(row[0])
                c.execute(
                    "UPDATE goals SET description=COALESCE(?, description), priority=MAX(priority, ?) WHERE id=?",
                    (description, int(priority or 0), gid),
                )
                return gid

            cur = c.execute(
                "INSERT INTO goals(created_at,status,title,description,priority) VALUES(?,?,?,?,?)",
                (_ts(), "open", t, description, int(priority or 0)),
            )
            return int(cur.lastrowid)

    def list_goals(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            if status:
                rows = c.execute(
                    "SELECT id, created_at, status, title, description, priority FROM goals WHERE status=? ORDER BY priority DESC, id DESC LIMIT ?",
                    (status, int(limit)),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT id, created_at, status, title, description, priority FROM goals ORDER BY id DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        return [dict(r) for r in rows][::-1]

    def get_active_goal(self) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                """
                SELECT id, created_at, status, title, description, priority
                FROM goals
                WHERE status='active'
                ORDER BY priority DESC, id ASC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    def activate_goal(self, goal_id: int) -> bool:
        gid = int(goal_id)
        with self._conn() as c:
            row = c.execute("SELECT id FROM goals WHERE id=? AND status IN ('open','active')", (gid,)).fetchone()
            if not row:
                return False
            c.execute("UPDATE goals SET status='open' WHERE status='active' AND id<>?", (gid,))
            c.execute("UPDATE goals SET status='active' WHERE id=?", (gid,))
        return True

    def activate_next_goal(self) -> dict[str, Any] | None:
        active = self.get_active_goal()
        if active:
            return active
        with self._conn() as c:
            row = c.execute(
                """
                SELECT id, created_at, status, title, description, priority
                FROM goals
                WHERE status='open'
                ORDER BY priority DESC, id ASC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        gid = int(row[0])
        self.activate_goal(gid)
        return self.get_active_goal()

    def mark_goal_done(self, goal_id: int):
        with self._conn() as c:
            c.execute("UPDATE goals SET status='done' WHERE id=?", (int(goal_id),))

    def add_goal_milestone(self, goal_id: int, week_index: int, title: str, progress_criteria: str | None = None) -> int:
        now = _ts()
        with self._conn() as c:
            # de-dupe por goal+title
            row = c.execute(
                "SELECT id FROM goal_milestones WHERE goal_id=? AND title=? AND status IN ('open','active') ORDER BY id DESC LIMIT 1",
                (int(goal_id), (title or '').strip()),
            ).fetchone()
            if row:
                return int(row[0])
            cur = c.execute(
                """
                INSERT INTO goal_milestones(goal_id,created_at,updated_at,week_index,title,progress_criteria,status,progress)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (int(goal_id), now, now, int(max(1, week_index)), (title or 'Milestone')[:180], progress_criteria, 'open', 0.0),
            )
            return int(cur.lastrowid)

    def list_goal_milestones(self, goal_id: int, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as c:
            if status:
                rows = c.execute(
                    """
                    SELECT id, goal_id, created_at, updated_at, week_index, title, progress_criteria, status, progress
                    FROM goal_milestones
                    WHERE goal_id=? AND status=?
                    ORDER BY week_index ASC, id ASC
                    LIMIT ?
                    """,
                    (int(goal_id), status, int(limit)),
                ).fetchall()
            else:
                rows = c.execute(
                    """
                    SELECT id, goal_id, created_at, updated_at, week_index, title, progress_criteria, status, progress
                    FROM goal_milestones
                    WHERE goal_id=?
                    ORDER BY week_index ASC, id ASC
                    LIMIT ?
                    """,
                    (int(goal_id), int(limit)),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_next_open_milestone(self, goal_id: int) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                """
                SELECT id, goal_id, created_at, updated_at, week_index, title, progress_criteria, status, progress
                FROM goal_milestones
                WHERE goal_id=? AND status IN ('open','active')
                ORDER BY week_index ASC, id ASC
                LIMIT 1
                """,
                (int(goal_id),),
            ).fetchone()
        return dict(row) if row else None

    def update_milestone_progress(self, milestone_id: int, progress: float, status: str | None = None):
        now = _ts()
        p = max(0.0, min(1.0, float(progress)))
        with self._conn() as c:
            if status:
                c.execute(
                    "UPDATE goal_milestones SET progress=?, status=?, updated_at=? WHERE id=?",
                    (p, status, now, int(milestone_id)),
                )
            else:
                c.execute(
                    "UPDATE goal_milestones SET progress=?, updated_at=? WHERE id=?",
                    (p, now, int(milestone_id)),
                )

    # --- procedures (procedural memory)
    def add_procedure(
        self,
        name: str,
        goal: str | None,
        steps_json: str,
        domain: str | None = None,
        proc_type: str = 'analysis',
        preconditions: str | None = None,
        success_criteria: str | None = None,
        source_experience_id: int | None = None,
    ) -> int:
        now = _ts()
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO procedures(created_at,updated_at,status,domain,proc_type,name,goal,preconditions,steps_json,success_criteria,source_experience_id)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (now, now, 'active', domain, (proc_type or 'analysis')[:40], (name or '')[:140], goal, preconditions, steps_json, success_criteria, source_experience_id),
            )
            return int(cur.lastrowid)

    def list_procedures(self, limit: int = 50, domain: str | None = None) -> list[dict[str, Any]]:
        with self._conn() as c:
            if domain:
                rows = c.execute(
                    """
                    SELECT id, created_at, updated_at, status, domain, proc_type, name, goal, preconditions, steps_json, success_criteria,
                           attempts, successes, avg_score, source_experience_id
                    FROM procedures
                    WHERE status='active' AND domain=?
                    ORDER BY avg_score DESC, id DESC
                    LIMIT ?
                    """,
                    (domain, int(limit)),
                ).fetchall()
            else:
                rows = c.execute(
                    """
                    SELECT id, created_at, updated_at, status, domain, proc_type, name, goal, preconditions, steps_json, success_criteria,
                           attempts, successes, avg_score, source_experience_id
                    FROM procedures
                    WHERE status='active'
                    ORDER BY avg_score DESC, id DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
        return [dict(r) for r in rows][::-1]

    def get_procedure(self, procedure_id: int) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                """
                SELECT id, created_at, updated_at, status, domain, proc_type, name, goal, preconditions, steps_json, success_criteria,
                       attempts, successes, avg_score, source_experience_id
                FROM procedures
                WHERE id=?
                """,
                (int(procedure_id),),
            ).fetchone()
        return dict(row) if row else None

    def add_procedure_run(
        self,
        procedure_id: int,
        input_text: str | None,
        output_text: str | None,
        score: float,
        success: bool,
        notes: str | None = None,
    ) -> int:
        now = _ts()
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO procedure_runs(procedure_id,created_at,input_text,output_text,score,success,notes)
                VALUES(?,?,?,?,?,?,?)
                """,
                (int(procedure_id), now, input_text, output_text, float(score), 1 if success else 0, notes),
            )

            prow = c.execute(
                "SELECT attempts, successes, avg_score FROM procedures WHERE id=?",
                (int(procedure_id),),
            ).fetchone()
            if prow:
                attempts = int(prow[0] or 0) + 1
                successes = int(prow[1] or 0) + (1 if success else 0)
                prev_avg = float(prow[2] or 0.0)
                new_avg = ((prev_avg * (attempts - 1)) + float(score)) / max(1, attempts)
                c.execute(
                    "UPDATE procedures SET attempts=?, successes=?, avg_score=?, updated_at=? WHERE id=?",
                    (attempts, successes, new_avg, now, int(procedure_id)),
                )

            return int(cur.lastrowid)

    # --- analogies (cross-domain transfer)
    def add_analogy(
        self,
        source_domain: str | None,
        target_domain: str | None,
        source_concept: str | None,
        target_concept: str | None,
        mapping_json: str | None,
        inference_rule: str | None,
        confidence: float = 0.5,
        status: str = 'hypothesis',
        evidence_refs_json: str | None = None,
        notes: str | None = None,
    ) -> int:
        now = _ts()
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO analogies(created_at,updated_at,status,source_domain,target_domain,source_concept,target_concept,mapping_json,inference_rule,confidence,evidence_refs_json,notes)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    now,
                    now,
                    (status or 'hypothesis')[:24],
                    source_domain,
                    target_domain,
                    source_concept,
                    target_concept,
                    mapping_json,
                    inference_rule,
                    float(confidence),
                    evidence_refs_json,
                    notes,
                ),
            )
            return int(cur.lastrowid)

    def list_analogies(self, limit: int = 50, status: str | None = None, target_domain: str | None = None) -> list[dict[str, Any]]:
        with self._conn() as c:
            q = """
                SELECT id, created_at, updated_at, status, source_domain, target_domain, source_concept, target_concept,
                       mapping_json, inference_rule, confidence, evidence_refs_json, notes
                FROM analogies
                WHERE 1=1
            """
            params: list[Any] = []
            if status:
                q += " AND status=?"
                params.append(status)
            if target_domain:
                q += " AND target_domain=?"
                params.append(target_domain)
            q += " ORDER BY confidence DESC, id DESC LIMIT ?"
            params.append(int(limit))
            rows = c.execute(q, tuple(params)).fetchall()
        return [dict(r) for r in rows][::-1]

    def update_analogy_status(self, analogy_id: int, status: str, confidence: float | None = None, notes: str | None = None):
        with self._conn() as c:
            c.execute(
                """
                UPDATE analogies
                SET status=?, updated_at=?, confidence=COALESCE(?,confidence), notes=COALESCE(?,notes)
                WHERE id=?
                """,
                ((status or 'hypothesis')[:24], _ts(), float(confidence) if confidence is not None else None, notes, int(analogy_id)),
            )

    def get_analogy(self, analogy_id: int) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                """
                SELECT id, created_at, updated_at, status, source_domain, target_domain, source_concept, target_concept,
                       mapping_json, inference_rule, confidence, evidence_refs_json, notes
                FROM analogies
                WHERE id=?
                """,
                (int(analogy_id),),
            ).fetchone()
        return dict(row) if row else None

    # --- global workspace (metacognição compartilhada)
    def publish_workspace(
        self,
        module: str,
        channel: str,
        payload_json: str | None,
        salience: float = 0.5,
        ttl_sec: int = 900,
    ) -> int:
        now = _ts()
        exp = now + max(30, int(ttl_sec or 900))
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO global_workspace(created_at,module,channel,payload_json,salience,ttl_sec,expires_at,consumed_by_json)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (now, (module or 'unknown')[:60], (channel or 'general')[:80], payload_json, float(salience), int(ttl_sec), exp, '{}'),
            )
            return int(cur.lastrowid)

    def read_workspace(self, channels: list[str] | None = None, limit: int = 30, include_expired: bool = False) -> list[dict[str, Any]]:
        now = _ts()
        with self._conn() as c:
            if channels:
                ph = ",".join(["?" for _ in channels])
                q = f"""
                    SELECT id, created_at, module, channel, payload_json, salience, ttl_sec, expires_at, consumed_by_json
                    FROM global_workspace
                    WHERE channel IN ({ph})
                """
                params: list[Any] = list(channels)
                if not include_expired:
                    q += " AND (expires_at IS NULL OR expires_at > ?)"
                    params.append(now)
                q += " ORDER BY salience DESC, id DESC LIMIT ?"
                params.append(int(limit))
                rows = c.execute(q, tuple(params)).fetchall()
            else:
                q = """
                    SELECT id, created_at, module, channel, payload_json, salience, ttl_sec, expires_at, consumed_by_json
                    FROM global_workspace
                """
                params2: list[Any] = []
                if not include_expired:
                    q += " WHERE (expires_at IS NULL OR expires_at > ?)"
                    params2.append(now)
                q += " ORDER BY salience DESC, id DESC LIMIT ?"
                params2.append(int(limit))
                rows = c.execute(q, tuple(params2)).fetchall()
        return [dict(r) for r in rows][::-1]

    def mark_workspace_consumed(self, item_id: int, consumer_module: str):
        with self._conn() as c:
            row = c.execute("SELECT consumed_by_json FROM global_workspace WHERE id=?", (int(item_id),)).fetchone()
            if not row:
                return
            try:
                data = row[0] or '{}'
                d = __import__('json').loads(data)
                if not isinstance(d, dict):
                    d = {}
            except Exception:
                d = {}
            d[(consumer_module or 'unknown')[:60]] = _ts()
            c.execute("UPDATE global_workspace SET consumed_by_json=? WHERE id=?", (__import__('json').dumps(d, ensure_ascii=False), int(item_id)))

    # --- questions
    def list_open_questions(self, limit: int = 50) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT question FROM questions WHERE status='open' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r[0] for r in rows]

    def get_question(self, qid: int) -> dict[str, Any] | None:
        """Busca uma pergunta por ID com todos os campos."""
        with self._conn() as c:
            row = c.execute(
                """
                SELECT id, question, context, priority, created_at, status, 
                       answered_at, answer, template_id, concept
                FROM questions
                WHERE id=?
                """,
                (int(qid),),
            ).fetchone()
        return dict(row) if row else None

    def add_questions(self, qs: list[dict[str, Any]]):
        if not qs:
            return
        with self._conn() as c:
            # Migrate: add template_id and concept columns if missing
            cols = {r[1] for r in c.execute("PRAGMA table_info(questions)").fetchall()}
            if "template_id" not in cols:
                try:
                    c.execute("ALTER TABLE questions ADD COLUMN template_id TEXT")
                except Exception:
                    pass
            if "concept" not in cols:
                try:
                    c.execute("ALTER TABLE questions ADD COLUMN concept TEXT")
                except Exception:
                    pass
            
            existing = set(
                r[0]
                for r in c.execute("SELECT question FROM questions WHERE status='open'").fetchall()
            )
            for q in qs:
                qt = (q.get("question") or "").strip()
                if not qt or qt in existing:
                    continue
                if qt.startswith("```") or "\"question\"" in qt:
                    continue
                c.execute(
                    "INSERT INTO questions(created_at,status,question,context,priority,template_id,concept) VALUES(?,?,?,?,?,?,?)",
                    (
                        _ts(),
                        "open",
                        qt,
                        q.get("context"),
                        int(q.get("priority") or 0),
                        q.get("template_id"),
                        q.get("concept"),
                    ),
                )
                existing.add(qt)

    def next_question(self) -> dict[str, Any] | None:
        with self._conn() as c:
            # ensure adaptive columns exist
            cols = {r[1] for r in c.execute("PRAGMA table_info(questions)").fetchall()}
            if "template_id" not in cols:
                try:
                    c.execute("ALTER TABLE questions ADD COLUMN template_id TEXT")
                except Exception:
                    pass
            if "concept" not in cols:
                try:
                    c.execute("ALTER TABLE questions ADD COLUMN concept TEXT")
                except Exception:
                    pass

            row = c.execute(
                """
                SELECT id, question, context, priority, created_at, template_id, concept
                FROM questions
                WHERE status='open'
                ORDER BY priority DESC, id ASC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    def list_open_questions_full(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, question, context, priority, created_at, template_id, concept
                FROM questions
                WHERE status='open'
                ORDER BY priority DESC, id ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def answer_question(self, qid: int, answer: str):
        with self._conn() as c:
            c.execute(
                "UPDATE questions SET status='answered', answered_at=?, answer=? WHERE id=?",
                (_ts(), answer, qid),
            )

    def dismiss_question(self, qid: int):
        with self._conn() as c:
            c.execute("UPDATE questions SET status='dismissed' WHERE id=?", (qid,))

    # --- knowledge graph
    def _add_or_reinforce_triple_conn(
        self,
        c: sqlite3.Connection,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float = 0.5,
        experience_id: int | None = None,
        note: str | None = None,
        contradicts: bool = False,
    ) -> int:
        now = _ts()
        subject = subject.strip()
        predicate = predicate.strip()
        object_ = object_.strip()

        row = c.execute(
            "SELECT id, confidence, support_count, contradict_count FROM triples WHERE subject=? AND predicate=? AND object=? ORDER BY id DESC LIMIT 1",
            (subject, predicate, object_),
        ).fetchone()

        if row:
            tid = int(row[0])
            support = int(row[2] or 1)
            contra = int(row[3] or 0)
            if contradicts:
                contra += 1
            else:
                support += 1

            # simple confidence update: move towards 1 with support, towards 0 with contradiction
            base = float(row[1] or 0.5)
            delta = 0.05
            if contradicts:
                base = max(0.05, base - delta)
            else:
                base = min(0.95, base + delta)

            c.execute(
                "UPDATE triples SET updated_at=?, confidence=?, support_count=?, contradict_count=? WHERE id=?",
                (now, base, support, contra, tid),
            )
        else:
            # initial counts
            support = 0 if contradicts else 1
            contra = 1 if contradicts else 0
            cur = c.execute(
                "INSERT INTO triples(created_at,updated_at,subject,predicate,object,confidence,support_count,contradict_count) VALUES(?,?,?,?,?,?,?,?)",
                (now, now, subject, predicate, object_, float(confidence), support, contra),
            )
            tid = int(cur.lastrowid)

        if experience_id or note:
            c.execute(
                "INSERT INTO triple_evidence(triple_id, experience_id, note, created_at) VALUES(?,?,?,?)",
                (tid, experience_id, note, now),
            )

            # governance: bump source support based on experience.source_id
            try:
                if experience_id:
                    er = c.execute(
                        "SELECT source_id FROM experiences WHERE id=?",
                        (int(experience_id),),
                    ).fetchone()
                    src = (er[0] if er else None)
                    if src and not contradicts:
                        c.execute(
                            "UPDATE sources SET support_count=support_count+1, updated_at=? WHERE id=?",
                            (now, src),
                        )
                        self._recompute_source_trust(c, src)
            except Exception:
                pass

        return tid

    def add_or_reinforce_triple(
        self,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float = 0.5,
        experience_id: int | None = None,
        note: str | None = None,
        contradicts: bool = False,
    ) -> int:
        """Upsert-ish triple.

        - If (s,p,o) exists, increase support_count and update confidence.
        - If contradicts=True, increase contradict_count instead.
        """
        with self._conn() as c:
            return self._add_or_reinforce_triple_conn(
                c,
                subject,
                predicate,
                object_,
                confidence=confidence,
                experience_id=experience_id,
                note=note,
                contradicts=contradicts,
            )

    # Backwards-compatible alias
    def add_triple(self, subject: str, predicate: str, object_: str, confidence: float = 0.5, experience_id: int | None = None, note: str | None = None) -> int:
        return self.add_or_reinforce_triple(subject, predicate, object_, confidence, experience_id, note)

    def list_triples_since(self, since_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, subject, predicate, object, confidence
                FROM triples
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (int(since_id), int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_triples_with_embeddings(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, subject, predicate, object, confidence, embedding_json
                FROM triples
                WHERE embedding_json IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_triples_without_embeddings(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, subject, predicate, object
                FROM triples
                WHERE embedding_json IS NULL
                ORDER BY id ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_triple_embedding(self, tid: int, embedding_json: str):
        with self._conn() as c:
            c.execute("UPDATE triples SET embedding_json=? WHERE id=?", (embedding_json, int(tid)))

    def list_norms(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return compiled norms for gating (subject='AGI')."""
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, subject, predicate, object, confidence
                FROM triples
                WHERE subject='AGI'
                ORDER BY confidence DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_triples(self, q: str, limit: int = 20) -> list[dict[str, Any]]:
        like = f"%{q}%"
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, subject, predicate, object, confidence
                FROM triples
                WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ?
                ORDER BY confidence DESC, id DESC
                LIMIT ?
                """,
                (like, like, like, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._conn() as c:
            exp = c.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            q_open = c.execute("SELECT COUNT(*) FROM questions WHERE status='open'").fetchone()[0]
            q_ans = c.execute("SELECT COUNT(*) FROM questions WHERE status='answered'").fetchone()[0]
            tri = c.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        return {"experiences": exp, "questions_open": q_open, "questions_answered": q_ans, "triples": tri}

    def reset_questions(self):
        with self._conn() as c:
            c.execute("DELETE FROM questions")

    # --- contradictions (tese ↔ antítese → síntese)
    def find_contradictions(self, min_conf: float = 0.6) -> list[dict[str, Any]]:
        """Find contradiction candidates.

        1) Same (subject,predicate) with multiple objects.
        2) Explicit negation pair: (subject,'é',X) and (subject,'não_é',X) or different X.
        """
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT subject, predicate, COUNT(DISTINCT object) as n
                FROM triples
                WHERE confidence >= ?
                  AND subject <> 'AGI' -- norms can have multiple objects; don't treat as contradiction
                GROUP BY subject, predicate
                HAVING n >= 2
                ORDER BY n DESC
                LIMIT 20
                """,
                (float(min_conf),),
            ).fetchall()

            # negation candidates: both 'é' and 'não_é' exist for same subject
            neg_rows = c.execute(
                """
                SELECT subject, COUNT(DISTINCT predicate) as n
                FROM triples
                WHERE confidence >= ? AND predicate IN ('é','não_é')
                GROUP BY subject
                HAVING n >= 2
                LIMIT 20
                """,
                (float(min_conf),),
            ).fetchall()

        out = []
        # regular contradictions
        for r in rows:
            subject, predicate = r[0], r[1]
            with self._conn() as c:
                objs = c.execute(
                    """
                    SELECT object, confidence FROM triples
                    WHERE subject=? AND predicate=? AND confidence >= ?
                    ORDER BY confidence DESC, id DESC
                    LIMIT 5
                    """,
                    (subject, predicate, float(min_conf)),
                ).fetchall()
            out.append({"subject": subject, "predicate": predicate, "objects": [dict(o) for o in objs]})

        # negation contradictions
        for nr in neg_rows:
            subject = nr[0]
            with self._conn() as c:
                objs = c.execute(
                    """
                    SELECT predicate, object, confidence FROM triples
                    WHERE subject=? AND predicate IN ('é','não_é') AND confidence >= ?
                    ORDER BY confidence DESC, id DESC
                    LIMIT 8
                    """,
                    (subject, float(min_conf)),
                ).fetchall()
            # represent as predicate 'é/Não_é'
            out.append({"subject": subject, "predicate": "(é vs não_é)", "objects": [dict(o) for o in objs]})

        return out

    def register_contradiction(self, contradiction: dict[str, Any]):
        """Update source contradict counts for evidence tied to conflicting triples.

        NOTE: This must be called only when a conflict is NEW or CHANGED.
        Otherwise, repeated penalty every loop will destroy trust incorrectly.
        """
        subject = contradiction.get('subject')
        predicate = contradiction.get('predicate')
        if not subject or not predicate:
            return

        # Only applies to normal (s,p) contradictions; skip the synthetic negation key.
        if predicate == "(é vs não_é)":
            return

        with self._conn() as c:
            tids = [
                int(r[0])
                for r in c.execute(
                    "SELECT id FROM triples WHERE subject=? AND predicate=?",
                    (subject, predicate),
                ).fetchall()
            ]
            if not tids:
                return

            now = _ts()
            srcs: set[str] = set()
            q = """
                SELECT e.source_id
                FROM triple_evidence te
                JOIN experiences e ON e.id = te.experience_id
                WHERE te.triple_id = ? AND e.source_id IS NOT NULL
            """
            for tid in tids:
                for r in c.execute(q, (tid,)).fetchall():
                    if r[0]:
                        srcs.add(str(r[0]))

            for src in srcs:
                self.ensure_source(src)
                c.execute(
                    "UPDATE sources SET contradict_count=contradict_count+1, updated_at=? WHERE id=?",
                    (now, src),
                )
                self._recompute_source_trust(c, src)

    def upsert_conflict(self, contradiction: dict[str, Any]) -> dict[str, Any] | None:
        """Persist 'doubt' about a contradiction so it doesn't evaporate.

        Returns: {id, is_new, has_new_variant}
        """
        subject = (contradiction.get('subject') or '').strip()
        predicate = (contradiction.get('predicate') or '').strip()
        objs = contradiction.get('objects') or []
        if not subject or not predicate or len(objs) < 2:
            return None
        if predicate == "(é vs não_é)":
            # keep as ephemeral for now (could be persisted later with normalization)
            return None

        key = f"{subject}\u241F{predicate}"
        now = _ts()

        # Build a short summary snapshot
        opts = ", ".join((o.get('object') or '') for o in objs if o.get('object'))
        summary = f"{subject} {predicate} -> {opts}"[:500]

        is_new = False
        has_new_variant = False

        with self._conn() as c:
            row = c.execute(
                "SELECT id, seen_count FROM conflicts WHERE key=? AND status='open' ORDER BY id DESC LIMIT 1",
                (key,),
            ).fetchone()

            if row:
                cid = int(row[0])
                seen = int(row[1] or 1) + 1
                c.execute(
                    "UPDATE conflicts SET updated_at=?, last_seen_at=?, seen_count=?, last_summary=? WHERE id=?",
                    (now, now, seen, summary, cid),
                )
            else:
                is_new = True
                cur = c.execute(
                    "INSERT INTO conflicts(created_at,updated_at,status,subject,predicate,key,first_seen_at,last_seen_at,seen_count,last_summary,last_question_at,question_count) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (now, now, 'open', subject, predicate, key, now, now, 1, summary, None, 0),
                )
                cid = int(cur.lastrowid)

            # Update variants
            for o in objs[:12]:
                obj = (o.get('object') or '').strip() if isinstance(o, dict) else str(o)
                conf = float(o.get('confidence') or 0.5) if isinstance(o, dict) else 0.5
                tid = o.get('id') if isinstance(o, dict) and o.get('id') is not None else None

                # de-dupe on (conflict_id, object)
                v = c.execute(
                    "SELECT id, seen_count FROM conflict_variants WHERE conflict_id=? AND object=? ORDER BY id DESC LIMIT 1",
                    (cid, obj),
                ).fetchone()
                if v:
                    vid = int(v[0])
                    vseen = int(v[1] or 1) + 1
                    c.execute(
                        "UPDATE conflict_variants SET last_seen_at=?, seen_count=?, confidence=?, triple_id=? WHERE id=?",
                        (now, vseen, conf, tid, vid),
                    )
                else:
                    has_new_variant = True
                    c.execute(
                        "INSERT INTO conflict_variants(conflict_id,triple_id,object,confidence,first_seen_at,last_seen_at,seen_count) VALUES(?,?,?,?,?,?,?)",
                        (cid, tid, obj, conf, now, now, 1),
                    )

            return {"id": cid, "is_new": is_new, "has_new_variant": has_new_variant}

    def list_conflicts(self, status: str = 'open', limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            if status == 'open':
                # norms are allowed to have many variants; don't show them as "conflicts" to humans.
                where = "status=? AND subject <> 'AGI'"
            else:
                where = "status=?"

            rows = c.execute(
                f"""
                SELECT id, created_at, updated_at, status, subject, predicate, first_seen_at, last_seen_at, seen_count, question_count, last_question_at, last_summary
                FROM conflicts
                WHERE {where}
                ORDER BY last_seen_at DESC, id DESC
                LIMIT ?
                """,
                (status, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_prioritized_conflicts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Prioriza conflitos persistentes + impacto no grafo para ciclo tese↔antítese↔síntese."""
        items = self.list_conflicts(status='open', limit=max(20, int(limit) * 3))
        now = _ts()

        scored = []
        with self._conn() as c:
            for cf in items:
                age_h = max(0.0, (now - float(cf.get('first_seen_at') or now)) / 3600.0)
                seen = int(cf.get('seen_count') or 0)
                qcount = int(cf.get('question_count') or 0)
                subj = cf.get('subject')
                pred = cf.get('predicate')

                # impacto: quantas triplas usam sujeito/predicado relacionado
                impact_row = c.execute(
                    "SELECT COUNT(*) FROM triples WHERE subject=? OR predicate=?",
                    (subj, pred),
                ).fetchone()
                impact = int(impact_row[0] or 0) if impact_row else 0

                # score crítico
                score = (seen * 1.0) + (min(age_h, 96.0) / 10.0) + (qcount * 0.8) + (min(impact, 80) * 0.35)
                risk = "high" if score >= 25 else ("medium" if score >= 12 else "low")

                c2 = dict(cf)
                c2['impact_score'] = impact
                c2['criticality'] = risk
                c2['priority_score'] = round(score, 3)
                scored.append(c2)

        scored.sort(key=lambda x: (x.get('priority_score', 0), x.get('impact_score', 0), x.get('id', 0)), reverse=True)
        return scored[: int(limit)]

    def archive_norm_conflicts(self) -> int:
        """One-shot cleanup: archive conflicts created from norms (subject='AGI')."""
        now = _ts()
        with self._conn() as c:
            cur = c.execute(
                "UPDATE conflicts SET status='archived', updated_at=? WHERE status='open' AND subject='AGI'",
                (now,),
            )
            return int(cur.rowcount or 0)

    def get_conflict(self, cid: int) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT id, created_at, updated_at, status, subject, predicate, first_seen_at, last_seen_at, seen_count, last_summary FROM conflicts WHERE id=?",
                (int(cid),),
            ).fetchone()
            if not row:
                return None
            variants = c.execute(
                """
                SELECT id, triple_id, object, confidence, first_seen_at, last_seen_at, seen_count
                FROM conflict_variants
                WHERE conflict_id=?
                ORDER BY confidence DESC, seen_count DESC, id DESC
                """,
                (int(cid),),
            ).fetchall()

            out_vars = []
            for v in variants:
                vd = dict(v)
                tid = vd.get("triple_id")
                if tid:
                    srow = c.execute(
                        """
                        SELECT AVG(COALESCE(s.trust,0.5))
                        FROM triple_evidence te
                        JOIN experiences e ON e.id=te.experience_id
                        LEFT JOIN sources s ON s.id=e.source_id
                        WHERE te.triple_id=?
                        """,
                        (int(tid),),
                    ).fetchone()
                    vd["source_trust"] = float(srow[0]) if srow and srow[0] is not None else 0.5
                else:
                    vd["source_trust"] = 0.5
                out_vars.append(vd)

        out = dict(row)
        out['variants'] = out_vars
        return out

    def resolve_conflict(
        self,
        cid: int,
        resolution: str | None = None,
        chosen_object: str | None = None,
        decided_by: str | None = None,
        notes: str | None = None,
    ):
        now = _ts()
        with self._conn() as c:
            # store resolution record
            c.execute(
                "INSERT INTO conflict_resolutions(conflict_id, created_at, decided_by, chosen_object, resolution_text, notes) VALUES(?,?,?,?,?,?)",
                (int(cid), now, decided_by, chosen_object, resolution, notes),
            )

            # mark resolved
            c.execute(
                "UPDATE conflicts SET status='resolved', updated_at=?, last_summary=COALESCE(?, last_summary) WHERE id=?",
                (now, resolution, int(cid)),
            )

            # consolidate learning into the graph (tutor -> evidence)
            if chosen_object:
                conf = c.execute(
                    "SELECT subject, predicate FROM conflicts WHERE id=?",
                    (int(cid),),
                ).fetchone()
                if conf:
                    subject, predicate = conf[0], conf[1]

                    # create a dedicated experience for the resolution (evidence)
                    src = None
                    if decided_by:
                        src = f"human:{decided_by.strip()[:80]}"
                    else:
                        src = "human:unknown"
                    # ensure source exists
                    try:
                        self._ensure_source_conn(c, src, kind="human", label=src)
                    except Exception:
                        pass

                    exp_text = f"RESOLUÇÃO CONFLITO #{cid}: '{subject}' {predicate} -> '{chosen_object}'.\nJustificativa: {resolution or '(sem justificativa)'}"
                    cur = c.execute(
                        "INSERT INTO experiences(created_at, processed_at, user_id, source_id, modality, text, blob_path, mime) VALUES(?,?,?,?,?,?,?,?)",
                        (now, now, None, src, "resolution", exp_text[:20000], None, "text/plain"),
                    )
                    eid = int(cur.lastrowid)

                    # reinforce chosen triple with strong confidence and evidence link
                    self._add_or_reinforce_triple_conn(
                        c,
                        subject,
                        predicate,
                        chosen_object,
                        confidence=0.85,
                        experience_id=eid,
                        note=f"conflict_resolution:{cid}",
                        contradicts=False,
                    )

                    # optionally mark other variants as contradicted (and drop confidence below contradiction threshold)
                    others = [
                        str(r[0])
                        for r in c.execute(
                            "SELECT object FROM conflict_variants WHERE conflict_id=? AND object<>?",
                            (int(cid), chosen_object),
                        ).fetchall()
                    ]
                    for obj in others[:5]:
                        # ensure the losing variant won't keep re-triggering contradictions
                        c.execute(
                            """
                            UPDATE triples
                            SET confidence = MIN(confidence, 0.45),
                                updated_at = ?,
                                contradict_count = contradict_count + 1
                            WHERE subject=? AND predicate=? AND object=?
                            """,
                            (now, subject, predicate, obj),
                        )
                        c.execute(
                            "INSERT INTO triple_evidence(triple_id, experience_id, note, created_at) SELECT id, ?, ?, ? FROM triples WHERE subject=? AND predicate=? AND object=? ORDER BY id DESC LIMIT 1",
                            (eid, f"conflict_resolution_contradict:{cid}", now, subject, predicate, obj),
                        )

                    try:
                        c.execute(
                            "INSERT INTO events(created_at, kind, text, meta_json) VALUES(?,?,?,?)",
                            (now, 'conflict_resolved', f"✅ conflito resolvido #{cid}: {subject} {predicate} -> {chosen_object}", None),
                        )
                    except Exception:
                        pass

                    # governance: reward sources supporting chosen_object, penalize sources supporting other variants
                    chosen_tids = [
                        int(r[0])
                        for r in c.execute(
                            "SELECT id FROM triples WHERE subject=? AND predicate=? AND object=?",
                            (subject, predicate, chosen_object),
                        ).fetchall()
                    ]
                    other_tids = [
                        int(r[0])
                        for r in c.execute(
                            "SELECT id FROM triples WHERE subject=? AND predicate=? AND object<>?",
                            (subject, predicate, chosen_object),
                        ).fetchall()
                    ]

                    def sources_for_tids(tids: list[int]) -> set[str]:
                        srcs: set[str] = set()
                        q = """
                            SELECT DISTINCT e.source_id
                            FROM triple_evidence te
                            JOIN experiences e ON e.id = te.experience_id
                            WHERE te.triple_id = ? AND e.source_id IS NOT NULL
                        """
                        for tid in tids:
                            for r in c.execute(q, (int(tid),)).fetchall():
                                if r[0]:
                                    srcs.add(str(r[0]))
                        return srcs

                    chosen_srcs = sources_for_tids(chosen_tids)
                    other_srcs = sources_for_tids(other_tids)

                    for ssrc in chosen_srcs:
                        self._ensure_source_conn(c, ssrc)
                        c.execute(
                            "UPDATE sources SET support_count=support_count+1, updated_at=? WHERE id=?",
                            (now, ssrc),
                        )
                        self._recompute_source_trust(c, ssrc)

                    for ssrc in (other_srcs - chosen_srcs):
                        self._ensure_source_conn(c, ssrc)
                        c.execute(
                            "UPDATE sources SET contradict_count=contradict_count+1, updated_at=? WHERE id=?",
                            (now, ssrc),
                        )
                        self._recompute_source_trust(c, ssrc)

    def archive_conflict(self, cid: int):
        now = _ts()
        with self._conn() as c:
            c.execute(
                "UPDATE conflicts SET status='archived', updated_at=? WHERE id=?",
                (now, int(cid)),
            )

    def should_prompt_conflict(self, conflict_id: int, *, is_new: bool, has_new_variant: bool, cooldown_hours: float = 12.0) -> bool:
        if is_new or has_new_variant:
            return True
        with self._conn() as c:
            row = c.execute(
                "SELECT last_question_at FROM conflicts WHERE id=?",
                (int(conflict_id),),
            ).fetchone()
        lastq = float(row[0]) if row and row[0] is not None else None
        if lastq is None:
            return True
        return (_ts() - lastq) >= (cooldown_hours * 3600.0)

    def mark_conflict_questioned(self, conflict_id: int):
        now = _ts()
        with self._conn() as c:
            c.execute(
                "UPDATE conflicts SET last_question_at=?, question_count=question_count+1, updated_at=? WHERE id=?",
                (now, now, int(conflict_id)),
            )

    def add_synthesis_question_if_needed(self, contradiction: dict[str, Any], conflict_id: int | None = None):
        subject = contradiction.get('subject')
        predicate = contradiction.get('predicate')
        objs = contradiction.get('objects') or []
        if not subject or not predicate or len(objs) < 2:
            return

        opts = ", ".join(o.get('object') for o in objs if isinstance(o, dict) and o.get('object'))

        # Include 'persisted doubt' context
        ctx = None
        if conflict_id:
            c = self.get_conflict(int(conflict_id))
            if c:
                age_days = max(0.0, (_ts() - float(c.get('first_seen_at') or _ts())) / 86400.0)
                ctx = f"Conflito ativo há {age_days:.1f} dias; visto {c.get('seen_count')} vezes; last={c.get('last_summary')}"

        q = f"(síntese) Encontrei contradição: '{subject}' {predicate} -> {opts}. Qual é a formulação correta? Em que contexto cada uma vale?"
        self.add_questions([{ "question": q, "context": ctx, "priority": 5 }])
        if conflict_id:
            self.mark_conflict_questioned(int(conflict_id))

import os

# Singleton Instance
DB_PATH = os.getenv("ULTRONPRO_DB_PATH", "/app/data/ultron.db")
db = Store(DB_PATH)

# Module-level wrappers for compatibility
def init_db():
    # Database is initialized on module import via Store.__init__
    pass

def get_stats():
    return db.stats()

def add_experience(text: str, source_id: str = None, modality: str = "text", **kwargs):
    return db.add_experience(user_id=None, text=text, source_id=source_id, modality=modality, **kwargs)

def get_question(qid: int):
    return db.get_question(qid)

def count_uncurated_experiences():
    return db.count_uncurated_experiences()

def list_uncurated_experiences(limit: int = 50):
    return db.list_uncurated_experiences(limit)

def count_archived_experiences():
    return db.count_archived_experiences()

def count_distilled_experiences():
    return db.count_distilled_experiences()

def mark_experiences_curated(ids: list[int]):
    return db.mark_experiences_curated(ids)

def rebuild_sources_from_experiences(limit: int = 5000):
    return db.rebuild_sources_from_experiences(limit)

def prune_low_utility_experiences(limit: int = 200, focus_terms: list[str] | None = None):
    return db.prune_low_utility_experiences(limit, focus_terms=focus_terms)

def mark_question_answered(qid: int, answer: str):
    return db.answer_question(qid, answer)

def dismiss_question(qid: int):
    return db.dismiss_question(qid)

def list_open_questions_full(limit: int = 50):
    return db.list_open_questions_full(limit=limit)

def get_triples(since_id: int = 0, limit: int = 500):
    return db.list_triples_since(since_id, limit)

def get_events(since_id: int = 0, limit: int = 50):
    return db.list_events(since_id, limit)

def add_insight(kind: str, title: str, text: str, priority: int = 3, conflict_id: int | None = None, source_id: str | None = None, meta_json: str | None = None):
    return db.add_insight(kind, title, text, priority=priority, conflict_id=conflict_id, source_id=source_id, meta_json=meta_json)

def list_insights(limit: int = 50):
    return db.list_insights(limit)

def search_insights(query: str, limit: int = 50):
    return db.search_insights(query, limit)

def get_sources(limit: int = 50):
    return db.list_sources(limit)

def search(query: str, limit: int = 10):
    return db.search_triples(query, limit)

def search_triples(query: str, limit: int = 10):
    return db.search_triples(query, limit)

def add_or_reinforce_triple(subject, predicate, object_, confidence=0.5, note=None, experience_id=None):
    return db.add_or_reinforce_triple(subject, predicate, object_, confidence, experience_id, note)

def upsert_goal(title: str, description: str | None = None, priority: int = 0):
    return db.upsert_goal(title, description, priority)

def list_goals(status: str | None = None, limit: int = 50):
    return db.list_goals(status=status, limit=limit)

def add_procedure(name: str, goal: str | None, steps_json: str, domain: str | None = None, proc_type: str = 'analysis', preconditions: str | None = None, success_criteria: str | None = None, source_experience_id: int | None = None):
    return db.add_procedure(name, goal, steps_json, domain=domain, proc_type=proc_type, preconditions=preconditions, success_criteria=success_criteria, source_experience_id=source_experience_id)

def list_procedures(limit: int = 50, domain: str | None = None):
    return db.list_procedures(limit=limit, domain=domain)

def get_procedure(procedure_id: int):
    return db.get_procedure(procedure_id)

def add_procedure_run(procedure_id: int, input_text: str | None, output_text: str | None, score: float, success: bool, notes: str | None = None):
    return db.add_procedure_run(procedure_id, input_text, output_text, score=score, success=success, notes=notes)

def add_analogy(source_domain: str | None, target_domain: str | None, source_concept: str | None, target_concept: str | None, mapping_json: str | None, inference_rule: str | None, confidence: float = 0.5, status: str = 'hypothesis', evidence_refs_json: str | None = None, notes: str | None = None):
    return db.add_analogy(source_domain, target_domain, source_concept, target_concept, mapping_json, inference_rule, confidence=confidence, status=status, evidence_refs_json=evidence_refs_json, notes=notes)

def list_analogies(limit: int = 50, status: str | None = None, target_domain: str | None = None):
    return db.list_analogies(limit=limit, status=status, target_domain=target_domain)

def update_analogy_status(analogy_id: int, status: str, confidence: float | None = None, notes: str | None = None):
    return db.update_analogy_status(analogy_id, status=status, confidence=confidence, notes=notes)

def get_analogy(analogy_id: int):
    return db.get_analogy(analogy_id)

def publish_workspace(module: str, channel: str, payload_json: str | None, salience: float = 0.5, ttl_sec: int = 900):
    return db.publish_workspace(module, channel, payload_json, salience=salience, ttl_sec=ttl_sec)

def read_workspace(channels: list[str] | None = None, limit: int = 30, include_expired: bool = False):
    return db.read_workspace(channels=channels, limit=limit, include_expired=include_expired)

def mark_workspace_consumed(item_id: int, consumer_module: str):
    return db.mark_workspace_consumed(item_id, consumer_module)

def get_active_goal():
    return db.get_active_goal()

def activate_next_goal():
    return db.activate_next_goal()

def mark_goal_done(goal_id: int):
    return db.mark_goal_done(goal_id)

def add_goal_milestone(goal_id: int, week_index: int, title: str, progress_criteria: str | None = None):
    return db.add_goal_milestone(goal_id, week_index, title, progress_criteria)

def list_goal_milestones(goal_id: int, status: str | None = None, limit: int = 20):
    return db.list_goal_milestones(goal_id, status=status, limit=limit)

def get_next_open_milestone(goal_id: int):
    return db.get_next_open_milestone(goal_id)

def update_milestone_progress(milestone_id: int, progress: float, status: str | None = None):
    return db.update_milestone_progress(milestone_id, progress, status=status)

# Backwards compatibility alias
add_triple = add_or_reinforce_triple
