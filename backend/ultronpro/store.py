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
                  mime TEXT
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
                  title TEXT,
                  text TEXT NOT NULL,
                  source_id TEXT,
                  source_experience_id INTEGER
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
                  contradict_count INTEGER NOT NULL DEFAULT 0
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

            # lightweight migrations (add columns if upgrading)

            # migrations for laws
            if c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='laws'").fetchone():
                law_cols = {r[1] for r in c.execute("PRAGMA table_info(laws)").fetchall()}
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
            ]:
                if col not in exp_cols:
                    try:
                        c.execute(ddl)
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

    # --- experiences
    def ensure_source(
        self,
        source_id: str,
        kind: str | None = None,
        label: str | None = None,
        trust: float | None = None,
    ):
        with self._conn() as c:
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

    def add_experience(
        self,
        user_id: str | None,
        text: str | None,
        source_id: str | None = None,
        modality: str = "text",
        blob_path: str | None = None,
        mime: str | None = None,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO experiences(created_at, processed_at, user_id, source_id, modality, text, blob_path, mime) VALUES(?,?,?,?,?,?,?,?)",
                (_ts(), None, user_id, source_id, modality, text, blob_path, mime),
            )
            return int(cur.lastrowid)

    def list_experiences(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, created_at, processed_at, user_id, source_id, modality, text, blob_path, mime FROM experiences ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows][::-1]

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
                "INSERT INTO laws(created_at,updated_at,status,title,text,source_id,source_experience_id) VALUES(?,?,?,?,?,?,?)",
                (now, now, 'active', title, text, source_id, source_experience_id),
            )
            return int(cur.lastrowid)

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

    # --- questions
    def list_open_questions(self, limit: int = 50) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT question FROM questions WHERE status='open' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r[0] for r in rows]

    def add_questions(self, qs: list[dict[str, Any]]):
        if not qs:
            return
        with self._conn() as c:
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
                    "INSERT INTO questions(created_at,status,question,context,priority) VALUES(?,?,?,?,?)",
                    (_ts(), "open", qt, q.get("context"), int(q.get("priority") or 0)),
                )
                existing.add(qt)

    def next_question(self) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                """
                SELECT id, question, context, priority, created_at
                FROM questions
                WHERE status='open'
                ORDER BY priority DESC, id ASC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

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
        now = _ts()
        subject = subject.strip()
        predicate = predicate.strip()
        object_ = object_.strip()
        with self._conn() as c:
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

    # Backwards-compatible alias
    def add_triple(self, subject: str, predicate: str, object_: str, confidence: float = 0.5, experience_id: int | None = None, note: str | None = None) -> int:
        return self.add_or_reinforce_triple(subject, predicate, object_, confidence, experience_id, note)

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
            rows = c.execute(
                """
                SELECT id, created_at, updated_at, status, subject, predicate, first_seen_at, last_seen_at, seen_count, last_summary
                FROM conflicts
                WHERE status=?
                ORDER BY last_seen_at DESC, id DESC
                LIMIT ?
                """,
                (status, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

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
        out = dict(row)
        out['variants'] = [dict(v) for v in variants]
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

            # governance: reward sources supporting chosen_object, penalize sources supporting other variants
            if chosen_object:
                conf = c.execute(
                    "SELECT subject, predicate FROM conflicts WHERE id=?",
                    (int(cid),),
                ).fetchone()
                if conf:
                    subject, predicate = conf[0], conf[1]
                    # find triple ids for chosen & other
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

                    for src in chosen_srcs:
                        self.ensure_source(src)
                        c.execute(
                            "UPDATE sources SET support_count=support_count+1, updated_at=? WHERE id=?",
                            (now, src),
                        )
                        self._recompute_source_trust(c, src)

                    for src in (other_srcs - chosen_srcs):
                        self.ensure_source(src)
                        c.execute(
                            "UPDATE sources SET contradict_count=contradict_count+1, updated_at=? WHERE id=?",
                            (now, src),
                        )
                        self._recompute_source_trust(c, src)

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
