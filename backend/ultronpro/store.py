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
        c = sqlite3.connect(str(self.path))
        c.row_factory = sqlite3.Row
        return c

    def _init(self):
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sources(
                  id TEXT PRIMARY KEY,
                  created_at REAL NOT NULL,
                  kind TEXT,
                  label TEXT,
                  trust REAL NOT NULL DEFAULT 0.5,
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
    def ensure_source(self, source_id: str, kind: str | None = None, label: str | None = None, trust: float | None = None):
        with self._conn() as c:
            row = c.execute("SELECT id FROM sources WHERE id=?", (source_id,)).fetchone()
            if row:
                return
            c.execute(
                "INSERT INTO sources(id, created_at, kind, label, trust, notes) VALUES(?,?,?,?,?,?)",
                (source_id, _ts(), kind, label, float(trust) if trust is not None else 0.5, None),
            )

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

    def add_synthesis_question_if_needed(self, contradiction: dict[str, Any]):
        subject = contradiction.get('subject')
        predicate = contradiction.get('predicate')
        objs = contradiction.get('objects') or []
        if not subject or not predicate or len(objs) < 2:
            return
        opts = ", ".join(o.get('object') for o in objs if o.get('object'))
        q = f"(síntese) Encontrei contradição: '{subject}' {predicate} -> {opts}. Qual é a formulação correta? Em que contexto cada uma vale?"
        self.add_questions([{ "question": q, "context": None, "priority": 5 }])
