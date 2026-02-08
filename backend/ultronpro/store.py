
import os

# ... (existing Store class code) ...

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

def mark_question_answered(qid: int, answer: str):
    return db.answer_question(qid, answer)

def dismiss_question(qid: int):
    return db.dismiss_question(qid)

def get_triples(since_id: int = 0, limit: int = 500):
    return db.list_triples_since(since_id, limit)

def get_events(since_id: int = 0, limit: int = 50):
    return db.list_events(since_id, limit)

def get_sources(limit: int = 50):
    return db.list_sources(limit)

def search(query: str, limit: int = 10):
    return db.search_triples(query, limit)

def search_triples(query: str, limit: int = 10):
    return db.search_triples(query, limit)

def add_or_reinforce_triple(subject, predicate, object_, confidence=0.5, note=None, experience_id=None):
    return db.add_or_reinforce_triple(subject, predicate, object_, confidence, experience_id, note)

# Backwards compatibility alias
add_triple = add_or_reinforce_triple
