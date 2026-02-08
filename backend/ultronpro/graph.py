from ultronpro import store

def init():
    """Initialize graph (managed by store)."""
    # Store already initializes tables on import/init
    pass

def add_triple(triple: dict, source_id: str = None) -> bool:
    """Add a triple to the graph."""
    try:
        store.add_or_reinforce_triple(
            subject=triple["subject"],
            predicate=triple["predicate"],
            object_=triple["object"],
            confidence=triple.get("confidence", 0.5),
            note=source_id
        )
        return True
    except Exception as e:
        print(f"Graph Error: {e}")
        return False
