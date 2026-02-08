from ultronpro import store

def init():
    """Initialize graph (managed by store)."""
    # Store already initializes tables on import/init
    pass

def add_triple(triple: dict, source_id: str = None) -> bool:
    """Add a triple to the graph."""
    import logging
    logger = logging.getLogger("uvicorn")
    
    try:
        logger.info(f"add_triple: {triple} from {source_id}")
        store.add_or_reinforce_triple(
            subject=triple["subject"],
            predicate=triple["predicate"],
            object_=triple["object"],
            confidence=triple.get("confidence", 0.5),
            note=source_id
        )
        logger.info(f"add_triple: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"add_triple ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
