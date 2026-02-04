from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ActiveConflict:
    id: int
    subject: str
    predicate: str
    status: str
    first_seen_at: float
    last_seen_at: float
    seen_count: int

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - float(self.first_seen_at or 0.0))
