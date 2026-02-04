from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrustSignal:
    source_id: str
    score: float  # 0..1
    reason: str


def combine_trust(signals: list[TrustSignal]) -> float:
    """Combine trust signals conservatively."""
    if not signals:
        return 0.5
    # conservative: weighted min-ish
    score = 1.0
    for s in signals:
        score = min(score, max(0.0, min(1.0, s.score)))
    return max(0.05, min(0.95, score))
