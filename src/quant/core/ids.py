"""Deterministic ids, derived from what they identify, so replays reproduce them."""

from __future__ import annotations

from datetime import datetime


def decision_id(tau: datetime) -> str:
    """`d20190702T1446Z` for the decision at 2019-07-02 14:46 UTC."""
    raise NotImplementedError("red:B1")
