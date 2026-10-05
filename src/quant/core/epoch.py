"""Times as integers: microseconds since the Unix epoch, UTC. This is how the stores keep them."""

from __future__ import annotations

from datetime import datetime


def to_us(value: datetime) -> int:
    raise NotImplementedError("red:B2")


def from_us(value: int) -> datetime:
    raise NotImplementedError("red:B2")
