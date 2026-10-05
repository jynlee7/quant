"""Times as integers: microseconds since the Unix epoch, UTC. This is how the stores keep them.

Integer arithmetic keeps the conversion exact in both directions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quant.core.clock import ensure_utc

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
MINUTE_US = 60_000_000


def to_us(value: datetime) -> int:
    delta = ensure_utc(value) - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def from_us(value: int) -> datetime:
    return EPOCH + timedelta(microseconds=int(value))
