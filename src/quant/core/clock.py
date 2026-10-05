"""The Clock port: the only source of "now" in the code (K6).

Live trading plugs in the wall clock and the backtest a simulated one, so the decision code
never knows which it runs on. Inside the code every time is a timezone-aware UTC datetime.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol

from quant.core.errors import ClockError


class Clock(Protocol):
    def now(self) -> datetime:
        """The current time, timezone-aware UTC."""
        ...

    def sleep_until(self, when: datetime) -> None:
        """Wait (wall clock) or jump (simulated clock) until `when`."""
        ...


def ensure_utc(value: datetime) -> datetime:
    """`value` in UTC. Naive datetimes are refused: their meaning depends on the machine."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ClockError(f"naive datetime {value.isoformat()}; times must carry a timezone")
    return value.astimezone(UTC)


class WallClock:
    """Real time, for live trading."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def sleep_until(self, when: datetime) -> None:
        delay = (ensure_utc(when) - self.now()).total_seconds()
        if delay > 0:
            time.sleep(delay)


class SimClock:
    """Simulated time, for the backtest and replays. It moves only when told, never backwards."""

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, when: datetime) -> None:
        when = ensure_utc(when)
        if when < self._now:
            raise ClockError(
                f"simulated time can't go back from {self._now.isoformat()} to {when.isoformat()}"
            )
        self._now = when

    def advance(self, delta: timedelta) -> None:
        self.advance_to(self._now + delta)

    def sleep_until(self, when: datetime) -> None:
        self.advance_to(when)
