"""The Clock port: the only source of "now" in the code (K6).

Live trading plugs in the wall clock and the backtest a simulated one, so the decision code
never knows which it runs on. Inside the code every time is a timezone-aware UTC datetime.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """The current time, timezone-aware UTC."""
        ...

    def sleep_until(self, when: datetime) -> None:
        """Wait (wall clock) or jump (simulated clock) until `when`."""
        ...


def ensure_utc(value: datetime) -> datetime:
    """`value` in UTC. Naive datetimes are refused: their meaning depends on the machine."""
    raise NotImplementedError("red:B1")


class WallClock:
    """Real time, for live trading."""

    def now(self) -> datetime:
        raise NotImplementedError("red:B1")

    def sleep_until(self, when: datetime) -> None:
        raise NotImplementedError("red:B1")


class SimClock:
    """Simulated time, for the backtest and replays. It moves only when told, never backwards."""

    def __init__(self, start: datetime) -> None:
        raise NotImplementedError("red:B1")

    def now(self) -> datetime:
        raise NotImplementedError("red:B1")

    def advance_to(self, when: datetime) -> None:
        raise NotImplementedError("red:B1")

    def advance(self, delta: timedelta) -> None:
        raise NotImplementedError("red:B1")

    def sleep_until(self, when: datetime) -> None:
        raise NotImplementedError("red:B1")
