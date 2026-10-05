"""PointInTimeView: market data as it stood at decision time (I1, ARCHITECTURE §6.2).

A row is visible when its bar was available strictly before tau (available_at < tau), in the
latest version known strictly before as_of (known_at < as_of). In the backtest as_of is tau.
Live, the engine ingests the bar at tau and reads the view just after, so as_of is that later
moment; the decision record keeps it, and a replay sees exactly the same versions.

The view hashes every row it hands out. The digest names the data a computation actually read,
so caches keyed on it can't hide a lookahead behind a hit.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

from quant.data.frames import HourlyBar, MinuteFrame
from quant.data.policy import DataPolicy, Purpose


class PointInTimeView:
    def __init__(
        self,
        conn: sqlite3.Connection,
        tau: datetime,
        *,
        as_of: datetime | None = None,
        policy: DataPolicy | None = None,
    ) -> None:
        raise NotImplementedError("red:B2")

    def hourly(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        last: int | None = None,
        purpose: Purpose = Purpose.OPERATE,
    ) -> list[HourlyBar]:
        """Visible hourly bars in time order: all, those opening at or after `since`, or the
        latest `last`."""
        raise NotImplementedError("red:B2")

    def minutes(
        self, symbol: str, start: datetime, end: datetime, *, purpose: Purpose = Purpose.OPERATE
    ) -> MinuteFrame:
        """Visible minutes starting in [start, end)."""
        raise NotImplementedError("red:B2")

    def dividends(
        self, symbol: str, start: date, end: date, *, purpose: Purpose = Purpose.OPERATE
    ) -> list[tuple[date, float]]:
        raise NotImplementedError("red:B2")

    def splits(
        self, symbol: str, start: date, end: date, *, purpose: Purpose = Purpose.OPERATE
    ) -> list[tuple[date, float]]:
        raise NotImplementedError("red:B2")

    def digest(self) -> str:
        """SHA-256 over the canonical bytes of every row read so far."""
        raise NotImplementedError("red:B2")
