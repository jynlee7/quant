"""market.db: bars and corporate actions, stored point in time (ARCHITECTURE §6.2, §9.1).

Rows are never overwritten. A changed bar becomes a new version with the time it became known,
so a decision can be replayed against exactly what was known then. History loaded in bulk is
stamped known_at = available_at.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from quant.data.frames import HourlyBar, MinuteFrame
from quant.data.policy import DataPolicy
from quant.data.view import PointInTimeView
from quant.store.sqlite import Migration

MIGRATIONS: tuple[Migration, ...] = ()


@dataclass(frozen=True)
class WriteSummary:
    inserted: int
    revised: int
    unchanged: int


@dataclass(frozen=True)
class MinuteVersion:
    version: int
    known_at: datetime
    source: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class BarStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError("red:B2")

    @classmethod
    def open(cls, path: Path) -> BarStore:
        raise NotImplementedError("red:B2")

    @classmethod
    def memory(cls) -> BarStore:
        raise NotImplementedError("red:B2")

    def close(self) -> None:
        raise NotImplementedError("red:B2")

    def put_minutes(
        self, symbol: str, minutes: MinuteFrame, *, source: str, known_at: datetime | None = None
    ) -> WriteSummary:
        raise NotImplementedError("red:B2")

    def put_hourly(
        self, symbol: str, bars: Sequence[HourlyBar], *, known_at: datetime | None = None
    ) -> WriteSummary:
        raise NotImplementedError("red:B2")

    def put_dividend(
        self, symbol: str, *, ex_date: date, amount: float, known_at: datetime
    ) -> None:
        raise NotImplementedError("red:B2")

    def put_split(self, symbol: str, *, ex_date: date, ratio: float, known_at: datetime) -> None:
        raise NotImplementedError("red:B2")

    def versions(self, symbol: str, t: datetime) -> list[MinuteVersion]:
        """Every stored version of one minute, oldest first."""
        raise NotImplementedError("red:B2")

    def view(
        self, tau: datetime, *, as_of: datetime | None = None, policy: DataPolicy | None = None
    ) -> PointInTimeView:
        raise NotImplementedError("red:B2")
