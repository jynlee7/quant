"""market.db: bars and corporate actions, stored point in time (ARCHITECTURE §6.2, §9.1).

Rows are never overwritten. A changed bar becomes a new version with the time it became known,
so a decision can be replayed against exactly what was known then. History loaded in bulk is
stamped known_at = available_at; a revision must say when it became known.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

from quant.core.epoch import MINUTE_US, from_us, to_us
from quant.core.errors import DataError
from quant.core.schedule import Timing
from quant.data.frames import HourlyBar, MinuteFrame
from quant.data.policy import DataPolicy
from quant.data.view import PointInTimeView
from quant.store.sqlite import Migration, connect, migrate, transaction

# Times are epoch microseconds UTC; session and ex dates are ISO text. Primary keys put the
# version last, so the latest version of a row is an index lookup.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        (
            """CREATE TABLE minute_bars (
                symbol TEXT NOT NULL,
                t INTEGER NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                available_at INTEGER NOT NULL,
                known_at INTEGER NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY (symbol, t, version)
            ) WITHOUT ROWID""",
            """CREATE TABLE hourly_bars (
                symbol TEXT NOT NULL,
                t_open INTEGER NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                session TEXT NOT NULL,
                slot TEXT NOT NULL,
                t_close INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                minutes INTEGER NOT NULL,
                complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
                available_at INTEGER NOT NULL,
                known_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, t_open, version)
            ) WITHOUT ROWID""",
            """CREATE TABLE dividends (
                symbol TEXT NOT NULL,
                ex_date TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                amount REAL NOT NULL,
                known_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, ex_date, version)
            ) WITHOUT ROWID""",
            """CREATE TABLE splits (
                symbol TEXT NOT NULL,
                ex_date TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                ratio REAL NOT NULL,
                known_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, ex_date, version)
            ) WITHOUT ROWID""",
            """CREATE TABLE calendar_cache (
                source TEXT NOT NULL,
                session TEXT NOT NULL,
                open INTEGER NOT NULL,
                close INTEGER NOT NULL,
                known_at INTEGER NOT NULL,
                PRIMARY KEY (source, session, known_at)
            ) WITHOUT ROWID""",
            """CREATE TABLE data_flags (
                symbol TEXT NOT NULL,
                t INTEGER NOT NULL,
                kind TEXT NOT NULL,
                detail TEXT NOT NULL,
                known_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, t, kind)
            ) WITHOUT ROWID""",
        ),
    ),
)


# (latest version, insert) for each corporate-action table.
_ACTION_SQL = {
    "dividends": (
        "SELECT version, amount FROM dividends WHERE symbol = ? AND ex_date = ?"
        " ORDER BY version DESC LIMIT 1",
        "INSERT INTO dividends (symbol, ex_date, version, amount, known_at) VALUES (?, ?, ?, ?, ?)",
    ),
    "splits": (
        "SELECT version, ratio FROM splits WHERE symbol = ? AND ex_date = ?"
        " ORDER BY version DESC LIMIT 1",
        "INSERT INTO splits (symbol, ex_date, version, ratio, known_at) VALUES (?, ?, ?, ?, ?)",
    ),
}


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


def _hourly_values(bar: HourlyBar) -> tuple[object, ...]:
    """What a revision compares: everything but the version and known_at."""
    return (
        bar.session.isoformat(),
        str(bar.slot),
        to_us(bar.t_close),
        bar.open,
        bar.high,
        bar.low,
        bar.close,
        bar.volume,
        bar.minutes,
        int(bar.complete),
        to_us(bar.available_at),
    )


class BarStore:
    def __init__(self, conn: sqlite3.Connection, *, timing: Timing | None = None) -> None:
        self._conn = conn
        self._timing = timing or Timing()
        migrate(conn, MIGRATIONS, name="market.db")

    @classmethod
    def open(cls, path: Path) -> BarStore:
        return cls(connect(path))

    @classmethod
    def memory(cls) -> BarStore:
        return cls(connect(":memory:"))

    def close(self) -> None:
        self._conn.close()

    def _known_at(self, available: int, known_at: datetime | None, revision: bool) -> int:
        if known_at is not None:
            return to_us(known_at)
        if revision:
            raise DataError("a revision must say when it became known (known_at)")
        return available

    def put_minutes(
        self, symbol: str, minutes: MinuteFrame, *, source: str, known_at: datetime | None = None
    ) -> WriteSummary:
        """Store minutes. New ones become version 1; a changed one becomes a new version."""
        if len(minutes) == 0:
            return WriteSummary(0, 0, 0)
        if (np.diff(minutes.t) <= 0).any():
            raise DataError("minutes must be sorted and unique")
        latest = {
            row[0]: (row[1], row[2:])
            for row in self._conn.execute(
                "SELECT t, MAX(version), open, high, low, close, volume FROM minute_bars"
                " WHERE symbol = ? AND t >= ? AND t <= ? GROUP BY t",
                (symbol, int(minutes.t[0]), int(minutes.t[-1])),
            )
        }
        delay = int(self._timing.feed_delay.total_seconds() * 1_000_000)
        inserted = revised = unchanged = 0
        rows = []
        for i, t in enumerate(minutes.t.tolist()):
            values = (
                float(minutes.open[i]),
                float(minutes.high[i]),
                float(minutes.low[i]),
                float(minutes.close[i]),
                float(minutes.volume[i]),
            )
            stored = latest.get(t)
            if stored is not None and tuple(stored[1]) == values:
                unchanged += 1
                continue
            available = t + MINUTE_US + delay
            version = 1 if stored is None else stored[0] + 1
            known = self._known_at(available, known_at, revision=stored is not None)
            rows.append((symbol, t, version, *values, available, known, source))
            if stored is None:
                inserted += 1
            else:
                revised += 1
        with transaction(self._conn):
            self._conn.executemany(
                "INSERT INTO minute_bars (symbol, t, version, open, high, low, close, volume,"
                " available_at, known_at, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return WriteSummary(inserted, revised, unchanged)

    def put_hourly(
        self, symbol: str, bars: Sequence[HourlyBar], *, known_at: datetime | None = None
    ) -> WriteSummary:
        """Store hourly bars. New ones become version 1; a changed one becomes a new version."""
        if not bars:
            return WriteSummary(0, 0, 0)
        starts = [to_us(b.t_open) for b in bars]
        latest: dict[int, tuple[int, tuple[object, ...]]] = {}
        for row in self._conn.execute(
            "SELECT t_open, MAX(version), session, slot, t_close, open, high, low, close, volume,"
            " minutes, complete, available_at FROM hourly_bars"
            " WHERE symbol = ? AND t_open >= ? AND t_open <= ? GROUP BY t_open",
            (symbol, min(starts), max(starts)),
        ):
            latest[row[0]] = (row[1], tuple(row[2:]))
        inserted = revised = unchanged = 0
        rows = []
        for t_open, bar in zip(starts, bars, strict=True):
            values = _hourly_values(bar)
            stored = latest.get(t_open)
            if stored is not None and stored[1] == values:
                unchanged += 1
                continue
            version = 1 if stored is None else stored[0] + 1
            known = self._known_at(to_us(bar.available_at), known_at, revision=stored is not None)
            rows.append((symbol, t_open, version, *values, known))
            if stored is None:
                inserted += 1
            else:
                revised += 1
        with transaction(self._conn):
            self._conn.executemany(
                "INSERT INTO hourly_bars (symbol, t_open, version, session, slot, t_close, open,"
                " high, low, close, volume, minutes, complete, available_at, known_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return WriteSummary(inserted, revised, unchanged)

    def _put_action(
        self, table: str, symbol: str, ex_date: date, value: float, known_at: datetime
    ) -> None:
        latest, insert = _ACTION_SQL[table]
        with transaction(self._conn):
            row = self._conn.execute(latest, (symbol, ex_date.isoformat())).fetchone()
            if row is not None and row[1] == value:
                return
            version = 1 if row is None else row[0] + 1
            self._conn.execute(
                insert, (symbol, ex_date.isoformat(), version, value, to_us(known_at))
            )

    def put_dividend(
        self, symbol: str, *, ex_date: date, amount: float, known_at: datetime
    ) -> None:
        self._put_action("dividends", symbol, ex_date, amount, known_at)

    def put_split(self, symbol: str, *, ex_date: date, ratio: float, known_at: datetime) -> None:
        self._put_action("splits", symbol, ex_date, ratio, known_at)

    def versions(self, symbol: str, t: datetime) -> list[MinuteVersion]:
        """Every stored version of one minute, oldest first."""
        rows = self._conn.execute(
            "SELECT version, known_at, source, open, high, low, close, volume FROM minute_bars"
            " WHERE symbol = ? AND t = ? ORDER BY version",
            (symbol, to_us(t)),
        ).fetchall()
        return [
            MinuteVersion(row[0], from_us(row[1]), row[2], row[3], row[4], row[5], row[6], row[7])
            for row in rows
        ]

    def view(
        self, tau: datetime, *, as_of: datetime | None = None, policy: DataPolicy | None = None
    ) -> PointInTimeView:
        return PointInTimeView(self._conn, tau, as_of=as_of, policy=policy)
