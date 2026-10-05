"""market.db: bars and corporate actions, stored point in time (ARCHITECTURE §6.2, §9.1).

Rows are never overwritten. A changed bar becomes a new version with the time it became known,
so a decision can be replayed against exactly what was known then.

Minutes arrive through `ingest_minutes`, which validates them, stores them and rebuilds the
hourly bars they fall in, all in one transaction. Every ingest says when its data was known: the
fetch time for live and settle ingests, or HISTORY for a bulk load of the past, which stamps each
row known at its own available_at. A revision must be known after the version it replaces, and
can't be history: a backdated revision would let a replay see data that live trading didn't have.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Final

import numpy as np

from quant.core.calendar import TradingCalendar
from quant.core.epoch import MINUTE_US, from_us, to_us
from quant.core.errors import DataError
from quant.core.schedule import Timing
from quant.data.bars import build_hourly
from quant.data.frames import HourlyBar, MinuteFrame
from quant.data.policy import DataPolicy, ScoringTicket
from quant.data.validate import DataFlag, validate_minutes
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


class Stamp(Enum):
    HISTORY = "history"


HISTORY: Final = Stamp.HISTORY
"""The known_at of a bulk load of the past: each row is stamped known at its own available_at.

Only for data that live trading was never due to see. History adds rows and never changes one:
a minute or hourly bar that would change needs a revision, stamped with when it became known.
"""

# The latest stored version of a minute: t, version, known_at, open, high, low, close, volume.
_MinuteRow = tuple[int, int, int, float, float, float, float, float]

_INSERT_MINUTE = (
    "INSERT INTO minute_bars (symbol, t, version, open, high, low, close, volume, available_at,"
    " known_at, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_INSERT_HOURLY = (
    "INSERT INTO hourly_bars (symbol, t_open, version, session, slot, t_close, open, high, low,"
    " close, volume, minutes, complete, available_at, known_at)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_INSERT_FLAG = (
    "INSERT OR IGNORE INTO data_flags (symbol, t, kind, detail, known_at) VALUES (?, ?, ?, ?, ?)"
)

# (latest version, insert) for each corporate-action table.
_ACTION_SQL = {
    "dividends": (
        "SELECT version, known_at, amount FROM dividends WHERE symbol = ? AND ex_date = ?"
        " ORDER BY version DESC LIMIT 1",
        "INSERT INTO dividends (symbol, ex_date, version, amount, known_at) VALUES (?, ?, ?, ?, ?)",
    ),
    "splits": (
        "SELECT version, known_at, ratio FROM splits WHERE symbol = ? AND ex_date = ?"
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
class IngestSummary:
    """What one ingest wrote, and the flags its minutes raised (used, but alerted)."""

    minutes: WriteSummary
    hourly: WriteSummary
    flags: tuple[DataFlag, ...]


_NOTHING = WriteSummary(0, 0, 0)


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


def _known(stamp: int | None, available: int) -> int:
    """A row's known_at: the ingest's, or under HISTORY the row's own available_at."""
    return available if stamp is None else stamp


def _check_revision(what: str, known_at: int | None, replaced: int) -> None:
    """A revision says when it became known (not HISTORY), and that is after the version it
    replaces; otherwise a replay would see it before live trading could have."""
    if known_at is None:
        raise DataError(
            f"{what} would change, and history can't revise a stored row. A history load brings "
            "each session's minutes in one call; a revision says when it became known"
        )
    if known_at <= replaced:
        raise DataError(
            f"{what}: a revision known at {from_us(known_at).isoformat()} must be known after "
            f"the version it replaces ({from_us(replaced).isoformat()})"
        )


def _overlay(stored: list[_MinuteRow], minutes: MinuteFrame) -> MinuteFrame:
    """The latest version of each minute once `minutes` are written: theirs, and the stored rest."""
    if not stored:
        return minutes
    t = np.array([row[0] for row in stored], dtype=np.int64)
    rest = ~np.isin(t, minutes.t)
    if not rest.any():
        return minutes
    values = np.array([row[3:] for row in stored], dtype=np.float64)[rest]
    times = np.concatenate([t[rest], minutes.t])
    order = np.argsort(times, kind="stable")
    own = (minutes.open, minutes.high, minutes.low, minutes.close, minutes.volume)
    o, h, lo, c, v = (np.concatenate([values[:, j], own[j]])[order] for j in range(5))
    return MinuteFrame(times[order], o, h, lo, c, v)


class BarStore:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        calendar: TradingCalendar | None = None,
        timing: Timing | None = None,
    ) -> None:
        self._conn = conn
        self._calendar = calendar or TradingCalendar.xnys()
        self._timing = timing or Timing()
        self._delay = self._timing.feed_delay // timedelta(microseconds=1)
        migrate(conn, MIGRATIONS, name="market.db")

    @classmethod
    def open(cls, path: Path) -> BarStore:
        return cls(connect(path))

    @classmethod
    def memory(cls) -> BarStore:
        return cls(connect(":memory:"))

    def close(self) -> None:
        self._conn.close()

    def ingest_minutes(
        self, symbol: str, minutes: MinuteFrame, *, known_at: datetime | Stamp, source: str
    ) -> IngestSummary:
        """Store minutes and rebuild the hourly bars they fall in, in one transaction.

        A new minute or bar becomes version 1, a changed one a new version, and an unchanged one
        is left alone, so ingesting the same data twice changes nothing. Bad minutes, or a
        revision that is history or isn't known after the version it replaces, raise DataError
        and store nothing.
        """
        if len(minutes) == 0:
            return IngestSummary(_NOTHING, _NOTHING, ())
        flags = validate_minutes(self._calendar, minutes)
        touched = build_hourly(self._calendar, minutes, self._timing)
        start, end = to_us(touched[0].t_open), to_us(touched[-1].t_close)
        stamp = None if isinstance(known_at, Stamp) else to_us(known_at)
        flag_rows = []
        for flag in flags:
            t = to_us(flag.t)
            flag_rows.append((symbol, t, flag.kind, flag.detail, _known(stamp, self._available(t))))
        # Read and write under one write lock, so a concurrent writer can't take the same version.
        with transaction(self._conn):
            stored = self._latest_minutes(symbol, start, end)
            minute_rows, minute_summary = self._minute_versions(
                symbol, minutes, stored, stamp, source
            )
            affected = {bar.t_open for bar in touched}
            rebuilt = [
                bar
                for bar in build_hourly(self._calendar, _overlay(stored, minutes), self._timing)
                if bar.t_open in affected
            ]
            hourly_rows, hourly_summary = self._hourly_versions(symbol, rebuilt, stamp, start, end)
            self._conn.executemany(_INSERT_MINUTE, minute_rows)
            self._conn.executemany(_INSERT_HOURLY, hourly_rows)
            self._conn.executemany(_INSERT_FLAG, flag_rows)
        return IngestSummary(minute_summary, hourly_summary, tuple(flags))

    def _available(self, t: int) -> int:
        """When the minute starting at `t` becomes available: its close plus the feed delay."""
        return t + MINUTE_US + self._delay

    def _latest_minutes(self, symbol: str, start: int, end: int) -> list[_MinuteRow]:
        """The latest stored version of each minute starting in [start, end), in time order."""
        rows: list[_MinuteRow] = self._conn.execute(
            "SELECT t, MAX(version), known_at, open, high, low, close, volume FROM minute_bars"
            " WHERE symbol = ? AND t >= ? AND t < ? GROUP BY t ORDER BY t",
            (symbol, start, end),
        ).fetchall()
        return rows

    def _minute_versions(
        self,
        symbol: str,
        minutes: MinuteFrame,
        stored: list[_MinuteRow],
        stamp: int | None,
        source: str,
    ) -> tuple[list[tuple[object, ...]], WriteSummary]:
        latest = {row[0]: row for row in stored}
        o, h, lo, c, v = (
            column.tolist()
            for column in (minutes.open, minutes.high, minutes.low, minutes.close, minutes.volume)
        )
        rows: list[tuple[object, ...]] = []
        inserted = revised = unchanged = 0
        for i, t in enumerate(minutes.t.tolist()):
            values = (o[i], h[i], lo[i], c[i], v[i])
            row = latest.get(t)
            if row is None:
                version = 1
                inserted += 1
            elif row[3:] == values:
                unchanged += 1
                continue
            else:
                _check_revision(f"minute {from_us(t).isoformat()}", stamp, row[2])
                version = row[1] + 1
                revised += 1
            available = self._available(t)
            rows.append((symbol, t, version, *values, available, _known(stamp, available), source))
        return rows, WriteSummary(inserted, revised, unchanged)

    def _hourly_versions(
        self, symbol: str, bars: list[HourlyBar], stamp: int | None, start: int, end: int
    ) -> tuple[list[tuple[object, ...]], WriteSummary]:
        latest = {
            row[0]: (row[1], row[2], tuple(row[3:]))
            for row in self._conn.execute(
                "SELECT t_open, MAX(version), known_at, session, slot, t_close, open, high, low,"
                " close, volume, minutes, complete, available_at FROM hourly_bars"
                " WHERE symbol = ? AND t_open >= ? AND t_open < ? GROUP BY t_open",
                (symbol, start, end),
            )
        }
        rows: list[tuple[object, ...]] = []
        inserted = revised = unchanged = 0
        for bar in bars:
            t_open = to_us(bar.t_open)
            values = _hourly_values(bar)
            stored = latest.get(t_open)
            if stored is None:
                version = 1
                inserted += 1
            elif stored[2] == values:
                unchanged += 1
                continue
            else:
                _check_revision(f"the hourly bar at {bar.t_open.isoformat()}", stamp, stored[1])
                version = stored[0] + 1
                revised += 1
            rows.append((symbol, t_open, version, *values, _known(stamp, to_us(bar.available_at))))
        return rows, WriteSummary(inserted, revised, unchanged)

    def _put_action(
        self, table: str, symbol: str, ex_date: date, value: float, known_at: datetime
    ) -> None:
        latest, insert = _ACTION_SQL[table]
        known = to_us(known_at)
        with transaction(self._conn):
            row = self._conn.execute(latest, (symbol, ex_date.isoformat())).fetchone()
            if row is not None and row[2] == value:
                return
            if row is not None:
                _check_revision(f"{table} for {symbol} on {ex_date}", known, row[1])
            version = 1 if row is None else row[0] + 1
            self._conn.execute(insert, (symbol, ex_date.isoformat(), version, value, known))

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

    def flags(self, symbol: str) -> list[DataFlag]:
        """Every flag the symbol's minutes raised, in time order."""
        rows = self._conn.execute(
            "SELECT t, kind, detail FROM data_flags WHERE symbol = ? ORDER BY t, kind", (symbol,)
        )
        return [DataFlag(from_us(t), kind, detail) for t, kind, detail in rows]

    def view(
        self,
        tau: datetime,
        *,
        as_of: datetime | None = None,
        policy: DataPolicy | None = None,
        ticket: ScoringTicket | None = None,
    ) -> PointInTimeView:
        return PointInTimeView(self._conn, tau, as_of=as_of, policy=policy, ticket=ticket)
