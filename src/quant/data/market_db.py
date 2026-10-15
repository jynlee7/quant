"""market.db: bars and corporate actions, stored point in time (ARCHITECTURE §6.2, §9.1).

Rows are never overwritten. A changed bar becomes a new version with the time it became known,
so a decision can be replayed against exactly what was known then.

Minutes arrive through `ingest_minutes`, which validates them, stores them, rebuilds the hourly
bars they fall in and recomputes their sessions' move flags, all in one transaction. Every write
says how it is known:

- NOW stamps its rows from the store's clock, read under the write lock, so known_at orders
  exactly with commits. A live view takes its as_of the same way (`view(tau, as_of=NOW)`). The
  store keeps the last stamp it used, on rows or a live view, and each stamp comes after it even
  if the clock steps back. A row is then known before a live view's as_of exactly when it was
  committed before the view was stamped, and a replay at that as_of sees what live trading saw.
- HISTORY, for a bulk load of the past, stamps each row known at its own available_at, and an
  action at the end of its declaration day. It only adds rows, and only rows known after the
  store's last stamp: anything earlier, a live view may have read without it.

A revision is known NOW, after the version it replaces.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Final, assert_never

import numpy as np

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.clock import Clock
from quant.core.epoch import MINUTE_US, from_us, to_us
from quant.core.errors import ClockError, DataError
from quant.core.schedule import Timing
from quant.data.bars import build_hourly
from quant.data.frames import HourlyBar, Ints, MinuteFrame
from quant.data.policy import DataPolicy, ScoringTicket
from quant.data.validate import MOVE_KIND, DataFlag, check_minutes, move_flags
from quant.data.view import PointInTimeView
from quant.store.sqlite import Migration, connect, migrate, transaction

# Times are epoch microseconds UTC; session, ex and declaration dates are ISO text. Primary keys
# put the version last, so the latest version of a row is an index lookup.
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
                declared TEXT NOT NULL,
                known_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, ex_date, version)
            ) WITHOUT ROWID""",
            """CREATE TABLE splits (
                symbol TEXT NOT NULL,
                ex_date TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                ratio REAL NOT NULL,
                declared TEXT NOT NULL,
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
            # A flag's versions say when it was raised and when a revision cleared it (raised 0).
            """CREATE TABLE data_flags (
                symbol TEXT NOT NULL,
                t INTEGER NOT NULL,
                kind TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                raised INTEGER NOT NULL CHECK (raised IN (0, 1)),
                detail TEXT NOT NULL,
                known_at INTEGER NOT NULL,
                PRIMARY KEY (symbol, t, kind, version)
            ) WITHOUT ROWID""",
            # The last NOW stamp the store used, on rows or a live view.
            """CREATE TABLE high_water (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                stamp INTEGER NOT NULL
            )""",
        ),
    ),
    # The sessions the backfill fetched whole and stored (B6 review): a resume skips only
    # these, so a session that holds some minutes some other way is still fetched.
    Migration(
        2,
        (
            """CREATE TABLE backfilled_sessions (
                symbol TEXT NOT NULL,
                session TEXT NOT NULL,
                minutes INTEGER NOT NULL CHECK (minutes >= 1),
                PRIMARY KEY (symbol, session)
            ) WITHOUT ROWID""",
        ),
    ),
)


class Stamp(Enum):
    NOW = "now"
    HISTORY = "history"


NOW: Final = Stamp.NOW
"""Known now: by the store's clock under the write lock, after every stamp used before."""

HISTORY: Final = Stamp.HISTORY
"""Known when it became available: for a bulk load of the past that live trading never read.

History adds rows and never changes one, and only adds rows known after the store's last stamp.
A history load brings each session's minutes in one call; a row that would change, or that comes
too late, is known NOW.
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
    "INSERT INTO data_flags (symbol, t, kind, version, raised, detail, known_at)"
    " VALUES (?, ?, ?, ?, ?, ?, ?)"
)
_SET_HIGH_WATER = (
    "INSERT INTO high_water (id, stamp) VALUES (1, ?)"
    " ON CONFLICT (id) DO UPDATE SET stamp = excluded.stamp"
)
_FLAGS = """
    SELECT t, kind, detail FROM data_flags f
    WHERE symbol = :symbol AND raised = 1 AND known_at < :as_of
      AND version = (SELECT MAX(x.version) FROM data_flags x
                     WHERE x.symbol = f.symbol AND x.t = f.t AND x.kind = f.kind
                       AND x.known_at < :as_of)
    ORDER BY t, kind
"""
_FOREVER = 2**63 - 1

# (latest version, insert) for each corporate-action table.
_ACTION_SQL = {
    "dividends": (
        "SELECT version, known_at, amount, declared FROM dividends"
        " WHERE symbol = ? AND ex_date = ? ORDER BY version DESC LIMIT 1",
        "INSERT INTO dividends (symbol, ex_date, version, amount, declared, known_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
    ),
    "splits": (
        "SELECT version, known_at, ratio, declared FROM splits"
        " WHERE symbol = ? AND ex_date = ? ORDER BY version DESC LIMIT 1",
        "INSERT INTO splits (symbol, ex_date, version, ratio, declared, known_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
    ),
}


@dataclass(frozen=True)
class WriteSummary:
    inserted: int
    revised: int
    unchanged: int


@dataclass(frozen=True)
class IngestSummary:
    """What one ingest wrote, and the move flags it raised or cleared. A flagged minute is used,
    but alerted."""

    minutes: WriteSummary
    hourly: WriteSummary
    flags: tuple[DataFlag, ...]
    cleared: tuple[DataFlag, ...]


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


@dataclass
class _Stamping:
    """How one write stamps known_at: every row with NOW's stamp, or under HISTORY each row with
    its own time, after the store's last stamp. A NOW stamp handed out is used, and the write
    records it as the store's last stamp (`BarStore._write`)."""

    now: int | None
    floor: int | None
    used: bool = False

    def stamp(self) -> int:
        """The NOW stamp, handed out and so used."""
        if self.now is None:
            raise DataError("a history write has no stamp of its own")
        self.used = True
        return self.now

    def new(self, available: int) -> int:
        """known_at for the first version of a row, which became available at `available`."""
        if self.now is not None:
            return self.stamp()
        if self.floor is not None and available <= self.floor:
            raise DataError(
                f"history known at {from_us(available).isoformat()} comes at or before the "
                f"store's last stamp ({from_us(self.floor).isoformat()}), and a live view may "
                "have read that far without it; a late row is known NOW"
            )
        return available

    def revision(self, what: str, replaced: int) -> int:
        """known_at for a new version of `what`, whose latest version was known at `replaced`.
        A replay must not see a revision before live trading could have."""
        if self.now is None:
            raise DataError(
                f"{what} would change, and history can't revise a stored row. A history load "
                "brings each session's minutes in one call; a revision is known NOW"
            )
        if self.now <= replaced:
            raise DataError(
                f"{what}: a revision known at {from_us(self.now).isoformat()} must be known "
                f"after the version it replaces ({from_us(replaced).isoformat()})"
            )
        return self.stamp()


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


def _end_of_day(day: date) -> int:
    """When an action declared on `day` is known as history: the end of that day in New York,
    since the feed gives the date and not the time."""
    return to_us(datetime.combine(day + timedelta(days=1), time(), tzinfo=NEW_YORK))


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
    """market.db. A store with a clock can write NOW and serve live views; one without, such as
    a backtest's, can load history and serve views at given times."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        clock: Clock | None = None,
        calendar: TradingCalendar | None = None,
        timing: Timing | None = None,
    ) -> None:
        self._conn = conn
        self._clock = clock
        self._calendar = calendar or TradingCalendar.xnys()
        self._timing = timing or Timing()
        self._delay = self._timing.feed_delay // timedelta(microseconds=1)
        migrate(conn, MIGRATIONS, name="market.db")

    @classmethod
    def open(cls, path: Path, *, clock: Clock | None = None) -> BarStore:
        return cls(connect(path), clock=clock)

    @classmethod
    def memory(cls, *, clock: Clock | None = None) -> BarStore:
        return cls(connect(":memory:"), clock=clock)

    def close(self) -> None:
        self._conn.close()

    def ingest_minutes(
        self, symbol: str, minutes: MinuteFrame, *, known_at: Stamp, source: str
    ) -> IngestSummary:
        """Store minutes, rebuild the hourly bars they fall in and recompute the move flags of
        their sessions, in one transaction. Other sessions are left alone.

        A new minute, bar or flag becomes version 1, a changed one a new version, and an
        unchanged one is left alone, so ingesting the same data twice changes nothing. Bad
        minutes, history that would change a row or comes too late, or a revision not known
        after the version it replaces raise DataError and store nothing.
        """
        if len(minutes) == 0:
            return IngestSummary(_NOTHING, _NOTHING, (), ())
        specs, which = check_minutes(self._calendar, minutes)
        touched = [specs[i] for i in np.unique(which).tolist()]
        affected = {spec.t_open for spec in touched}
        # Bars and flags are rebuilt from the latest stored version of every minute of the
        # sessions touched, with the new minutes laid over them, so a flag sees the stored
        # minutes on either side of the new ones.
        spans = self._spans({spec.session for spec in touched})
        # Stamp, read and write under one write lock: known_at orders with commits, and a
        # concurrent writer can't take the same version.
        with self._write(known_at) as stamping:
            stored = [row for a, b in spans for row in self._latest_minutes(symbol, a, b)]
            overlay = _overlay(stored, minutes)
            minute_rows, minute_summary = self._minute_versions(
                symbol, minutes, stored, stamping, source
            )
            rebuilt = [
                bar
                for bar in build_hourly(self._calendar, overlay, self._timing)
                if bar.t_open in affected
            ]
            hourly_rows, hourly_summary = self._hourly_versions(symbol, rebuilt, stamping, spans)
            flag_rows, raised, cleared = self._flag_versions(symbol, overlay, stamping, spans)
            self._conn.executemany(_INSERT_MINUTE, minute_rows)
            self._conn.executemany(_INSERT_HOURLY, hourly_rows)
            self._conn.executemany(_INSERT_FLAG, flag_rows)
        return IngestSummary(minute_summary, hourly_summary, raised, cleared)

    def _spans(self, days: set[date]) -> list[tuple[int, int]]:
        """[open, close) of each run of consecutive sessions among `days`, in time order."""
        spans: list[tuple[int, int]] = []
        previous: date | None = None
        for day in sorted(days):
            session = self._calendar.session(day)
            if previous is not None and self._calendar.next_session(previous) == day:
                spans[-1] = (spans[-1][0], to_us(session.close))
            else:
                spans.append((to_us(session.open), to_us(session.close)))
            previous = day
        return spans

    @contextmanager
    def _write(self, known_at: Stamp) -> Iterator[_Stamping]:
        """A write transaction, stamped as `known_at` says. A NOW stamp handed out to rows or a
        live view becomes the store's last stamp as the transaction commits; one that stamped
        nothing leaves history's floor where it was."""
        with transaction(self._conn):
            stamping = self._stamping(known_at)
            yield stamping
            if stamping.used:
                self._conn.execute(_SET_HIGH_WATER, (stamping.now,))

    def _stamping(self, known_at: Stamp) -> _Stamping:
        """How a write is stamped; called under the write lock."""
        if known_at is NOW:
            return _Stamping(now=self._next_stamp(), floor=None)
        if known_at is HISTORY:
            return _Stamping(now=None, floor=self._high_water())
        assert_never(known_at)

    @property
    def high_water(self) -> datetime | None:
        """Last NOW stamp (write or live view), without issuing a new stamp."""
        stamp = self._high_water()
        return None if stamp is None else from_us(stamp)

    def minute_times(self, symbol: str, start: datetime, end: datetime) -> Ints:
        """The start times (epoch microseconds UTC, sorted) of the minutes stored in [start, end),
        in any version: which minutes the store holds, for the data-quality report. It carries
        no prices: those are read only through a view, at a time and for a purpose."""
        rows = self._conn.execute(
            "SELECT DISTINCT t FROM minute_bars WHERE symbol = ? AND t >= ? AND t < ? ORDER BY t",
            (symbol, to_us(start), to_us(end)),
        ).fetchall()
        return np.array([row[0] for row in rows], dtype=np.int64)

    def has_minutes(self, symbol: str, start: datetime, end: datetime) -> bool:
        """Whether any minute is stored in [start, end)."""
        row = self._conn.execute(
            "SELECT EXISTS (SELECT 1 FROM minute_bars WHERE symbol = ? AND t >= ? AND t < ?)",
            (symbol, to_us(start), to_us(end)),
        ).fetchone()
        return bool(row[0])

    def is_backfilled(self, symbol: str, session: date) -> bool:
        """Whether the backfill recorded `session` fetched whole and stored
        (`mark_backfilled`)."""
        row = self._conn.execute(
            "SELECT EXISTS (SELECT 1 FROM backfilled_sessions WHERE symbol = ? AND session = ?)",
            (symbol, session.isoformat()),
        ).fetchone()
        return bool(row[0])

    def mark_backfilled(self, symbol: str, session: date, minutes: int) -> None:
        """Record that the backfill fetched `session` whole and stored its `minutes` minutes."""
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO backfilled_sessions (symbol, session, minutes) VALUES (?, ?, ?)"
                " ON CONFLICT (symbol, session) DO UPDATE SET minutes = excluded.minutes",
                (symbol, session.isoformat(), minutes),
            )

    def _high_water(self) -> int | None:
        row = self._conn.execute("SELECT stamp FROM high_water").fetchone()
        return None if row is None else int(row[0])

    def _next_stamp(self) -> int:
        """A NOW stamp, under the write lock: the clock's reading, but after the last stamp
        used."""
        if self._clock is None:
            raise ClockError("this store has no clock, so it can't stamp NOW")
        stamp = to_us(self._clock.now())
        last = self._high_water()
        return stamp if last is None else max(stamp, last + 1)

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
        stamping: _Stamping,
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
            available = self._available(t)
            row = latest.get(t)
            if row is None:
                version, known = 1, stamping.new(available)
                inserted += 1
            elif row[3:] == values:
                unchanged += 1
                continue
            else:
                version = row[1] + 1
                known = stamping.revision(f"minute {from_us(t).isoformat()}", row[2])
                revised += 1
            rows.append((symbol, t, version, *values, available, known, source))
        return rows, WriteSummary(inserted, revised, unchanged)

    def _hourly_versions(
        self,
        symbol: str,
        bars: list[HourlyBar],
        stamping: _Stamping,
        spans: list[tuple[int, int]],
    ) -> tuple[list[tuple[object, ...]], WriteSummary]:
        latest = {
            row[0]: (row[1], row[2], tuple(row[3:]))
            for a, b in spans
            for row in self._conn.execute(
                "SELECT t_open, MAX(version), known_at, session, slot, t_close, open, high, low,"
                " close, volume, minutes, complete, available_at FROM hourly_bars"
                " WHERE symbol = ? AND t_open >= ? AND t_open < ? GROUP BY t_open",
                (symbol, a, b),
            )
        }
        rows: list[tuple[object, ...]] = []
        inserted = revised = unchanged = 0
        for bar in bars:
            t_open = to_us(bar.t_open)
            values = _hourly_values(bar)
            stored = latest.get(t_open)
            if stored is None:
                version, known = 1, stamping.new(to_us(bar.available_at))
                inserted += 1
            elif stored[2] == values:
                unchanged += 1
                continue
            else:
                version = stored[0] + 1
                known = stamping.revision(f"the hourly bar at {bar.t_open.isoformat()}", stored[1])
                revised += 1
            rows.append((symbol, t_open, version, *values, known))
        return rows, WriteSummary(inserted, revised, unchanged)

    def _flag_versions(
        self,
        symbol: str,
        overlay: MinuteFrame,
        stamping: _Stamping,
        spans: list[tuple[int, int]],
    ) -> tuple[list[tuple[object, ...]], tuple[DataFlag, ...], tuple[DataFlag, ...]]:
        """New versions of the move flags of minutes in `spans`: raised where a move is big and
        wasn't flagged so, cleared where a raised flag no longer holds."""
        computed = {to_us(flag.t): flag for flag in move_flags(self._calendar, overlay)}
        stored: dict[int, tuple[int, int, str, int]] = {
            row[0]: (row[1], row[2], row[3], row[4])
            for a, b in spans
            for row in self._conn.execute(
                "SELECT t, MAX(version), raised, detail, known_at FROM data_flags"
                " WHERE symbol = ? AND kind = ? AND t >= ? AND t < ? GROUP BY t",
                (symbol, MOVE_KIND, a, b),
            )
        }
        rows: list[tuple[object, ...]] = []
        raised: list[DataFlag] = []
        cleared: list[DataFlag] = []
        for t in sorted(computed.keys() | stored.keys()):
            flag, old = computed.get(t), stored.get(t)
            what = f"the move flag at {from_us(t).isoformat()}"
            if flag is not None and old is None:
                rows.append(
                    (symbol, t, MOVE_KIND, 1, 1, flag.detail, stamping.new(self._available(t)))
                )
                raised.append(flag)
            elif flag is not None and old is not None and (old[1] == 0 or old[2] != flag.detail):
                known = stamping.revision(what, old[3])
                rows.append((symbol, t, MOVE_KIND, old[0] + 1, 1, flag.detail, known))
                raised.append(flag)
            elif flag is None and old is not None and old[1] == 1:
                known = stamping.revision(what, old[3])
                rows.append((symbol, t, MOVE_KIND, old[0] + 1, 0, old[2], known))
                cleared.append(DataFlag(from_us(t), MOVE_KIND, old[2]))
        return rows, tuple(raised), tuple(cleared)

    def _put_action(
        self,
        table: str,
        symbol: str,
        ex_date: date,
        value: float,
        declared: date,
        known_at: Stamp,
    ) -> None:
        what = f"{table} for {symbol} on {ex_date}"
        if declared > ex_date:
            raise DataError(f"{what}: declared on {declared}, after its ex-date")
        latest, insert = _ACTION_SQL[table]
        values = (value, declared.isoformat())
        with self._write(known_at) as stamping:
            row = self._conn.execute(latest, (symbol, ex_date.isoformat())).fetchone()
            if row is None:
                version, known = 1, stamping.new(_end_of_day(declared))
            elif tuple(row[2:]) == values:
                return
            else:
                version, known = row[0] + 1, stamping.revision(what, row[1])
            self._conn.execute(insert, (symbol, ex_date.isoformat(), version, *values, known))

    def put_dividend(
        self, symbol: str, *, ex_date: date, amount: float, declared: date, known_at: Stamp
    ) -> None:
        self._put_action("dividends", symbol, ex_date, amount, declared, known_at)

    def put_split(
        self, symbol: str, *, ex_date: date, ratio: float, declared: date, known_at: Stamp
    ) -> None:
        self._put_action("splits", symbol, ex_date, ratio, declared, known_at)

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

    def flags(self, symbol: str, *, as_of: datetime | None = None) -> list[DataFlag]:
        """The flags that stand, in time order: now, or as they were known before `as_of`."""
        bound = _FOREVER if as_of is None else to_us(as_of)
        rows = self._conn.execute(_FLAGS, {"symbol": symbol, "as_of": bound})
        return [DataFlag(from_us(t), kind, detail) for t, kind, detail in rows]

    def view(
        self,
        tau: datetime,
        *,
        as_of: datetime | Stamp,
        policy: DataPolicy | None = None,
        ticket: ScoringTicket | None = None,
    ) -> PointInTimeView:
        """The data as it stood at decision time tau, in the versions known before as_of. Every
        view says what it knows: tau in a backtest, the recorded as_of in a replay, and NOW
        live, stamped like a write so it orders with every commit."""
        if as_of is NOW:
            with self._write(NOW) as stamping:  # a view refused for preceding tau stamps nothing
                as_of = from_us(stamping.stamp())
                view = PointInTimeView(self._conn, tau, as_of=as_of, policy=policy, ticket=ticket)
            return view
        if as_of is HISTORY:
            raise ValueError("a view's as_of is a time, or NOW for a live read")
        return PointInTimeView(self._conn, tau, as_of=as_of, policy=policy, ticket=ticket)
