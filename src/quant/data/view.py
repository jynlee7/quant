"""PointInTimeView: market data as it stood at decision time (I1, ARCHITECTURE §6.2).

A row is visible when its bar was available strictly before tau (available_at < tau), in the
latest version known strictly before as_of (known_at < as_of). In the backtest as_of is tau.
Live, the engine ingests the bar at tau and reads the view just after, with as_of stamped by the
store like a write (`BarStore.view(tau, as_of=NOW)`), so a row is known before as_of exactly when
it was committed before the view. The decision record keeps as_of, and a replay sees exactly the
same versions.

Every read names its purpose, and DataPolicy checks it against the days read (ARCHITECTURE
§6.3). A view that scores DEV carries the trial ticket that allows it (C11).

The view hashes every row it hands out. The digest names the data a computation actually read,
so caches keyed on it can't hide a lookahead behind a hit. Versions and known_at times are left
out of the hash: two stores holding the same visible values give the same digest.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np

from quant.core.calendar import NEW_YORK, Slot
from quant.core.canonical import canonical_dumps
from quant.core.clock import ensure_utc
from quant.core.epoch import from_us, to_us
from quant.core.errors import ClockError
from quant.data.frames import HourlyBar, MinuteFrame
from quant.data.policy import DataPolicy, Purpose, ScoringTicket

# Every query is a fixed string; values are bound parameters. Each picks the latest version of a
# row known before as_of with a correlated MAX, an index lookup on the (symbol, time, version)
# primary key. `t_open < :tau` repeats what available_at < tau implies, so the planner can walk
# the primary key backwards from tau.
_HOURLY_LATEST_N = """
    SELECT session, slot, t_open, t_close, open, high, low, close, volume, minutes, complete,
           available_at, version, known_at
    FROM hourly_bars h
    WHERE symbol = :symbol AND t_open < :tau AND available_at < :tau AND known_at < :as_of
      AND version = (SELECT MAX(x.version) FROM hourly_bars x
                     WHERE x.symbol = h.symbol AND x.t_open = h.t_open AND x.known_at < :as_of)
    ORDER BY t_open DESC LIMIT :limit
"""
_HOURLY_SINCE = """
    SELECT session, slot, t_open, t_close, open, high, low, close, volume, minutes, complete,
           available_at, version, known_at
    FROM hourly_bars h
    WHERE symbol = :symbol AND t_open >= :since AND t_open < :tau AND available_at < :tau
      AND known_at < :as_of
      AND version = (SELECT MAX(x.version) FROM hourly_bars x
                     WHERE x.symbol = h.symbol AND x.t_open = h.t_open AND x.known_at < :as_of)
    ORDER BY t_open
"""
_MINUTES = """
    SELECT t, open, high, low, close, volume
    FROM minute_bars m
    WHERE symbol = :symbol AND t >= :start AND t < :end AND t < :tau AND available_at < :tau
      AND known_at < :as_of
      AND version = (SELECT MAX(x.version) FROM minute_bars x
                     WHERE x.symbol = m.symbol AND x.t = m.t AND x.known_at < :as_of)
    ORDER BY t
"""
_ACTIONS = {
    "dividends": """
        SELECT ex_date, amount FROM dividends a
        WHERE symbol = :symbol AND ex_date >= :start AND ex_date <= :end AND known_at < :as_of
          AND version = (SELECT MAX(x.version) FROM dividends x
                         WHERE x.symbol = a.symbol AND x.ex_date = a.ex_date
                           AND x.known_at < :as_of)
        ORDER BY ex_date
    """,
    "splits": """
        SELECT ex_date, ratio FROM splits a
        WHERE symbol = :symbol AND ex_date >= :start AND ex_date <= :end AND known_at < :as_of
          AND version = (SELECT MAX(x.version) FROM splits x
                         WHERE x.symbol = a.symbol AND x.ex_date = a.ex_date
                           AND x.known_at < :as_of)
        ORDER BY ex_date
    """,
}


_US = timedelta(microseconds=1)


def _ny_date(moment: datetime) -> date:
    return moment.astimezone(NEW_YORK).date()


class PointInTimeView:
    def __init__(
        self,
        conn: sqlite3.Connection,
        tau: datetime,
        *,
        as_of: datetime,
        policy: DataPolicy | None = None,
        ticket: ScoringTicket | None = None,
    ) -> None:
        self.tau = ensure_utc(tau)
        self.as_of = ensure_utc(as_of)
        if self.as_of < self.tau:
            raise ClockError(
                f"as_of {self.as_of.isoformat()} precedes tau {self.tau.isoformat()}; a view "
                "can't know less than its decision time"
            )
        self._conn = conn
        self._policy = policy or DataPolicy()
        self._ticket = ticket
        self._params = {"tau": to_us(self.tau), "as_of": to_us(self.as_of)}
        self._hash = hashlib.sha256()

    def _check(self, purpose: Purpose, first: date, last: date) -> None:
        self._policy.check(purpose, first, last, ticket=self._ticket)

    def _check_to_tau(self, purpose: Purpose, first: date) -> None:
        """Check a read of bars, which reaches up to tau but not to it."""
        last = _ny_date(self.tau - _US)
        self._check(purpose, min(first, last), last)

    def hourly(
        self,
        symbol: str,
        *,
        purpose: Purpose,
        since: datetime | None = None,
        last: int | None = None,
    ) -> list[HourlyBar]:
        """Visible hourly bars in time order: all, those opening at or after `since`, or the
        latest `last`."""
        if since is not None and last is not None:
            raise ValueError("ask for bars since a time, or for the last n, not both")
        if last is not None and last < 0:
            raise ValueError(f"last must be zero or more, not {last}")
        params: dict[str, object] = {**self._params, "symbol": symbol}
        if since is not None:
            params["since"] = to_us(since)
            rows = self._conn.execute(_HOURLY_SINCE, params).fetchall()
        else:
            params["limit"] = -1 if last is None else last  # a negative LIMIT means no limit
            rows = self._conn.execute(_HOURLY_LATEST_N, params).fetchall()
            rows.reverse()
        bars = [
            HourlyBar(
                session=date.fromisoformat(row[0]),
                slot=Slot(row[1]),
                t_open=from_us(row[2]),
                t_close=from_us(row[3]),
                open=row[4],
                high=row[5],
                low=row[6],
                close=row[7],
                volume=row[8],
                minutes=row[9],
                complete=bool(row[10]),
                available_at=from_us(row[11]),
                version=row[12],
                known_at=from_us(row[13]),
            )
            for row in rows
        ]
        if bars:
            first = bars[0].session
        elif since is not None:
            first = _ny_date(since)
        else:
            first = _ny_date(self.tau)
        self._check_to_tau(purpose, first)
        for row in rows:
            self._hash.update(canonical_dumps(["hourly", symbol, row[2], *row[4:11]]) + b"\n")
        return bars

    def minutes(
        self, symbol: str, start: datetime, end: datetime, *, purpose: Purpose
    ) -> MinuteFrame:
        """Visible minutes starting in [start, end)."""
        last = _ny_date(min(ensure_utc(end), self.tau) - _US)  # the last day the read can reach
        self._check(purpose, min(_ny_date(start), last), last)
        params = {**self._params, "symbol": symbol, "start": to_us(start), "end": to_us(end)}
        rows = self._conn.execute(_MINUTES, params).fetchall()
        if not rows:
            return MinuteFrame.empty()
        columns = np.array(rows, dtype=np.float64).T
        frame = MinuteFrame(
            np.array([row[0] for row in rows], dtype=np.int64),
            columns[1].copy(),
            columns[2].copy(),
            columns[3].copy(),
            columns[4].copy(),
            columns[5].copy(),
        )
        self._hash.update(f"minutes|{symbol}|{len(frame)}|".encode())
        for column in (frame.t, frame.open, frame.high, frame.low, frame.close, frame.volume):
            self._hash.update(column.astype(column.dtype.newbyteorder("<")).tobytes())
        return frame

    def _actions(
        self, table: str, symbol: str, start: date, end: date, purpose: Purpose
    ) -> list[tuple[date, float]]:
        self._check(purpose, start, end)
        rows = self._conn.execute(
            _ACTIONS[table],
            {**self._params, "symbol": symbol, "start": start.isoformat(), "end": end.isoformat()},
        ).fetchall()
        found = [(date.fromisoformat(day), float(amount)) for day, amount in rows]
        self._hash.update(canonical_dumps([table, symbol, rows]) + b"\n")
        return found

    def dividends(
        self, symbol: str, start: date, end: date, *, purpose: Purpose
    ) -> list[tuple[date, float]]:
        """(ex_date, amount per share) for dividends known before as_of."""
        return self._actions("dividends", symbol, start, end, purpose)

    def splits(
        self, symbol: str, start: date, end: date, *, purpose: Purpose
    ) -> list[tuple[date, float]]:
        """(ex_date, ratio) for splits known before as_of."""
        return self._actions("splits", symbol, start, end, purpose)

    def digest(self) -> str:
        """SHA-256 over the canonical bytes of every row read so far."""
        return self._hash.hexdigest()
