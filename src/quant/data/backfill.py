"""The resumable backfill of minute bars and corporate actions into market.db (ARCHITECTURE §6.1,
§6.2; `quant data backfill`).

It walks the sessions from `start` to `end` in order and ingests each session's minutes in one
`BarStore.ingest_minutes` call, so a session is stored whole or not at all, and history never
has to revise a row (B2). Once a session is stored, the store records it as backfilled
(`BarStore.mark_backfilled`), and a resume skips exactly those sessions, by an index lookup, so a
run that was interrupted (a crash, a network failure, Ctrl-C) resumes where it stopped, and the
rows it ends with are the rows one straight run would have stored. A session the source had no
minutes for isn't recorded, so the next run asks again.

A session can hold minutes the backfill never recorded: written live, or by another load. It is
fetched again. Where history can only add to it, history does; where history would have to
revise a row (an hourly bar the stored minutes built), it can't, so the session is written NOW
instead, after every other session and the actions, and reported in `BackfillResult.revised`. A
NOW write ends history for the store (ARCHITECTURE §6.2), which is why it comes last.

The run never goes past the last session whose minutes are all available: its close plus the
feed delay at or before now, whatever `end` says. Nor do the corporate actions it asks for.

**The supplement** (B7): dividends the feed misses (`quant.data.actions.DividendSupplement`).
When one is given, its dividends for `symbol` with ex-dates in [start, last session covered]
are stored after the feed's actions, under the same stamp, with `store_actions`, and before any
deferred NOW session. A supplement row equal to a stored dividend changes nothing; one that
differs (from the feed's row, or an earlier supplement) is left as stored and reported in
`BackfillResult.supplement.conflicts`. The supplement has no splits.

How rows are stamped (B2): HISTORY, each row known at its own available_at, while live trading
has never read the store; once it has (the store holds a NOW stamp), late data is known NOW.

Downloading lockbox-period bars is allowed (C1); nothing here computes a strategy result. A
backfill runs alone, never beside a backtest: a view's reads don't share one snapshot, so a
history load landing between two of them would change what the backtest saw (B2 "Snapshots").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Final, Protocol

from quant.core.calendar import TradingCalendar
from quant.core.clock import Clock
from quant.core.errors import DataError
from quant.core.schedule import Timing
from quant.data.actions import ActionSync, CorporateActions, DividendSupplement, store_actions
from quant.data.frames import MinuteFrame
from quant.data.market_db import HISTORY, NOW, BarStore, Stamp

SOURCE: Final = "alpaca:sip:raw"


class MarketDataSource(Protocol):
    """Where the backfill gets its data: Alpaca (`AlpacaMarketData`), or a fake in tests."""

    def session_minutes(self, symbol: str, day: date) -> MinuteFrame: ...

    def corporate_actions(self, symbol: str, start: date, end: date) -> CorporateActions: ...


@dataclass(frozen=True)
class BackfillResult:
    fetched: tuple[date, ...]  # sessions this run fetched, in order, empty ones included
    skipped: int  # sessions the store had recorded as backfilled, not fetched again
    empty: tuple[date, ...]  # sessions the source had no minutes for
    stamp: Stamp  # HISTORY, or NOW once live trading has read the store
    actions: ActionSync
    last: date | None  # the last session the run covered
    revised: tuple[date, ...] = ()  # partly stored sessions history couldn't complete: NOW
    # The supplement's dividends in the span, stored after the feed's (ActionSync(0, ()) without
    # a supplement or a session).
    supplement: ActionSync = field(default_factory=lambda: ActionSync(0, ()))


def backfill(
    store: BarStore,
    source: MarketDataSource,
    symbol: str,
    start: date,
    end: date | None = None,
    *,
    clock: Clock,
    calendar: TradingCalendar | None = None,
    timing: Timing | None = None,
    supplement: DividendSupplement | None = None,
) -> BackfillResult:
    """Backfill `symbol`'s sessions from `start` to `end` (inclusive), then its corporate
    actions over the same span (`quant.data.actions.store_actions`).

    The run stops at the last session whose minutes are all available by `clock`: its close
    plus the feed delay at or before now. `end` can stop it earlier, never later. An error from
    the source or the store propagates, and whatever sessions were stored before it stay stored
    and recorded, so running again resumes.

    `supplement`, when given, is stored after the feed's actions as the module docstring says,
    and reported in `BackfillResult.supplement`.
    """
    calendar = calendar or TradingCalendar.xnys()
    timing = timing or Timing()
    now = clock.now()
    last = now.date() if end is None else min(end, now.date())
    days = [
        day
        for day in (calendar.sessions(start, last) if start <= last else [])
        if timing.available_at(calendar.session(day).close) <= now
    ]
    stamp = HISTORY if store.high_water is None else NOW
    fetched: list[date] = []
    empty: list[date] = []
    deferred: list[tuple[date, MinuteFrame]] = []
    skipped = 0
    for day in days:
        if store.is_backfilled(symbol, day):
            skipped += 1
            continue
        minutes = source.session_minutes(symbol, day)
        fetched.append(day)
        if not len(minutes):
            empty.append(day)
            continue
        try:
            store.ingest_minutes(symbol, minutes, known_at=stamp, source=SOURCE)
        except DataError:
            session = calendar.session(day)
            if stamp is NOW or not store.has_minutes(symbol, session.open, session.close):
                raise
            deferred.append((day, minutes))  # history can't revise what is stored: NOW, last
            continue
        store.mark_backfilled(symbol, day, len(minutes))
    actions = (
        store_actions(
            store, symbol, source.corporate_actions(symbol, start, days[-1]), known_at=stamp
        )
        if days
        else ActionSync(0, ())
    )
    supplemented = ActionSync(0, ())
    if supplement is not None and days:
        selected = CorporateActions(
            tuple(
                d
                for d in supplement.actions.dividends
                if d.symbol == symbol and start <= d.ex_date <= days[-1]
            ),
            (),
        )
        supplemented = store_actions(store, symbol, selected, known_at=stamp, allow_revision=False)
    for day, minutes in deferred:
        store.ingest_minutes(symbol, minutes, known_at=NOW, source=SOURCE)
        store.mark_backfilled(symbol, day, len(minutes))
    return BackfillResult(
        fetched=tuple(fetched),
        skipped=skipped,
        empty=tuple(empty),
        stamp=stamp,
        actions=actions,
        last=days[-1] if days else None,
        revised=tuple(day for day, _ in deferred),
        supplement=supplemented,
    )
