"""The resumable backfill of minute bars and corporate actions into market.db (ARCHITECTURE §6.1,
§6.2; `quant data backfill`).

It walks the sessions from `start` to `end` in order and ingests each session's minutes in one
`BarStore.ingest_minutes` call, so a session is stored whole or not at all, and history never
has to revise a row (B2). A session the store already holds is skipped without a fetch, so a run
that was interrupted (a crash, a network failure, Ctrl-C) resumes where it stopped, and the
rows it ends with are the rows one straight run would have stored.

How rows are stamped (B2): HISTORY, each row known at its own available_at, while live trading
has never read the store; once it has (the store holds a NOW stamp), late data is known NOW.

Downloading lockbox-period bars is allowed (C1); nothing here computes a strategy result. A
backfill runs alone, never beside a backtest: a view's reads don't share one snapshot, so a
history load landing between two of them would change what the backtest saw (B2 "Snapshots").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from quant.core.calendar import TradingCalendar
from quant.core.clock import Clock
from quant.core.schedule import Timing
from quant.data.actions import ActionSync, CorporateActions, store_actions
from quant.data.frames import MinuteFrame
from quant.data.market_db import HISTORY, NOW, BarStore, Stamp


class MarketDataSource(Protocol):
    """Where the backfill gets its data: Alpaca (`AlpacaMarketData`), or a fake in tests."""

    def session_minutes(self, symbol: str, day: date) -> MinuteFrame: ...

    def corporate_actions(self, symbol: str, start: date, end: date) -> CorporateActions: ...


@dataclass(frozen=True)
class BackfillResult:
    fetched: tuple[date, ...]  # sessions this run fetched, in order, empty ones included
    skipped: int  # sessions the store already held, not fetched again
    empty: tuple[date, ...]  # sessions the source had no minutes for
    stamp: Stamp  # HISTORY, or NOW once live trading has read the store
    actions: ActionSync
    last: date | None  # the last session the run covered


def backfill(
    store: BarStore,
    source: MarketDataSource,
    symbol: str,
    start: date,
    end: date | None = None,
    *,
    clock: Clock,
    calendar: TradingCalendar | None = None,
) -> BackfillResult:
    """Backfill `symbol`'s sessions from `start` to `end` (inclusive), then its corporate
    actions over the same span (`quant.data.actions.store_actions`).

    `end` defaults to the last session whose minutes are all available by `clock`: its close
    plus the feed delay at or before now. An error from the source or the store propagates,
    and whatever sessions were ingested before it stay stored, so running again resumes.
    """
    calendar = calendar or TradingCalendar.xnys()
    days = calendar.sessions(start, end or clock.now().date())
    if end is None:
        days = [d for d in days if Timing().available_at(calendar.session(d).close) <= clock.now()]
    stamp = HISTORY if store.high_water is None else NOW
    fetched: list[date] = []
    empty: list[date] = []
    skipped = 0
    for day in days:
        session = calendar.session(day)
        if len(store.latest_minutes(symbol, session.open, session.close)):
            skipped += 1
            continue
        minutes = source.session_minutes(symbol, day)
        store.ingest_minutes(symbol, minutes, known_at=stamp, source="alpaca:sip:raw")
        fetched.append(day)
        if not len(minutes):
            empty.append(day)
    actions = (
        store_actions(
            store, symbol, source.corporate_actions(symbol, start, days[-1]), known_at=stamp
        )
        if days
        else ActionSync(0, ())
    )
    return BackfillResult(
        tuple(fetched), skipped, tuple(empty), stamp, actions, days[-1] if days else None
    )
