"""Small stores of synthetic bars for the point-in-time tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from quant.core.calendar import TradingCalendar
from quant.core.epoch import to_us
from quant.data.frames import HourlyBar, MinuteFrame
from quant.data.market_db import HISTORY, BarStore
from quant.data.synthetic import synthetic_minutes

SYMBOL = "SPY"


def synthetic_days(start: date, end: date, *, seed: int = 3) -> MinuteFrame:
    return synthetic_minutes(TradingCalendar.xnys(), start, end, seed=seed)


def store_with_days(start: date, end: date, *, seed: int = 3) -> BarStore:
    """An in-memory store holding synthetic minutes and their hourly bars, loaded as history
    (known_at = available_at)."""
    store = BarStore.memory()
    store.ingest_minutes(
        SYMBOL, synthetic_days(start, end, seed=seed), known_at=HISTORY, source="synthetic"
    )
    return store


def revise_close(
    store: BarStore, minutes: MinuteFrame, bar: HourlyBar, delta: float, *, known_at: datetime
) -> None:
    """Revise `bar`'s close the way a settle does: a new version of its last minute, known at
    `known_at`, which rebuilds the bar."""
    own = minutes.select((minutes.t >= to_us(bar.t_open)) & (minutes.t < to_us(bar.t_close)))
    last = len(own) - 1
    revised = own.with_close(last, float(own.close[last]) + delta)
    store.ingest_minutes(SYMBOL, revised, known_at=known_at, source="settle")


@dataclass(frozen=True)
class Ticket:
    """Stands in for the ledger's TrialTicket (B10): a trial and the days it may score."""

    trial_id: str
    first_day: date
    last_day: date
