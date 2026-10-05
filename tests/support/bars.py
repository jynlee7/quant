"""Small stores of synthetic bars for the point-in-time tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.epoch import to_us
from quant.data.frames import HourlyBar, MinuteFrame
from quant.data.market_db import HISTORY, NOW, BarStore
from quant.data.synthetic import synthetic_minutes

SYMBOL = "SPY"


def synthetic_days(start: date, end: date, *, seed: int = 3) -> MinuteFrame:
    return synthetic_minutes(TradingCalendar.xnys(), start, end, seed=seed)


def store_with_days(
    start: date, end: date, *, seed: int = 3, clock: SimClock | None = None
) -> BarStore:
    """An in-memory store holding synthetic minutes and their hourly bars, loaded as history
    (known_at = available_at). Give it a clock to revise them."""
    store = BarStore.memory(clock=clock)
    store.ingest_minutes(
        SYMBOL, synthetic_days(start, end, seed=seed), known_at=HISTORY, source="synthetic"
    )
    return store


def with_close(minutes: MinuteFrame, i: int, close: float) -> MinuteFrame:
    """A copy with minute `i` closing at `close`, its high and low widened to fit."""
    c, h, lo = minutes.close.copy(), minutes.high.copy(), minutes.low.copy()
    c[i] = close
    h[i] = max(h[i], close)
    lo[i] = min(lo[i], close)
    return MinuteFrame(minutes.t.copy(), minutes.open.copy(), h, lo, c, minutes.volume.copy())


def revise_close(
    store: BarStore,
    clock: SimClock,
    minutes: MinuteFrame,
    bar: HourlyBar,
    delta: float,
    *,
    at: datetime,
) -> None:
    """Revise `bar`'s close the way a settle does at time `at`: a new version of its last
    minute, known then, which rebuilds the bar. `clock` is the store's."""
    own = minutes.select((minutes.t >= to_us(bar.t_open)) & (minutes.t < to_us(bar.t_close)))
    last = len(own) - 1
    revised = with_close(own, last, float(own.close[last]) + delta)
    clock.advance_to(at)
    store.ingest_minutes(SYMBOL, revised, known_at=NOW, source="settle")


@dataclass(frozen=True)
class Ticket:
    """Stands in for the ledger's TrialTicket (B10): a trial and the days it may score."""

    trial_id: str
    first_day: date
    last_day: date
