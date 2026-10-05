"""Small stores of synthetic bars for the point-in-time tests."""

from __future__ import annotations

from datetime import date

from quant.core.calendar import TradingCalendar
from quant.core.schedule import Timing
from quant.data.bars import build_hourly
from quant.data.market_db import BarStore
from quant.data.synthetic import synthetic_minutes

SYMBOL = "SPY"


def store_with_days(start: date, end: date, *, seed: int = 3) -> BarStore:
    """An in-memory store holding synthetic minutes and hourly bars, loaded as history
    (known_at = available_at)."""
    calendar = TradingCalendar.xnys()
    minutes = synthetic_minutes(calendar, start, end, seed=seed)
    store = BarStore.memory()
    store.put_minutes(SYMBOL, minutes, source="synthetic")
    store.put_hourly(SYMBOL, build_hourly(calendar, minutes, Timing()))
    return store
