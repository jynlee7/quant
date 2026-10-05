"""Seeded synthetic minute bars on the real NYSE calendar, for tests and the fast tier (P4).

They follow a random walk in log price with an overnight gap at each open, rounded to cents,
with whole-number volumes. They are not a model of the market; they exercise the plumbing.
"""

from __future__ import annotations

from datetime import date

from quant.core.calendar import TradingCalendar
from quant.data.frames import MinuteFrame


def synthetic_minutes(
    calendar: TradingCalendar,
    start: date,
    end: date,
    *,
    seed: int,
    start_price: float = 200.0,
    minute_vol: float = 0.0007,
) -> MinuteFrame:
    """Every regular-hours minute of every session from `start` to `end`."""
    raise NotImplementedError("red:B2")
