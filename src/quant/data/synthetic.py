"""Seeded synthetic minute bars on the real NYSE calendar, for tests and the fast tier (P4).

They follow a random walk in log price with an overnight gap at each open, rounded to cents,
with whole-number volumes. They are not a model of the market; they exercise the plumbing.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from quant.core.calendar import TradingCalendar
from quant.core.epoch import MINUTE_US, to_us
from quant.data.frames import MinuteFrame

_OVERNIGHT_VOL = 0.005


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
    days = calendar.sessions(start, end)
    if not days:
        return MinuteFrame.empty()
    blocks = []
    for day in days:
        session = calendar.session(day)
        count = (to_us(session.close) - to_us(session.open)) // MINUTE_US
        blocks.append(to_us(session.open) + np.arange(count, dtype=np.int64) * MINUTE_US)
    t = np.concatenate(blocks)
    firsts = np.cumsum([0] + [len(b) for b in blocks[:-1]])

    rng = np.random.default_rng(seed)
    n = len(t)
    gap = np.zeros(n)
    gap[firsts[1:]] = rng.normal(0.0, _OVERNIGHT_VOL, len(firsts) - 1)
    noise = rng.normal(0.0, minute_vol, n)
    close = np.round(start_price * np.exp(np.cumsum(gap + noise)), 2)

    opens = np.empty(n)
    opens[0] = round(start_price, 2)
    opens[1:] = close[:-1]
    opens[firsts[1:]] = np.round(close[firsts[1:] - 1] * np.exp(gap[firsts[1:]]), 2)

    wick_up = np.abs(rng.normal(0.0, minute_vol / 2, n))
    wick_down = np.abs(rng.normal(0.0, minute_vol / 2, n))
    high = np.ceil(np.maximum(opens, close) * np.exp(wick_up) * 100) / 100
    low = np.floor(np.minimum(opens, close) * np.exp(-wick_down) * 100) / 100
    high = np.maximum(high, np.maximum(opens, close))
    low = np.minimum(low, np.minimum(opens, close))
    volume = rng.integers(1_000, 50_000, n).astype(np.float64)
    return MinuteFrame(t, opens, high, low, close, volume)
