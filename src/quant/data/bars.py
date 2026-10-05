"""The hourly bar builder (spec §2, ARCHITECTURE §6.1).

Each bar of the session map is built from its SIP minutes: the first open, the highest high,
the lowest low, the last close and the summed volume. A bar is complete when its last minute is
present; a bar missing that minute is kept but marked incomplete, and decisions treat it as
missing.
"""

from __future__ import annotations

import numpy as np

from quant.core.calendar import NEW_YORK, BarSpec, TradingCalendar
from quant.core.epoch import MINUTE_US, from_us, to_us
from quant.core.errors import DataError
from quant.core.schedule import Timing
from quant.data.frames import HourlyBar, Ints, MinuteFrame


def map_minutes(calendar: TradingCalendar, t: Ints) -> tuple[list[BarSpec], Ints]:
    """Where sorted minutes fall in the session map: the bars of every session from the first
    minute's day to the last's, and the index of the bar each minute starts in.

    The bars of a session tile it from the open to the close, so a minute outside every bar is
    outside regular trading hours, and raises DataError.
    """
    if len(t) == 0:
        return [], np.empty(0, dtype=np.int64)
    first = from_us(int(t[0])).astimezone(NEW_YORK).date()
    last = from_us(int(t[-1])).astimezone(NEW_YORK).date()
    specs = [bar for day in calendar.sessions(first, last) for bar in calendar.session(day).bars]
    if not specs:
        raise DataError(f"no trading session from {first} to {last}")
    opens = np.array([to_us(s.t_open) for s in specs], dtype=np.int64)
    closes = np.array([to_us(s.t_close) for s in specs], dtype=np.int64)
    which = np.searchsorted(opens, t, side="right") - 1
    outside = (which < 0) | (t >= closes[np.clip(which, 0, None)])
    if outside.any():
        i = int(np.flatnonzero(outside)[0])
        raise DataError(f"minute {from_us(int(t[i])).isoformat()}: outside regular trading hours")
    return specs, which


def build_hourly(
    calendar: TradingCalendar, minutes: MinuteFrame, timing: Timing
) -> list[HourlyBar]:
    """Hourly bars for every bar of the session map that has at least one minute."""
    if len(minutes) == 0:
        return []
    t = minutes.t
    if (np.diff(t) <= 0).any():
        raise DataError("minutes must be sorted and unique; validate them first")
    specs, which = map_minutes(calendar, t)

    starts = np.flatnonzero(np.r_[True, which[1:] != which[:-1]])
    ends = np.r_[starts[1:], len(t)]
    highs = np.maximum.reduceat(minutes.high, starts)
    lows = np.minimum.reduceat(minutes.low, starts)
    volumes = np.add.reduceat(minutes.volume, starts)
    bars = []
    for j, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        spec = specs[int(which[start])]
        bars.append(
            HourlyBar(
                session=spec.session,
                slot=spec.slot,
                t_open=spec.t_open,
                t_close=spec.t_close,
                open=float(minutes.open[start]),
                high=float(highs[j]),
                low=float(lows[j]),
                close=float(minutes.close[end - 1]),
                volume=float(volumes[j]),
                minutes=end - start,
                complete=int(t[end - 1]) == to_us(spec.t_close) - MINUTE_US,
                available_at=timing.available_at(spec.t_close),
            )
        )
    return bars
