"""Checks on minute bars before they are stored (ARCHITECTURE §6.1).

Bad data is refused: prices out of order, non-positive or non-finite, negative volume, duplicate
or unsorted minutes, minutes off the minute or outside a session. A move over 5% from one minute's
close to the next is kept but flagged, because it can be real (it is used, and it raises an
alert). The next minute is the next one present in the same session, so a missing minute can't
hide a jump; the overnight gap isn't a one-minute move.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.epoch import MINUTE_US, from_us, to_us
from quant.core.errors import DataError
from quant.data.frames import MinuteFrame

MOVE_FLAG = 0.05


@dataclass(frozen=True)
class DataFlag:
    t: datetime
    kind: str
    detail: str


def _fail_where(bad: npt.ArrayLike, t: npt.NDArray[np.int64], why: str) -> None:
    mask = np.asarray(bad, dtype=bool)
    if mask.any():
        i = int(np.flatnonzero(mask)[0])
        raise DataError(f"minute {from_us(int(t[i])).isoformat()}: {why}")


def validate_minutes(calendar: TradingCalendar, minutes: MinuteFrame) -> list[DataFlag]:
    """Raise DataError on bad minutes; return flags on suspicious ones."""
    if len(minutes) == 0:
        return []
    t, o, h, lo, c, v = (
        minutes.t,
        minutes.open,
        minutes.high,
        minutes.low,
        minutes.close,
        minutes.volume,
    )
    columns = np.vstack([o, h, lo, c, v])
    _fail_where(~np.isfinite(columns).all(axis=0), t, "a value is NaN or infinite")
    _fail_where((np.vstack([o, h, lo, c]) <= 0).any(axis=0), t, "a price is zero or negative")
    _fail_where(v < 0, t, "volume is negative")
    _fail_where(
        (lo > o) | (lo > c) | (h < o) | (h < c),
        t,
        "prices out of order (low <= open, close <= high)",
    )
    _fail_where(t % MINUTE_US != 0, t, "not on a whole minute")
    step = np.diff(t)
    _fail_where(np.r_[False, step == 0], t, "duplicate minute")
    _fail_where(np.r_[False, step < 0], t, "minutes out of order")

    first = from_us(int(t[0])).astimezone(NEW_YORK).date()
    last = from_us(int(t[-1])).astimezone(NEW_YORK).date()
    days = calendar.sessions(first, last)
    if not days:
        raise DataError(f"no trading session from {first} to {last}")
    opens = np.array([to_us(calendar.session(d).open) for d in days], dtype=np.int64)
    closes = np.array([to_us(calendar.session(d).close) for d in days], dtype=np.int64)
    k = np.searchsorted(opens, t, side="right") - 1
    inside = (k >= 0) & (t < closes[np.clip(k, 0, None)])
    _fail_where(~inside, t, "outside regular trading hours")

    same_session = k[1:] == k[:-1]
    moves = np.abs(c[1:] / c[:-1] - 1.0) > MOVE_FLAG
    return [
        DataFlag(
            from_us(int(t[i])),
            "move_over_5pct",
            f"close {c[i - 1]} -> {c[i]} over {(t[i] - t[i - 1]) // MINUTE_US} min",
        )
        for i in (np.flatnonzero(same_session & moves) + 1).tolist()
    ]
