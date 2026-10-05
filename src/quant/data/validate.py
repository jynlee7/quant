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

from quant.core.calendar import BarSpec, TradingCalendar
from quant.core.epoch import MINUTE_US, from_us
from quant.core.errors import DataError
from quant.data.bars import map_minutes
from quant.data.frames import Ints, MinuteFrame

MOVE_FLAG = 0.05
MOVE_KIND = "move_over_5pct"


@dataclass(frozen=True)
class DataFlag:
    t: datetime
    kind: str
    detail: str


def _fail_where(bad: npt.ArrayLike, t: Ints, why: str) -> None:
    mask = np.asarray(bad, dtype=bool)
    if mask.any():
        i = int(np.flatnonzero(mask)[0])
        raise DataError(f"minute {from_us(int(t[i])).isoformat()}: {why}")


def check_minutes(calendar: TradingCalendar, minutes: MinuteFrame) -> tuple[list[BarSpec], Ints]:
    """Raise DataError on bad minutes; return where they fall in the session map (`map_minutes`)."""
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
    return map_minutes(calendar, t)


def validate_minutes(calendar: TradingCalendar, minutes: MinuteFrame) -> list[DataFlag]:
    """Raise DataError on bad minutes; return flags on big moves."""
    if len(minutes) == 0:
        return []
    return _move_flags(minutes, *check_minutes(calendar, minutes))


def move_flags(calendar: TradingCalendar, minutes: MinuteFrame) -> list[DataFlag]:
    """Flags on the big moves in valid minutes, such as a stored session with new ones laid
    over it."""
    return _move_flags(minutes, *map_minutes(calendar, minutes.t))


def _move_flags(minutes: MinuteFrame, specs: list[BarSpec], which: Ints) -> list[DataFlag]:
    if len(minutes) < 2:
        return []
    t, c = minutes.t, minutes.close
    session = np.array([spec.session.toordinal() for spec in specs], dtype=np.int64)[which]
    big = (session[1:] == session[:-1]) & (np.abs(c[1:] / c[:-1] - 1.0) > MOVE_FLAG)
    return [
        DataFlag(
            from_us(int(t[i])),
            MOVE_KIND,
            f"close {c[i - 1]} -> {c[i]} over {(t[i] - t[i - 1]) // MINUTE_US} min",
        )
        for i in (np.flatnonzero(big) + 1).tolist()
    ]
