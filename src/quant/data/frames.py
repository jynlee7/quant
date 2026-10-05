"""Bars as the data layer passes them around.

Minutes come in hundreds of thousands, so they travel as columns (`MinuteFrame`). Hourly bars
number about 19,000 in ten years, so each is a small immutable object (`HourlyBar`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import numpy.typing as npt

from quant.core.calendar import Slot
from quant.core.epoch import from_us
from quant.core.errors import DataError

Ints = npt.NDArray[np.int64]
Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MinuteFrame:
    """Minute bars as aligned columns. `t` is each minute's start, in epoch microseconds UTC."""

    t: Ints
    open: Floats
    high: Floats
    low: Floats
    close: Floats
    volume: Floats

    def __post_init__(self) -> None:
        n = len(self.t)
        if self.t.dtype != np.int64 or self.t.ndim != 1:
            raise DataError(f"minute times must be a 1-D int64 array, not {self.t.dtype}")
        for name in ("open", "high", "low", "close", "volume"):
            column = getattr(self, name)
            if column.dtype != np.float64 or column.ndim != 1 or len(column) != n:
                raise DataError(f"minute {name} must be a 1-D float64 array of length {n}")

    def __len__(self) -> int:
        return len(self.t)

    def at(self, i: int) -> datetime:
        """The start of minute `i`."""
        return from_us(int(self.t[i]))

    def select(self, mask: npt.NDArray[np.bool_]) -> MinuteFrame:
        return MinuteFrame(
            self.t[mask],
            self.open[mask],
            self.high[mask],
            self.low[mask],
            self.close[mask],
            self.volume[mask],
        )

    def with_close(self, i: int, close: float) -> MinuteFrame:
        """A copy with minute `i` closing at `close`, its high and low widened to fit."""
        c, h, lo = self.close.copy(), self.high.copy(), self.low.copy()
        c[i] = close
        h[i] = max(h[i], close)
        lo[i] = min(lo[i], close)
        return MinuteFrame(self.t.copy(), self.open.copy(), h, lo, c, self.volume.copy())

    @classmethod
    def empty(cls) -> MinuteFrame:
        floats = np.empty(0, dtype=np.float64)
        return cls(np.empty(0, dtype=np.int64), floats, floats, floats, floats, floats)


@dataclass(frozen=True)
class HourlyBar:
    """One bar of the session map, built from its minutes. Times are UTC.

    `version` and `known_at` are set when the bar is stored: version 1 is the first build,
    each revision adds one, and `known_at` is when that version was known.
    """

    session: date
    slot: Slot
    t_open: datetime
    t_close: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    minutes: int
    complete: bool
    available_at: datetime
    version: int = 1
    known_at: datetime | None = None
