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

    def __len__(self) -> int:
        raise NotImplementedError("red:B2")

    def at(self, i: int) -> datetime:
        """The start of minute `i`."""
        raise NotImplementedError("red:B2")

    def select(self, mask: npt.NDArray[np.bool_]) -> MinuteFrame:
        raise NotImplementedError("red:B2")

    def with_close(self, i: int, close: float) -> MinuteFrame:
        """A copy with minute `i` closing at `close`, its high and low widened to fit."""
        raise NotImplementedError("red:B2")

    @classmethod
    def empty(cls) -> MinuteFrame:
        raise NotImplementedError("red:B2")


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
