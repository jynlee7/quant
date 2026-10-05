"""The XNYS trading calendar and the session map (spec §2, ARCHITECTURE §5 and §6.1).

Bars are anchored at the 09:30 ET open: hourly bars, then a final 30-minute bar. A full day has
seven bars and a 13:00 half day four. The final bar maps to the CLOSE slot on both, so the
slot-normalized features (F1, F3, F4) compare like with like.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class Slot(StrEnum):
    """A bar's place in the session, named by its start time in New York."""

    S0930 = "09:30"
    S1030 = "10:30"
    S1130 = "11:30"
    S1230 = "12:30"
    S1330 = "13:30"
    S1430 = "14:30"
    CLOSE = "CLOSE"


@dataclass(frozen=True)
class BarSpec:
    """Where one hourly bar sits in time. Times are UTC."""

    session: date
    index: int
    slot: Slot
    t_open: datetime
    t_close: datetime


@dataclass(frozen=True)
class Session:
    day: date
    open: datetime
    close: datetime
    bars: tuple[BarSpec, ...]

    @property
    def is_half_day(self) -> bool:
        raise NotImplementedError("red:B1")


class TradingCalendar:
    """Sessions of one exchange, built once from exchange_calendars."""

    @classmethod
    def xnys(cls) -> TradingCalendar:
        """The NYSE calendar, shared by every caller."""
        raise NotImplementedError("red:B1")

    def is_session(self, day: date) -> bool:
        raise NotImplementedError("red:B1")

    def sessions(self, start: date, end: date) -> list[date]:
        """Sessions from `start` to `end`, both included."""
        raise NotImplementedError("red:B1")

    def session(self, day: date) -> Session:
        raise NotImplementedError("red:B1")

    def next_session(self, day: date) -> date:
        """The first session after `day`."""
        raise NotImplementedError("red:B1")
