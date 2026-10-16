"""The XNYS trading calendar and the session map (spec §2, ARCHITECTURE §5 and §6.1).

Bars are anchored at the 09:30 ET open: hourly bars, then a final 30-minute bar. A full day has
seven bars and a 13:00 half day four. The final bar maps to the CLOSE slot on both, so the
slot-normalized features (F1, F3, F4) compare like with like.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from functools import cache
from typing import Any
from zoneinfo import ZoneInfo

from quant.core.clock import ensure_utc
from quant.core.errors import CalendarError

NEW_YORK = ZoneInfo("America/New_York")
OPEN_ET = time(9, 30)
HOUR_BAR = timedelta(hours=1)
CLOSE_BAR = timedelta(minutes=30)
FULL_DAY = timedelta(hours=6, minutes=30)

# The span the calendar is built for: the 2016 history with a margin, and far enough ahead that
# no running engine meets the end. Future holidays come from exchange_calendars' rules; the
# engine cross-checks them against Alpaca's calendar every day (ARCHITECTURE §5).
CALENDAR_START = date(2015, 12, 1)
CALENDAR_END = date(2040, 12, 31)


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
        return self.close - self.open < FULL_DAY


def _bars(day: date, open_utc: datetime, close_utc: datetime) -> tuple[BarSpec, ...]:
    """Hourly bars from the open, then the 30-minute CLOSE bar. Any other shape is an error,
    because the features assume it: an unknown session must not be traded as if it were known."""
    if open_utc.astimezone(NEW_YORK).time() != OPEN_ET:
        raise CalendarError(f"{day} opens at {open_utc.astimezone(NEW_YORK):%H:%M} ET, not 09:30")
    hours, rest = divmod(close_utc - open_utc, HOUR_BAR)
    if rest != CLOSE_BAR or hours < 1:
        raise CalendarError(
            f"{day} lasts {close_utc - open_utc}, which isn't whole hours plus a 30-minute close"
        )
    bars = []
    for index in range(hours):
        t_open = open_utc + index * HOUR_BAR
        slot = Slot(f"{t_open.astimezone(NEW_YORK):%H:%M}")
        bars.append(BarSpec(day, index, slot, t_open, t_open + HOUR_BAR))
    last_open = open_utc + hours * HOUR_BAR
    bars.append(BarSpec(day, hours, Slot.CLOSE, last_open, close_utc))
    return tuple(bars)


def _utc(timestamp: Any) -> datetime:
    """A pandas Timestamp from exchange_calendars as a plain UTC datetime."""
    value: datetime = timestamp.to_pydatetime()
    return value.astimezone(UTC)


class TradingCalendar:
    """Sessions of one exchange, built once from exchange_calendars."""

    def __init__(self, opens_closes: dict[date, tuple[datetime, datetime]]) -> None:
        self._hours = opens_closes
        self._days = sorted(opens_closes)
        self._sessions: dict[date, Session] = {}

    @classmethod
    def xnys(cls) -> TradingCalendar:
        """The NYSE calendar, shared by every caller."""
        return _xnys()

    @classmethod
    def from_exchange_calendar(cls, calendar: Any) -> TradingCalendar:
        hours = {
            row.Index.date(): (_utc(row.open), _utc(row.close))
            for row in calendar.schedule.itertuples()
        }
        return cls(hours)

    def _check_range(self, day: date) -> None:
        if not CALENDAR_START <= day <= CALENDAR_END:
            raise CalendarError(
                f"{day} is outside the calendar ({CALENDAR_START} to {CALENDAR_END})"
            )

    def is_session(self, day: date) -> bool:
        self._check_range(day)
        return day in self._hours

    def sessions(self, start: date, end: date) -> list[date]:
        """Sessions from `start` to `end`, both included."""
        self._check_range(start)
        self._check_range(end)
        lo = bisect.bisect_left(self._days, start)
        hi = bisect.bisect_right(self._days, end)
        return self._days[lo:hi]

    def session(self, day: date) -> Session:
        if not self.is_session(day):
            raise CalendarError(f"{day} is not a trading session")
        if day not in self._sessions:
            open_utc, close_utc = self._hours[day]
            self._sessions[day] = Session(day, open_utc, close_utc, _bars(day, open_utc, close_utc))
        return self._sessions[day]

    def next_session(self, day: date) -> date:
        """The first session after `day`."""
        self._check_range(day)
        index = bisect.bisect_right(self._days, day)
        if index == len(self._days):
            raise CalendarError(f"no session after {day} before {CALENDAR_END}")
        return self._days[index]

    def next_open(self, at: datetime) -> datetime:
        """The open, in UTC, of the first session that opens strictly after `at`: when a refit,
        promotion or rollback recorded at `at` takes effect (C7, ARCHITECTURE §9.2), never
        mid-session. During a session, or after its close, that is the next session's open,
        across weekends and holidays; before a session day's open, that day's open; exactly at
        an open, the next session's. Raises ClockError for a naive `at`, and CalendarError when
        `at`'s New York date is outside the calendar or no session opens after it before
        CALENDAR_END."""
        at = ensure_utc(at)
        day = at.astimezone(NEW_YORK).date()
        self._check_range(day)
        if self.is_session(day) and self.session(day).open > at:
            return self.session(day).open
        return self.session(self.next_session(day)).open


@cache
def _xnys() -> TradingCalendar:
    import exchange_calendars

    calendar = exchange_calendars.get_calendar(
        "XNYS", start=CALENDAR_START.isoformat(), end=CALENDAR_END.isoformat()
    )
    return TradingCalendar.from_exchange_calendar(calendar)
