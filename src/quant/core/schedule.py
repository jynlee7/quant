"""Decision times (D2, A3, spec §2, ARCHITECTURE §5).

A bar closing at `t_close` becomes available at `t_close + feed_delay` (15 minutes: free SIP data
is 15 minutes old). Its decision runs at tau = `t_close + decision_delay` (16 minutes), so the
bar is visible (available_at < tau) and the next one isn't. A decision whose tau falls before the
session close can fill; the last such decision applies the overnight cap (A3). A decision at or
after the close can only reduce exposure, and it executes at the next session's open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from quant.core.calendar import BarSpec, TradingCalendar


@dataclass(frozen=True)
class Timing:
    feed_delay: timedelta = timedelta(minutes=15)
    decision_delay: timedelta = timedelta(minutes=16)

    def __post_init__(self) -> None:
        raise NotImplementedError("red:B1")

    def available_at(self, t_close: datetime) -> datetime:
        raise NotImplementedError("red:B1")

    def decision_time(self, t_close: datetime) -> datetime:
        raise NotImplementedError("red:B1")


@dataclass(frozen=True)
class Decision:
    bar: BarSpec
    tau: datetime
    reduce_only: bool
    last_fillable: bool
    executes_at: datetime


def decision_schedule(
    calendar: TradingCalendar, day: date, timing: Timing | None = None
) -> tuple[Decision, ...]:
    """The session's decisions, in order."""
    raise NotImplementedError("red:B1")
