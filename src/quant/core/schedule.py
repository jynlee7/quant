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
from quant.core.errors import ConfigError
from quant.core.ids import decision_id


@dataclass(frozen=True)
class Timing:
    feed_delay: timedelta = timedelta(minutes=15)
    decision_delay: timedelta = timedelta(minutes=16)

    def __post_init__(self) -> None:
        if self.feed_delay < timedelta(0):
            raise ConfigError(f"feed_delay {self.feed_delay} is negative")
        if self.decision_delay <= self.feed_delay:
            raise ConfigError(
                f"decision_delay {self.decision_delay} must exceed feed_delay {self.feed_delay}, "
                "or a decision could not see the bar it is for"
            )

    def available_at(self, t_close: datetime) -> datetime:
        return t_close + self.feed_delay

    def decision_time(self, t_close: datetime) -> datetime:
        return t_close + self.decision_delay


@dataclass(frozen=True)
class Decision:
    bar: BarSpec
    tau: datetime
    reduce_only: bool
    last_fillable: bool
    executes_at: datetime

    @property
    def id(self) -> str:
        return decision_id(self.tau)


def decision_schedule(
    calendar: TradingCalendar, day: date, timing: Timing | None = None
) -> tuple[Decision, ...]:
    """The session's decisions, in order."""
    timing = timing or Timing()
    session = calendar.session(day)
    next_open = calendar.session(calendar.next_session(day)).open
    taus = [timing.decision_time(bar.t_close) for bar in session.bars]
    fillable = [tau < session.close for tau in taus]
    last_fillable = max((i for i, ok in enumerate(fillable) if ok), default=None)
    return tuple(
        Decision(
            bar=bar,
            tau=tau,
            reduce_only=not ok,
            last_fillable=i == last_fillable,
            executes_at=tau if ok else next_open,
        )
        for i, (bar, tau, ok) in enumerate(zip(session.bars, taus, fillable, strict=True))
    )
