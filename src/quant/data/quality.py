"""The data-quality report on market.db (PLAN B6, CP1; ARCHITECTURE §6.1).

For one symbol and a span of sessions it lists:

- the sessions the calendar expects, and its half days;
- the gaps: a session with no minutes at all, and each run of consecutive missing minutes in a
  session that has some (a minute is missing when the session map has it and the store
  doesn't);
- every hourly slot of the session map left missing or incomplete (its last minute isn't
  stored), which decisions treat as missing: a slot with no minutes at all, and every slot of a
  missing session, included;
- the move flags that stand (a one-minute move over 5%, used but alerted).

Every gap must be explained before CP1. Explanations are given by gap key: the session's ISO
date for a missing session, and the first missing minute's UTC time (`2017-11-24T15:15:00Z`)
for a run of missing minutes. A gap with no explanation is listed in `unexplained`, and the
report is `ok` only when there are none.

The report reads which minutes the store holds (`BarStore.minute_times`), never their prices. It
checks data, not a strategy, so it covers every segment, the lockbox and DEV included; it
computes no P&L.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from typing import Any

import numpy as np

from quant.core.calendar import TradingCalendar
from quant.core.epoch import MINUTE_US, from_us, to_us
from quant.data.market_db import BarStore
from quant.data.validate import DataFlag


class GapKind(StrEnum):
    MISSING_SESSION = "missing_session"
    MISSING_MINUTES = "missing_minutes"


@dataclass(frozen=True)
class Gap:
    kind: GapKind
    session: date
    start: datetime  # the first missing minute (the session's open, for a missing session)
    minutes: int  # how many minutes are missing
    explanation: str | None

    @property
    def key(self) -> str:
        """The key an explanation is given by: the session's ISO date for a missing session,
        the first missing minute in UTC (`...Z`) for a run of minutes."""
        if self.kind is GapKind.MISSING_SESSION:
            return self.session.isoformat()
        return self.start.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class QualityReport:
    symbol: str
    first: date
    last: date
    sessions: tuple[date, ...]  # every session the calendar expects in [first, last]
    half_days: tuple[date, ...]
    minutes_expected: int
    minutes_present: int
    gaps: tuple[Gap, ...]  # in time order
    incomplete_bars: tuple[datetime, ...]  # t_open of each hourly slot missing its last minute
    flags: tuple[DataFlag, ...]

    @property
    def unexplained(self) -> tuple[Gap, ...]:
        return tuple(g for g in self.gaps if not g.explanation or not g.explanation.strip())

    @property
    def ok(self) -> bool:
        """True when every gap is explained."""
        return not self.unexplained

    def to_json(self) -> dict[str, Any]:
        """The report as plain data for canonical JSON (`quant.core.canonical`), for CP1."""
        return {
            **asdict(self),
            "gaps": [{**asdict(g), "key": g.key} for g in self.gaps],
            "ok": self.ok,
            "unexplained": [g.key for g in self.unexplained],
        }


def quality_report(
    store: BarStore,
    symbol: str,
    first: date,
    last: date,
    *,
    explanations: Mapping[str, str] | None = None,
    calendar: TradingCalendar | None = None,
) -> QualityReport:
    """The quality of `symbol`'s stored minutes over the sessions from `first` to `last`."""
    calendar = calendar or TradingCalendar.xnys()
    days = calendar.sessions(first, last)
    gaps: list[Gap] = []
    half_days: list[date] = []
    incomplete: list[datetime] = []
    expected = present = 0
    for day in days:
        session = calendar.session(day)
        if session.is_half_day:
            half_days.append(day)
        times = np.arange(to_us(session.open), to_us(session.close), MINUTE_US, dtype=np.int64)
        stored = store.minute_times(symbol, session.open, session.close)
        expected += len(times)
        present += len(stored)
        lasts = np.array([to_us(bar.t_close) - MINUTE_US for bar in session.bars], dtype=np.int64)
        incomplete.extend(
            bar.t_open
            for bar, held in zip(session.bars, np.isin(lasts, stored).tolist(), strict=True)
            if not held
        )
        if not len(stored):
            gaps.append(Gap(GapKind.MISSING_SESSION, day, session.open, len(times), None))
            continue
        missing = np.setdiff1d(times, stored)
        if len(missing):
            for run in np.split(missing, np.flatnonzero(np.diff(missing) != MINUTE_US) + 1):
                gaps.append(Gap(GapKind.MISSING_MINUTES, day, from_us(int(run[0])), len(run), None))
    gaps = [replace(g, explanation=(explanations or {}).get(g.key)) for g in gaps]
    flags = tuple(
        f
        for f in store.flags(symbol)
        if days and calendar.session(days[0]).open <= f.t < calendar.session(days[-1]).close
    )
    return QualityReport(
        symbol,
        first,
        last,
        tuple(days),
        tuple(half_days),
        expected,
        present,
        tuple(gaps),
        tuple(incomplete),
        flags,
    )
