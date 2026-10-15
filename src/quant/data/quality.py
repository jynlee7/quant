"""The data-quality report on market.db (PLAN B6, CP1; ARCHITECTURE §6.1).

For one symbol and a span of sessions it lists:

- the sessions the calendar expects, and its half days;
- the gaps: a session with no minutes at all, and each run of consecutive missing minutes in a
  session that has some (a minute is missing when the session map has it and the store
  doesn't);
- the hourly bars left incomplete (their last minute is missing), which decisions treat as
  missing;
- the move flags that stand (a one-minute move over 5%, used but alerted).

Every gap must be explained before CP1. Explanations are given by gap key: the session's ISO
date for a missing session, and the first missing minute's UTC time (`2017-11-24T15:15:00Z`)
for a run of missing minutes. A gap with no explanation is listed in `unexplained`, and the
report is `ok` only when there are none.

The report reads the store's latest versions. It checks data, not a strategy, so it reads every
segment, the lockbox and DEV included; it computes no P&L.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from quant.core.calendar import TradingCalendar
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
        raise NotImplementedError("B6: Gap.key")


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
    incomplete_bars: tuple[datetime, ...]  # t_open of each hourly bar missing its last minute
    flags: tuple[DataFlag, ...]

    @property
    def unexplained(self) -> tuple[Gap, ...]:
        raise NotImplementedError("B6: QualityReport.unexplained")

    @property
    def ok(self) -> bool:
        """True when every gap is explained."""
        raise NotImplementedError("B6: QualityReport.ok")

    def to_json(self) -> dict[str, Any]:
        """The report as plain data for canonical JSON (`quant.core.canonical`), for CP1."""
        raise NotImplementedError("B6: QualityReport.to_json")


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
    raise NotImplementedError("B6: quality_report")
