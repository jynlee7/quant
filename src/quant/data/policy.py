"""Data segments and purpose-tagged reads (A2, C1, ARCHITECTURE §6.3).

| Segment | Span | May be read to |
|---|---|---|
| FIT | 2016-01 to 2018-12 | operate, score, calibrate, report |
| DEV | 2019-01 to 2024-12 | operate, report; score and calibrate only with a trial ticket that covers the read (C11) |
| LOCKBOX | 2025-01 to the seal | operate only (features, the filter, refits: C1) |
| FORWARD | after the seal | operate, calibrate and report; never score |

Calibration and Kelly edge estimates use the fit-window backtest, then DEV walk-forward history,
then forward data, and skip the lockbox (ARCHITECTURE §6.3). On DEV they come from a trial's
walk-forward, and Kelly's shadow returns are strategy P&L, so they need its ticket. Gates score
only FIT and DEV. Only the go-live lockbox test may score the lockbox (B19); v1 has no way to.
Days before the data starts can't be read for any purpose but operating, since no segment holds
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Protocol

from quant.core.errors import ConfigError, LockboxError, PolicyError


class Segment(StrEnum):
    FIT = "FIT"
    DEV = "DEV"
    LOCKBOX = "LOCKBOX"
    FORWARD = "FORWARD"


class Purpose(StrEnum):
    """Why a read happens. Every read through the view names one."""

    OPERATE = "operate"  # features, the filter and refits
    SCORE = "score"  # strategy P&L, trades and gate results
    CALIBRATE = "calibrate"  # calibration and Kelly edge estimates
    REPORT = "report"  # parity replays and reports


@dataclass(frozen=True)
class Segments:
    fit_start: date = date(2016, 1, 1)
    dev_start: date = date(2019, 1, 1)
    lockbox_start: date = date(2025, 1, 1)
    lockbox_end: date | None = None  # the seal: the day before the first paper decision

    def __post_init__(self) -> None:
        ordered = self.fit_start < self.dev_start < self.lockbox_start
        if not ordered or (self.lockbox_end is not None and self.lockbox_end < self.lockbox_start):
            raise ConfigError(f"data segments out of order: {self}")

    def spans(self) -> dict[Segment, tuple[date, date]]:
        """Each segment's first and last day. An unsealed lockbox runs to date.max."""
        spans = {
            Segment.FIT: (self.fit_start, self.dev_start - timedelta(days=1)),
            Segment.DEV: (self.dev_start, self.lockbox_start - timedelta(days=1)),
            Segment.LOCKBOX: (self.lockbox_start, self.lockbox_end or date.max),
        }
        if self.lockbox_end is not None:
            spans[Segment.FORWARD] = (self.lockbox_end + timedelta(days=1), date.max)
        return spans

    def segment_of(self, day: date) -> Segment:
        for segment, (first, last) in self.spans().items():
            if first <= day <= last:
                return segment
        raise PolicyError(f"{day} is before the data starts ({self.fit_start})")

    def overlapping(self, first: date, last: date) -> set[Segment]:
        return {s for s, (lo, hi) in self.spans().items() if lo <= last and first <= hi}


class ScoringTicket(Protocol):
    """What the policy reads from a `TrialTicket`: the trial, and the days it may score.

    Only the trial ledger issues tickets (B10, K5); this is the shape the data layer accepts.
    """

    @property
    def trial_id(self) -> str: ...

    @property
    def first_day(self) -> date: ...

    @property
    def last_day(self) -> date: ...


@dataclass(frozen=True)
class DataPolicy:
    segments: Segments = Segments()

    def check(
        self, purpose: Purpose, first: date, last: date, *, ticket: ScoringTicket | None = None
    ) -> None:
        """Raise unless `purpose` may read the days from `first` to `last`."""
        if last < first:
            raise PolicyError(f"empty span {first} to {last}")
        if purpose is Purpose.OPERATE:
            return  # market data may feed features, the filter and refits anywhere (C1)
        if first < self.segments.fit_start:
            raise PolicyError(
                f"a {purpose} read of {first} to {last} starts before the data "
                f"({self.segments.fit_start})"
            )
        touched = self.segments.overlapping(first, last)
        if Segment.LOCKBOX in touched:
            seal = self.segments.lockbox_end or "the seal"
            raise LockboxError(
                f"a {purpose} read of {first} to {last} touches the lockbox "
                f"({self.segments.lockbox_start} to {seal}); only the go-live test may (A2, C1)"
            )
        if purpose is Purpose.SCORE and Segment.FORWARD in touched:
            raise PolicyError(
                f"a score read of {first} to {last} touches forward data, which is for "
                "calibration, parity and reports only"
            )
        if purpose in (Purpose.SCORE, Purpose.CALIBRATE) and Segment.DEV in touched:
            dev_first, dev_last = self.segments.spans()[Segment.DEV]
            need = (max(first, dev_first), min(last, dev_last))
            if ticket is None:
                raise PolicyError(
                    f"a {purpose} read of {need[0]} to {need[1]} in DEV needs a trial ticket "
                    "from the ledger (C11, A2)"
                )
            if not ticket.first_day <= need[0] <= need[1] <= ticket.last_day:
                raise PolicyError(
                    f"trial {ticket.trial_id} may score {ticket.first_day} to "
                    f"{ticket.last_day}, not {need[0]} to {need[1]}"
                )
