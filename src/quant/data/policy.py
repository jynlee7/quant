"""Data segments and purpose-tagged reads (A2, C1, ARCHITECTURE §6.3).

| Segment | Span | May be read to |
|---|---|---|
| FIT | 2016-01 to 2018-12 | operate, score, report |
| DEV | 2019-01 to 2024-12 | operate, score, report |
| LOCKBOX | 2025-01 to the seal | operate only (features, the filter, refits: C1) |
| FORWARD | after the seal | operate and report (parity, reports), never score |

Only the go-live lockbox test may score the lockbox (B19); v1 has no way to.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class Segment(StrEnum):
    FIT = "FIT"
    DEV = "DEV"
    LOCKBOX = "LOCKBOX"
    FORWARD = "FORWARD"


class Purpose(StrEnum):
    """Why a read happens. Every read through the view names one."""

    OPERATE = "operate"  # features, the filter and refits
    SCORE = "score"  # strategy P&L, trades, gates and calibration results
    REPORT = "report"  # parity replays and reports


@dataclass(frozen=True)
class Segments:
    fit_start: date = date(2016, 1, 1)
    dev_start: date = date(2019, 1, 1)
    lockbox_start: date = date(2025, 1, 1)
    lockbox_end: date | None = None  # the seal: the day before the first paper decision

    def segment_of(self, day: date) -> Segment:
        raise NotImplementedError("red:B2")


@dataclass(frozen=True)
class DataPolicy:
    segments: Segments = Segments()

    def check(self, purpose: Purpose, first: date, last: date) -> None:
        """Raise unless `purpose` may read the days from `first` to `last`."""
        raise NotImplementedError("red:B2")
