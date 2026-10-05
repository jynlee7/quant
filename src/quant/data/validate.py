"""Checks on minute bars before they are stored (ARCHITECTURE §6.1).

Bad data is refused: prices out of order, non-positive or non-finite, negative volume, duplicate
or unsorted minutes, minutes outside a session. A one-minute move over 5% is kept but flagged,
because it can be real (it is used, and it raises an alert).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant.core.calendar import TradingCalendar
from quant.data.frames import MinuteFrame


@dataclass(frozen=True)
class DataFlag:
    t: datetime
    kind: str
    detail: str


def validate_minutes(calendar: TradingCalendar, minutes: MinuteFrame) -> list[DataFlag]:
    """Raise DataError on bad minutes; return flags on suspicious ones."""
    raise NotImplementedError("red:B2")
