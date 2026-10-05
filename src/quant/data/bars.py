"""The hourly bar builder (spec §2, ARCHITECTURE §6.1).

Each bar of the session map is built from its SIP minutes: the first open, the highest high,
the lowest low, the last close and the summed volume. A bar is complete when its last minute is
present; a bar missing that minute is kept but marked incomplete, and decisions treat it as
missing.
"""

from __future__ import annotations

from quant.core.calendar import TradingCalendar
from quant.core.schedule import Timing
from quant.data.frames import HourlyBar, MinuteFrame


def build_hourly(
    calendar: TradingCalendar, minutes: MinuteFrame, timing: Timing
) -> list[HourlyBar]:
    """Hourly bars for every bar of the session map that has at least one minute."""
    raise NotImplementedError("red:B2")
