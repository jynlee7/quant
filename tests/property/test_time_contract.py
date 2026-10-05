"""B1: the time contract behind I1. A bar becomes visible strictly before the decision that
uses it, and the next bar is never visible to it."""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant.core.calendar import TradingCalendar
from quant.core.schedule import Timing, decision_schedule

pytestmark = pytest.mark.xfail(strict=True, reason="red:B1", raises=NotImplementedError)


@pytest.mark.spec("I1", "D2", "SPEC§2")
@given(day=st.dates(min_value=date(2016, 1, 4), max_value=date(2026, 12, 31)))
def test_a_bar_is_available_before_its_decision_and_the_next_bar_is_not(day: date) -> None:
    cal = TradingCalendar.xnys()
    if not cal.is_session(day):
        day = cal.next_session(day)
    timing = Timing()
    decisions = decision_schedule(cal, day, timing)
    bars = cal.session(day).bars
    for i, decision in enumerate(decisions):
        assert decision.bar == bars[i]
        assert decision.tau > decision.bar.t_close
        assert timing.available_at(decision.bar.t_close) < decision.tau
        if i + 1 < len(bars):
            assert not timing.available_at(bars[i + 1].t_close) < decision.tau
