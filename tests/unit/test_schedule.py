"""B1: bars, slots and decision times on the XNYS calendar (spec §2, ARCHITECTURE §5)."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from quant.core.calendar import Slot, TradingCalendar
from quant.core.errors import ConfigError
from quant.core.schedule import Timing, decision_schedule

NY = ZoneInfo("America/New_York")


def et(day: date, hh: int, mm: int) -> datetime:
    return datetime.combine(day, time(hh, mm), NY).astimezone(UTC)


def xnys() -> TradingCalendar:
    return TradingCalendar.xnys()


def taus_et(day: date) -> list[str]:
    return [d.tau.astimezone(NY).strftime("%H:%M") for d in decision_schedule(xnys(), day)]


@pytest.mark.spec("D1", "D2", "SPEC§2", "ARCH§5")
def test_a_full_day_has_seven_bars_and_seven_decisions() -> None:
    day = date(2019, 7, 2)
    session = xnys().session(day)
    assert [b.slot for b in session.bars] == [
        Slot.S0930,
        Slot.S1030,
        Slot.S1130,
        Slot.S1230,
        Slot.S1330,
        Slot.S1430,
        Slot.CLOSE,
    ]
    assert session.bars[0].t_open == et(day, 9, 30)
    assert session.bars[-1].t_open == et(day, 15, 30)
    assert session.bars[-1].t_close == et(day, 16, 0)
    assert taus_et(day) == ["10:46", "11:46", "12:46", "13:46", "14:46", "15:46", "16:16"]


@pytest.mark.spec("D1", "D2", "A3", "SPEC§2", "ARCH§5")
@pytest.mark.parametrize(
    ("day", "next_open_day"),
    [
        (date(2019, 7, 3), date(2019, 7, 5)),
        (date(2019, 11, 29), date(2019, 12, 2)),
        (date(2019, 12, 24), date(2019, 12, 26)),
    ],
)
def test_a_half_day_has_four_bars_and_the_last_fillable_decision_at_1246(
    day: date, next_open_day: date
) -> None:
    session = xnys().session(day)
    assert [b.slot for b in session.bars] == [Slot.S0930, Slot.S1030, Slot.S1130, Slot.CLOSE]
    assert session.bars[-1].t_open == et(day, 12, 30)
    assert session.bars[-1].t_close == et(day, 13, 0)
    assert session.is_half_day

    decisions = decision_schedule(xnys(), day)
    assert taus_et(day) == ["10:46", "11:46", "12:46", "13:16"]
    assert [d.last_fillable for d in decisions] == [False, False, True, False]
    assert [d.reduce_only for d in decisions] == [False, False, False, True]
    assert decisions[-1].executes_at == et(next_open_day, 9, 30)


@pytest.mark.spec("D2", "A3")
def test_the_1546_decision_is_the_last_that_can_fill_and_1616_only_reduces() -> None:
    day = date(2019, 7, 2)
    decisions = decision_schedule(xnys(), day)
    assert [d.last_fillable for d in decisions] == [False] * 5 + [True, False]
    assert [d.reduce_only for d in decisions] == [False] * 6 + [True]
    assert all(d.executes_at == d.tau for d in decisions[:-1])
    assert decisions[-1].executes_at == et(date(2019, 7, 3), 9, 30)


@pytest.mark.spec("D2", "ARCH§5")
@pytest.mark.parametrize(
    ("before", "after"),
    [(date(2019, 3, 8), date(2019, 3, 11)), (date(2019, 11, 1), date(2019, 11, 4))],
    ids=["spring-forward", "fall-back"],
)
def test_decision_times_hold_in_new_york_time_across_daylight_saving(
    before: date, after: date
) -> None:
    assert taus_et(before) == taus_et(after)
    first_before = decision_schedule(xnys(), before)[0].tau
    first_after = decision_schedule(xnys(), after)[0].tau
    assert first_before.time() != first_after.time()  # the UTC time moves by an hour
    assert decision_schedule(xnys(), before)[-1].executes_at == et(after, 9, 30)


@pytest.mark.spec("D1")
def test_exchange_holidays_and_special_closures_are_not_sessions() -> None:
    cal = xnys()
    for closed in (date(2018, 12, 5), date(2025, 1, 9), date(2019, 7, 4), date(2019, 12, 25)):
        assert not cal.is_session(closed)
    assert cal.next_session(date(2019, 7, 3)) == date(2019, 7, 5)


@pytest.mark.spec("D1", "D2", "A3", "ARCH§5")
def test_every_session_from_2016_to_2026_builds_a_schedule() -> None:
    cal = xnys()
    days = cal.sessions(date(2016, 1, 1), date(2026, 12, 31))
    assert len(days) == 2765
    half_days = 0
    previous_tau = None
    for day in days:
        session = cal.session(day)
        decisions = decision_schedule(cal, day)
        assert len(session.bars) in (4, 7)
        half_days += session.is_half_day
        assert session.bars[0].t_open == session.open
        assert session.bars[-1].t_close == session.close
        assert session.bars[-1].slot is Slot.CLOSE
        assert session.bars[-1].t_close - session.bars[-1].t_open == timedelta(minutes=30)
        assert sum(d.last_fillable for d in decisions) == 1
        assert [d.reduce_only for d in decisions] == [False] * (len(decisions) - 1) + [True]
        for decision in decisions:
            assert previous_tau is None or decision.tau > previous_tau
            previous_tau = decision.tau
    assert half_days == 23


@pytest.mark.spec("D2")
def test_the_decision_delay_must_exceed_the_feed_delay() -> None:
    with pytest.raises(ConfigError):
        Timing(feed_delay=timedelta(minutes=15), decision_delay=timedelta(minutes=15))
    timing = Timing(feed_delay=timedelta(minutes=20), decision_delay=timedelta(minutes=21))
    decisions = decision_schedule(xnys(), date(2019, 7, 2), timing)
    assert decisions[0].tau.astimezone(NY).strftime("%H:%M") == "10:51"
