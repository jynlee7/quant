"""B2: hourly bars built from SIP minutes (spec §2, ARCHITECTURE §6.1)."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from quant.core.calendar import Slot, TradingCalendar
from quant.core.epoch import to_us
from quant.core.errors import DataError
from quant.core.schedule import Timing
from quant.data.bars import build_hourly
from quant.data.frames import MinuteFrame
from quant.data.synthetic import synthetic_minutes
from quant.data.validate import validate_minutes

NY = ZoneInfo("America/New_York")
CAL = TradingCalendar.xnys


def et(day: date, hh: int, mm: int) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=NY).astimezone(UTC)


def day_minutes(day: date, seed: int = 1) -> MinuteFrame:
    return synthetic_minutes(CAL(), day, day, seed=seed)


@pytest.mark.spec("D1", "SPEC§2", "ARCH§6.1")
def test_a_full_synthetic_day_builds_seven_correct_bars() -> None:
    day = date(2019, 7, 2)
    minutes = day_minutes(day)
    assert len(minutes) == 390
    bars = build_hourly(CAL(), minutes, Timing())
    assert [b.slot for b in bars] == [
        Slot.S0930,
        Slot.S1030,
        Slot.S1130,
        Slot.S1230,
        Slot.S1330,
        Slot.S1430,
        Slot.CLOSE,
    ]
    starts = [et(day, 9, 30) + timedelta(hours=i) for i in range(7)]
    for bar, start in zip(bars, starts, strict=True):
        inside = (minutes.t >= to_us(start)) & (minutes.t < to_us(bar.t_close))
        assert bar.t_open == start
        assert bar.open == minutes.open[inside][0]
        assert bar.close == minutes.close[inside][-1]
        assert bar.high == minutes.high[inside].max()
        assert bar.low == minutes.low[inside].min()
        assert bar.volume == minutes.volume[inside].sum()
        assert bar.minutes == inside.sum()
        assert bar.complete
        assert bar.available_at == bar.t_close + timedelta(minutes=15)
    assert [b.minutes for b in bars] == [60] * 6 + [30]
    assert bars[-1].t_close == et(day, 16, 0)


@pytest.mark.spec("D1", "SPEC§2", "ARCH§6.1")
def test_a_half_day_builds_four_bars() -> None:
    bars = build_hourly(CAL(), day_minutes(date(2019, 11, 29)), Timing())
    assert [b.slot for b in bars] == [Slot.S0930, Slot.S1030, Slot.S1130, Slot.CLOSE]
    assert [b.minutes for b in bars] == [60, 60, 60, 30]


@pytest.mark.spec("SPEC§2", "ARCH§6.1")
def test_a_missing_last_minute_leaves_its_bar_incomplete() -> None:
    day = date(2019, 7, 2)
    minutes = day_minutes(day)
    last_of_first_bar = to_us(et(day, 10, 29))
    middle_of_second_bar = to_us(et(day, 11, 0))
    bars = build_hourly(
        CAL(),
        minutes.select((minutes.t != last_of_first_bar) & (minutes.t != middle_of_second_bar)),
        Timing(),
    )
    assert [b.complete for b in bars] == [False] + [True] * 6
    assert bars[0].minutes == 59
    assert bars[1].minutes == 59  # a gap mid-bar is noted, but the bar is complete


@pytest.mark.spec("SPEC§2", "ARCH§6.1")
@pytest.mark.parametrize(
    "corruption",
    [
        "high-below-open",
        "low-above-close",
        "zero-price",
        "negative-volume",
        "duplicate",
        "unsorted",
        "before-open",
        "at-close",
        "on-a-holiday",
    ],
)
def test_validation_rejects_bad_minutes(corruption: str) -> None:
    day = date(2019, 7, 2)
    m = day_minutes(day)
    t, o, h, lo, c, v = (x.copy() for x in (m.t, m.open, m.high, m.low, m.close, m.volume))
    if corruption == "high-below-open":
        h[5] = o[5] - 0.01
    elif corruption == "low-above-close":
        lo[5] = c[5] + 0.01
    elif corruption == "zero-price":
        lo[5] = 0.0
    elif corruption == "negative-volume":
        v[5] = -1.0
    elif corruption == "duplicate":
        t[6] = t[5]
    elif corruption == "unsorted":
        t[5], t[6] = t[6], t[5]
    elif corruption == "before-open":
        t[0] = to_us(et(day, 9, 29))
    elif corruption == "at-close":
        t[-1] = to_us(et(day, 16, 0))
    elif corruption == "on-a-holiday":
        t[:] = t + to_us(datetime(1970, 1, 3, tzinfo=UTC))  # 2019-07-04
    with pytest.raises(DataError):
        validate_minutes(CAL(), MinuteFrame(t, o, h, lo, c, v))


@pytest.mark.spec("SPEC§2", "ARCH§6.1")
def test_a_one_minute_move_over_five_percent_is_flagged_but_kept() -> None:
    m = day_minutes(date(2019, 7, 2))
    o, h, lo, c = (x.copy() for x in (m.open, m.high, m.low, m.close))
    for column in (o, h, lo, c):  # the price level jumps 6% at minute 100 and stays there
        column[100:] *= 1.06
    flags = validate_minutes(CAL(), MinuteFrame(m.t, o, h, lo, c, m.volume))
    assert [(f.kind, f.t) for f in flags] == [("move_over_5pct", m.at(100))]
    assert validate_minutes(CAL(), m) == []


@pytest.mark.full
@pytest.mark.spec("ARCH§6.1")
def test_ten_years_of_minutes_build_in_under_a_minute() -> None:
    started = time.perf_counter()
    minutes = synthetic_minutes(CAL(), date(2016, 1, 1), date(2025, 12, 31), seed=7)
    validate_minutes(CAL(), minutes)
    bars = build_hourly(CAL(), minutes, Timing())
    elapsed = time.perf_counter() - started
    assert len(minutes) == 976_680
    assert len(bars) == sum(
        len(CAL().session(d).bars) for d in CAL().sessions(date(2016, 1, 1), date(2025, 12, 31))
    )
    assert all(b.complete for b in bars)
    assert np.isfinite([b.close for b in bars]).all()
    assert elapsed < 60, f"{elapsed:.1f}s"
