"""B1: the Clock port is the only source of "now" (K6), and time inside the code is UTC."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from quant.core.clock import SimClock, WallClock, ensure_utc
from quant.core.errors import ClockError

START = datetime(2019, 7, 2, 14, 46, tzinfo=UTC)


@pytest.mark.spec("K6", "I1")
def test_the_simulated_clock_moves_only_when_told_and_never_backwards() -> None:
    clock = SimClock(START)
    assert clock.now() == START
    assert clock.now() == START
    clock.advance(timedelta(minutes=60))
    assert clock.now() == START + timedelta(minutes=60)
    clock.sleep_until(START + timedelta(hours=2))
    assert clock.now() == START + timedelta(hours=2)
    with pytest.raises(ClockError):
        clock.advance_to(START)


@pytest.mark.spec("K6")
def test_the_wall_clock_reads_utc() -> None:
    now = WallClock().now()
    assert now.tzinfo is UTC
    assert now.year >= 2026


@pytest.mark.spec("K6", "I1")
def test_naive_datetimes_are_rejected_and_aware_ones_converted_to_utc() -> None:
    with pytest.raises(ClockError):
        ensure_utc(datetime(2019, 7, 2, 10, 46))  # noqa: DTZ001
    with pytest.raises(ClockError):
        SimClock(datetime(2019, 7, 2, 10, 46))  # noqa: DTZ001
    eastern = datetime(2019, 7, 2, 10, 46, tzinfo=timezone(timedelta(hours=-4)))
    assert ensure_utc(eastern) == START
    assert ensure_utc(eastern).tzinfo is UTC
