"""B2: the point-in-time view, the data half of I1 (spec §2, ARCHITECTURE §6.2).

A row is visible at decision time tau only if its bar was available strictly before tau, and it
shows the latest version known before as_of (tau in the backtest; later, live, once the bar has
been ingested).
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant.core.errors import ClockError
from quant.data.market_db import BarStore
from tests.support.bars import SYMBOL, store_with_days

DAY = date(2019, 7, 2)
US = timedelta(microseconds=1)


@pytest.mark.spec("I1", "SPEC§2", "ARCH§6.2")
def test_a_bar_available_exactly_at_tau_is_invisible() -> None:
    store = store_with_days(DAY, DAY)
    first = store.view(datetime(2019, 7, 3, tzinfo=UTC)).hourly(SYMBOL)[0]
    assert first.t_open == datetime(2019, 7, 2, 13, 30, tzinfo=UTC)
    tau = first.available_at
    assert first.t_open not in [b.t_open for b in store.view(tau).hourly(SYMBOL)]
    assert first.t_open in [b.t_open for b in store.view(tau + US).hourly(SYMBOL)]


@pytest.mark.spec("I1", "SPEC§2", "ARCH§6.2")
def test_a_closed_bar_is_invisible_until_it_is_available() -> None:
    store = store_with_days(DAY, DAY)
    bars = store.view(datetime(2019, 7, 3, tzinfo=UTC)).hourly(SYMBOL)
    second = bars[1]
    just_closed = second.t_close + timedelta(minutes=10)
    assert [b.t_open for b in store.view(just_closed).hourly(SYMBOL)] == [bars[0].t_open]
    at_its_decision = second.t_close + timedelta(minutes=16)
    assert [b.t_open for b in store.view(at_its_decision).hourly(SYMBOL)] == [
        bars[0].t_open,
        second.t_open,
    ]


@pytest.mark.spec("I1", "ARCH§6.2")
def test_a_revision_is_invisible_until_it_is_known() -> None:
    store = store_with_days(DAY, DAY)
    original = store.view(datetime(2019, 7, 3, tzinfo=UTC)).hourly(SYMBOL)[0]
    tau = original.t_close + timedelta(minutes=16)
    revised = dataclasses.replace(original, close=original.close + 0.25, high=original.high + 0.25)
    store.put_hourly(SYMBOL, [revised], known_at=tau + timedelta(minutes=10))

    assert store.view(tau).hourly(SYMBOL)[0].close == original.close
    later = store.view(tau + timedelta(minutes=11)).hourly(SYMBOL)[0]
    assert later.close == revised.close
    assert later.version == original.version + 1
    # Live, the view is read after ingest: as_of is later than tau and sees what was known then.
    live = store.view(tau, as_of=tau + timedelta(minutes=11)).hourly(SYMBOL)[0]
    assert live.close == revised.close


@pytest.mark.spec("I1", "ARCH§6.2")
def test_as_of_may_not_precede_tau() -> None:
    store = store_with_days(DAY, DAY)
    tau = datetime(2019, 7, 2, 15, 46, tzinfo=UTC)
    with pytest.raises(ClockError):
        store.view(tau, as_of=tau - timedelta(seconds=1))


@pytest.mark.spec("I1", "ARCH§6.2")
def test_the_view_returns_bars_in_order_and_the_last_n() -> None:
    store = store_with_days(date(2019, 7, 1), date(2019, 7, 3))
    view = store.view(datetime(2019, 7, 5, tzinfo=UTC))
    bars = view.hourly(SYMBOL)
    assert len(bars) == 7 + 7 + 4
    assert [b.t_open for b in bars] == sorted(b.t_open for b in bars)
    assert view.hourly(SYMBOL, last=3) == bars[-3:]
    since = bars[10].t_open
    assert view.hourly(SYMBOL, since=since) == bars[10:]


@pytest.mark.spec("I1", "SPEC§2", "ARCH§6.2")
def test_minutes_follow_the_same_contract() -> None:
    store = store_with_days(DAY, DAY)
    tau = datetime(2019, 7, 2, 14, 46, tzinfo=UTC)  # 10:46 ET
    minutes = store.view(tau).minutes(SYMBOL, tau - timedelta(hours=2), tau)
    assert len(minutes) > 0
    last_start = minutes.at(len(minutes) - 1)
    # a minute starting at t is available at t + 1 min + 15 min, strictly before tau
    assert last_start + timedelta(minutes=16) < tau
    assert last_start + timedelta(minutes=17) >= tau


_TIMES = st.integers(min_value=0, max_value=3 * 24 * 60).map(
    lambda m: datetime(2019, 7, 1, 13, 0, tzinfo=UTC) + timedelta(minutes=m)
)


@pytest.mark.spec("I1", "ARCH§6.2")
@given(tau1=_TIMES, gap=st.integers(min_value=1, max_value=3 * 24 * 60))
def test_an_earlier_view_is_contained_in_a_later_one(tau1: datetime, gap: int) -> None:
    store = _store_with_a_revision()
    tau2 = tau1 + timedelta(minutes=gap)
    early = {b.t_open: b for b in store.view(tau1).hourly(SYMBOL)}
    late = {b.t_open: b for b in store.view(tau2).hourly(SYMBOL)}
    assert set(early) <= set(late)
    for t_open, bar in early.items():
        if late[t_open].version == bar.version:
            assert late[t_open] == bar


def _store_with_a_revision() -> BarStore:
    store = store_with_days(date(2019, 7, 1), date(2019, 7, 3))
    bars = store.view(datetime(2019, 7, 5, tzinfo=UTC)).hourly(SYMBOL)
    target = bars[8]
    store.put_hourly(
        SYMBOL,
        [dataclasses.replace(target, volume=target.volume + 100)],
        known_at=target.available_at + timedelta(hours=5),
    )
    return store


@pytest.mark.spec("I1", "ARCH§6.2")
def test_the_cache_key_reflects_only_what_was_read() -> None:
    tau = datetime(2019, 7, 2, 17, 46, tzinfo=UTC)  # 13:46 ET: four bars visible

    def digest(store: BarStore) -> str:
        view = store.view(tau)
        view.hourly(SYMBOL)
        return view.digest()

    base = store_with_days(DAY, DAY)
    reference = digest(base)
    bars = base.view(datetime(2019, 7, 3, tzinfo=UTC)).hourly(SYMBOL)

    invisible = store_with_days(DAY, DAY)
    invisible.put_hourly(  # a bar that isn't available yet at tau
        SYMBOL,
        [dataclasses.replace(bars[6], close=bars[6].close + 1)],
        known_at=bars[6].available_at,
    )
    invisible.put_hourly(
        SYMBOL, [dataclasses.replace(bars[0], close=bars[0].close + 1)], known_at=tau + US
    )
    assert digest(invisible) == reference

    visible = store_with_days(DAY, DAY)
    visible.put_hourly(
        SYMBOL, [dataclasses.replace(bars[1], close=bars[1].close + 1)], known_at=tau - US
    )
    assert digest(visible) != reference
