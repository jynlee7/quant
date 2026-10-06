"""B3: features use no future rows, the features half of I1 (spec §3, ARCHITECTURE §6.2).

A bar's features depend only on the bars up to it, and a decision's features are read through
the point-in-time view, so a bar or a revision that came later can't reach them. T-NL1 (B5)
extends this to the whole decision record.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from functools import cache

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import quant.features.stream as stream_module
from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.epoch import to_us
from quant.core.errors import FeatureError
from quant.core.schedule import Decision, decision_schedule
from quant.data.frames import HourlyBar
from quant.data.market_db import HISTORY, BarStore
from quant.data.policy import Purpose
from quant.features.menu import FeatureConfig, r1_features
from quant.features.stream import (
    FeatureMatrix,
    FeatureRow,
    FeatureStream,
    compute_features,
    features_for,
)
from tests.support.bars import SYMBOL, revise_close, store_with_days, synthetic_days
from tests.support.features import bits, stream_rows, synthetic_bars, with_prices

FIRST, LAST = date(2016, 9, 1), date(2016, 12, 9)  # 70 sessions: the last day is warm
AFTER = datetime(2016, 12, 10, tzinfo=UTC)
CHANGES = ("prices", "volume", "drop", "nan", "huge", "negative", "shuffle", "revise", "append")
VALID = ("prices", "volume", "drop", "revise", "append")  # every bar stays one the stream takes


@cache
def _bars() -> tuple[HourlyBar, ...]:
    return tuple(synthetic_bars(FIRST, LAST))


@cache
def _base() -> dict[datetime, FeatureRow]:
    return stream_rows(_bars(), r1_features())


Rows = tuple[bytes, bytes, bytes]


def _up_to(matrix: FeatureMatrix, t_open: datetime) -> Rows:
    """The exact bits of a matrix's rows whose bars open at or before `t_open`."""
    keep = matrix.t_open <= to_us(t_open)
    return (
        matrix.t_open[keep].tobytes(),
        matrix.available_at[keep].tobytes(),
        matrix.values[keep].tobytes(),
    )


def _change_after(bars: tuple[HourlyBar, ...], cut: int, change: str, seed: int) -> list[HourlyBar]:
    """`bars` with every bar after index `cut` changed one way; bars up to `cut` untouched."""
    head, tail = list(bars[: cut + 1]), list(bars[cut + 1 :])
    rng = random.Random(seed)
    if change == "prices":
        tail = [with_prices(bar, 1.5) for bar in tail]
    elif change == "volume":
        tail = [replace(bar, volume=bar.volume * 7 + 1) for bar in tail]
    elif change == "drop":
        tail = []
    elif change == "nan":
        tail = [replace(bar, close=math.nan) for bar in tail]
    elif change == "huge":
        tail = [replace(bar, open=1e9, high=1e9, low=1e9, close=1e9) for bar in tail]
    elif change == "negative":
        tail = [replace(bar, close=-1e9, low=-1e9) for bar in tail]
    elif change == "shuffle":
        rng.shuffle(tail)
    elif change == "revise" and tail:
        i = rng.randrange(len(tail))
        close = tail[i].close + 0.25
        tail[i] = replace(tail[i], close=close, high=max(tail[i].high, close))
    elif change == "append":
        last = bars[-1]
        tail += [
            replace(
                last,
                t_open=last.t_open + timedelta(days=k),
                close=last.close * (1 + k),
                high=last.close * (1 + k),
            )
            for k in range(1, 4)
        ]
    return head + tail


@pytest.mark.spec("I1", "SPEC§3")
@given(
    position=st.integers(min_value=0, max_value=10**6),
    change=st.sampled_from(CHANGES),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_changing_bars_after_t_leaves_every_feature_at_t_unchanged(
    position: int, change: str, seed: int
) -> None:
    _check_unchanged_up_to(position % (len(_bars()) - 1), change, seed)


def _check_unchanged_up_to(cut: int, change: str, seed: int) -> None:
    """Assert that changing the bars after index `cut` leaves every row up to it unchanged,
    bar by bar through the stream and in one batch through `compute_features`. The batch sees
    the whole list at once, so it is where reading ahead could happen."""
    bars = _bars()
    base = _base()
    changed_bars = _change_after(bars, cut, change, seed)
    changed = stream_rows(changed_bars, r1_features())
    for bar in bars[: cut + 1]:
        assert bits(changed[bar.t_open]) == bits(base[bar.t_open]), (cut, change, bar.t_open)
    if change in VALID:  # a bad bar makes a batch raise, by design
        whole = compute_features(bars, r1_features())
        after = compute_features(changed_bars, r1_features())
        assert _up_to(after, bars[cut].t_open) == _up_to(whole, bars[cut].t_open), (cut, change)


def _peeking(bars: Sequence[HourlyBar], config: FeatureConfig) -> FeatureMatrix:
    """A planted leak: the real batch features shifted by the last bar's close."""
    real = stream_module.compute_features(bars, config)
    leaked = real.values + bars[-1].close
    return FeatureMatrix(real.columns, real.t_open, real.available_at, leaked)


@pytest.mark.spec("I1", "SPEC§3")
def test_the_lookahead_check_catches_a_planted_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    # The check must be able to fail: a batch path that reads a later bar is caught.
    monkeypatch.setattr(sys.modules[__name__], "compute_features", _peeking)
    with pytest.raises(AssertionError):
        _check_unchanged_up_to(len(_bars()) - 20, "prices", 0)


@pytest.mark.spec("I1", "SPEC§3", "ARCH§4.2")
def test_a_batch_over_the_first_k_bars_gives_the_rows_of_all_the_bars() -> None:
    bars = _bars()
    whole = compute_features(bars, r1_features())
    warm_cuts = 0
    for k in [1, 100, 420, *range(len(bars) - 70, len(bars) + 1, 3)]:
        prefix = compute_features(bars[:k], r1_features())
        assert len(prefix) == len(np.flatnonzero(whole.t_open <= to_us(bars[k - 1].t_open)))
        assert _up_to(prefix, bars[k - 1].t_open) == _up_to(whole, bars[k - 1].t_open), k
        warm_cuts += len(prefix) > 0
    assert warm_cuts > 20


@pytest.mark.spec("I1", "SPEC§3", "ARCH§6.2")
def test_a_decision_with_later_bars_in_view_reads_only_the_bars_up_to_it() -> None:
    store = store_with_days(FIRST, LAST)
    view = store.view(AFTER, as_of=AFTER)  # every bar of LAST is visible, later ones included
    bars = view.hourly(SYMBOL, purpose=Purpose.OPERATE)
    whole = compute_features(bars, r1_features())
    for decision in _decisions(LAST):
        vector = features_for(FeatureStream(r1_features()), view, SYMBOL, decision.bar)
        k = next(i for i, bar in enumerate(bars) if bar.t_open == decision.bar.t_open) + 1
        prefix = compute_features(bars[:k], r1_features())
        assert prefix.t_open[-1] == to_us(decision.bar.t_open)
        assert vector.tobytes() == prefix.values[-1].tobytes()
        row = np.flatnonzero(whole.t_open == to_us(decision.bar.t_open))
        assert vector.tobytes() == whole.values[row[0]].tobytes()


def _decisions(day: date) -> tuple[Decision, ...]:
    return decision_schedule(TradingCalendar.xnys(), day)


@pytest.mark.spec("I1", "SPEC§3", "ARCH§6.2")
def test_a_decision_reads_its_features_through_the_view() -> None:
    store = store_with_days(FIRST, LAST)
    seen = store.view(AFTER, as_of=AFTER).hourly(SYMBOL, purpose=Purpose.OPERATE)
    expected = stream_rows(seen, r1_features())
    stream = FeatureStream(r1_features())
    decisions = _decisions(LAST)
    for decision in decisions:
        view = store.view(decision.tau, as_of=decision.tau)
        vector = features_for(stream, view, SYMBOL, decision.bar)
        assert vector.tobytes() == expected[decision.bar.t_open].vector().tobytes()
        assert stream.last_t_open == decision.bar.t_open
    last = decisions[-1]
    with pytest.raises(FeatureError):  # this bar is consumed: its row is in the past
        features_for(stream, store.view(last.tau, as_of=last.tau), SYMBOL, last.bar)


@pytest.mark.spec("I1", "ARCH§6.2")
def test_a_revision_known_after_as_of_leaves_the_features_unchanged() -> None:
    clock = SimClock(datetime(2016, 12, 9, tzinfo=UTC))
    store = store_with_days(FIRST, LAST, clock=clock)
    decision = _decisions(LAST)[2]
    tau = decision.tau
    before = features_for(
        FeatureStream(r1_features()), store.view(tau, as_of=tau), SYMBOL, decision.bar
    )

    bar = next(
        b
        for b in store.view(AFTER, as_of=AFTER).hourly(SYMBOL, purpose=Purpose.OPERATE)
        if b.t_open == decision.bar.t_open
    )
    known = tau + timedelta(minutes=10)
    revise_close(store, clock, synthetic_days(FIRST, LAST), bar, 0.25, at=known)

    at_tau = store.view(tau, as_of=tau)
    assert features_for(FeatureStream(r1_features()), at_tau, SYMBOL, decision.bar).tobytes() == (
        before.tobytes()
    )
    later = known + timedelta(minutes=1)
    revised = store.view(tau, as_of=later)  # a live read after the revision landed sees it
    after = features_for(FeatureStream(r1_features()), revised, SYMBOL, decision.bar)
    assert after[0] != before[0]


@pytest.mark.spec("I1", "SPEC§3", "ARCH§6.2")
def test_a_missing_or_incomplete_decision_bar_is_a_model_error() -> None:
    store = store_with_days(FIRST, LAST)
    decisions = _decisions(LAST)

    # Not yet available: at its own available_at the bar is still invisible.
    first = decisions[0]
    available = first.tau - timedelta(minutes=1)
    with pytest.raises(FeatureError):
        features_for(
            FeatureStream(r1_features()),
            store.view(available, as_of=available),
            SYMBOL,
            first.bar,
        )

    # Missing: the next session has no bars at all. The bars before it stay consumed.
    monday = _decisions(date(2016, 12, 12))[0]
    stream = FeatureStream(r1_features())
    with pytest.raises(FeatureError):
        features_for(stream, store.view(monday.tau, as_of=monday.tau), SYMBOL, monday.bar)
    assert stream.last_t_open == decisions[-1].bar.t_open

    # Incomplete: the 10:30 bar lacks its last minute. Its decision fails; the next one reads
    # the features of a stream that skipped it.
    minutes = synthetic_days(FIRST, LAST)
    last_minute = to_us(decisions[1].bar.t_close) - 60_000_000
    partial = BarStore.memory()
    partial.ingest_minutes(
        SYMBOL, minutes.select(minutes.t != last_minute), known_at=HISTORY, source="synthetic"
    )
    stream = FeatureStream(r1_features())
    gap = decisions[1]
    with pytest.raises(FeatureError):
        features_for(stream, partial.view(gap.tau, as_of=gap.tau), SYMBOL, gap.bar)
    assert stream.last_t_open == decisions[0].bar.t_open
    following = decisions[2]
    vector = features_for(
        stream, partial.view(following.tau, as_of=following.tau), SYMBOL, following.bar
    )
    stored = partial.view(AFTER, as_of=AFTER).hourly(SYMBOL, purpose=Purpose.OPERATE)
    assert [b.complete for b in stored if b.t_open == gap.bar.t_open] == [False]
    expected = stream_rows(stored, r1_features())[following.bar.t_open].vector()
    assert vector.tobytes() == expected.tobytes()
    assert np.isfinite(vector).all()
