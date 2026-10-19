"""B8b: the edge behind Kelly sizing (spec §9, C12; quant.decide.edge).

Welford's accumulator against a plain two-pass reference (`statistics`), f* = max(0, mean -
s/sqrt(n)) / s^2 and its zeros (no edge, fewer than 100 bars, s = 0), seeding from a sequence,
the shadow-return definition (a stop exits at min(open, stop), shadow_cost_bp per change of
shadow position, both charged to held bars), and counting only under the label active at t - 1
(that last through decide). Every return here is synthetic: B8 never computes P&L on DEV.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

import pytest

from quant.core.regime import Label
from quant.decide.indicators import IndicatorValues
from quant.decide.playbook import Playbook
from quant.decide.primitives import Reason, TradeState
from tests.support.decide import priced, r1_playbook

MIN_BARS = 100
COST_BP = 2.0
COST = COST_BP * 1e-4


def _close(a: float, b: float, rel: float = 1e-12) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


# --- Welford and f* --------------------------------------------------------------------------


@pytest.mark.spec("C12", "SPEC§9")
def test_welford_matches_a_plain_two_pass_reference() -> None:
    from quant.decide.edge import EdgeStats

    xs = [0.0123 * math.sin(i * 0.7) + 0.0004 * (i % 5) for i in range(257)]
    stats = EdgeStats.empty()
    for x in xs:
        stats = stats.add(x)
    assert stats.n == len(xs)
    assert _close(stats.mean, statistics.fmean(xs))
    assert stats.variance is not None
    assert stats.sd is not None
    assert _close(stats.variance, statistics.variance(xs))
    assert _close(stats.sd, statistics.stdev(xs))


@pytest.mark.spec("C12")
def test_the_accumulator_is_pure_and_empty_has_no_variance() -> None:
    from quant.decide.edge import EdgeStats

    empty = EdgeStats.empty()
    assert (empty.n, empty.mean, empty.m2) == (0, 0.0, 0.0)
    assert empty.variance is None
    assert empty.sd is None
    one = empty.add(0.01)
    assert (empty.n, one.n) == (0, 1)
    assert one.variance is None
    with pytest.raises(ValueError, match="finite"):
        empty.add(math.nan)


@pytest.mark.spec("C12", "SPEC§9")
def test_kelly_is_the_lower_bound_over_the_variance() -> None:
    from quant.decide.edge import EdgeStats, kelly

    xs = [0.003, -0.001] * 60  # 120 bars: mean 0.001
    stats = EdgeStats.empty()
    for x in xs:
        stats = stats.add(x)
    s = statistics.stdev(xs)
    expected = (statistics.fmean(xs) - s / math.sqrt(len(xs))) / s**2
    assert _close(stats.f_star(min_bars=MIN_BARS), expected, 1e-9)
    assert _close(kelly(len(xs), statistics.fmean(xs), s, min_bars=MIN_BARS), expected, 1e-12)
    assert expected > 0


@pytest.mark.spec("C12", "SPEC§9")
def test_kelly_is_zero_without_an_edge() -> None:
    from quant.decide.edge import EdgeStats, kelly

    # A losing book, and a winning one whose mean lies within one standard error of 0.
    for xs in ([-0.003, 0.001] * 100, [0.0021, -0.002] * 100):
        stats = EdgeStats.empty()
        for x in xs:
            stats = stats.add(x)
        assert stats.mean - (stats.sd or 0.0) / math.sqrt(stats.n) <= 0
        assert stats.f_star(min_bars=MIN_BARS) == 0.0
    assert kelly(200, 0.0, 0.01, min_bars=MIN_BARS) == 0.0
    assert kelly(200, -0.5, 0.01, min_bars=MIN_BARS) == 0.0


@pytest.mark.spec("C12", "SPEC§9")
def test_kelly_is_zero_with_fewer_than_a_hundred_bars() -> None:
    from quant.decide.edge import EdgeStats

    stats = EdgeStats.empty()
    for x in [0.003, -0.001] * 49 + [0.003]:  # 99 bars with a clear edge
        stats = stats.add(x)
    assert stats.n == MIN_BARS - 1
    assert stats.f_star(min_bars=MIN_BARS) == 0.0
    assert stats.add(-0.001).f_star(min_bars=MIN_BARS) > 0  # the 100th bar counts


@pytest.mark.spec("C12", "SPEC§9")
def test_kelly_is_zero_when_every_return_is_equal() -> None:
    from quant.decide.edge import EdgeStats, kelly

    stats = EdgeStats.empty()
    for _ in range(150):
        stats = stats.add(0.001)
    assert stats.sd == 0.0
    assert stats.f_star(min_bars=MIN_BARS) == 0.0
    assert kelly(150, 0.001, 0.0, min_bars=MIN_BARS) == 0.0
    assert kelly(150, 0.001, None, min_bars=MIN_BARS) == 0.0


@pytest.mark.spec("C12", "SPEC§9")
def test_a_book_seeded_from_a_sequence_equals_adding_it_in_order() -> None:
    from quant.decide.edge import EdgeBook

    calm = [0.002 * ((i * 7) % 11 - 4) for i in range(130)]
    chop = [0.001, -0.0005, 0.0015]
    book = EdgeBook.seeded({Label.CALM_UP: calm, Label.CHOP: chop})
    added = EdgeBook.empty()
    for x in calm:
        added = added.add(Label.CALM_UP, x)
    for x in chop:
        added = added.add(Label.CHOP, x)
    assert book == added
    assert set(book.stats) == set(Label)
    assert book.stats[Label.STRESS].n == book.stats[Label.CRASH].n == 0
    assert book.stats[Label.CALM_UP].n == 130
    assert _close(book.stats[Label.CALM_UP].mean, statistics.fmean(calm))
    assert book.f_star(Label.CHOP, min_bars=MIN_BARS) == 0.0  # 3 bars
    with pytest.raises(ValueError, match="finite"):
        EdgeBook.seeded({Label.CHOP: [math.inf]})


# --- the shadow return -----------------------------------------------------------------------


@pytest.mark.spec("C12", "SPEC§9")
def test_a_held_bar_earns_the_close_over_the_previous_close() -> None:
    from quant.decide.edge import shadow_return

    bar = priced(10, 101.0, low=100.5, high=101.5)
    r = shadow_return(100.0, bar, held=True, stop=95.0, changes=0, cost_bp=COST_BP)
    assert r == 101.0 / 100.0 - 1
    assert shadow_return(100.0, bar, held=False, stop=None, changes=0, cost_bp=COST_BP) == 0.0


@pytest.mark.spec("C12", "SPEC§9")
def test_a_stop_inside_the_bar_exits_at_the_stop_and_a_gap_at_the_open() -> None:
    from quant.decide.edge import shadow_return

    # The low reaches the stop inside the bar: out at the stop, not the close.
    inside = replace(priced(10, 99.0, low=94.0, high=99.5), open=98.0)
    assert shadow_return(100.0, inside, held=True, stop=95.0, changes=0, cost_bp=0.0) == (
        95.0 / 100.0 - 1
    )
    # A gap through the stop: out at the open, below the stop.
    gap = replace(priced(10, 92.0, low=91.0, high=93.5), open=93.0)
    assert shadow_return(100.0, gap, held=True, stop=95.0, changes=0, cost_bp=0.0) == (
        93.0 / 100.0 - 1
    )
    # The low exactly at the stop is a hit.
    touch = replace(priced(10, 97.0, low=95.0, high=97.5), open=96.0)
    assert shadow_return(100.0, touch, held=True, stop=95.0, changes=0, cost_bp=0.0) == (
        95.0 / 100.0 - 1
    )


@pytest.mark.spec("C12", "SPEC§9")
def test_each_change_of_shadow_position_costs_shadow_cost_bp() -> None:
    from quant.decide.edge import shadow_return

    bar = priced(10, 101.0, low=100.5, high=101.5)
    gross = 101.0 / 100.0 - 1
    for changes in (0, 1, 2):
        r = shadow_return(100.0, bar, held=True, stop=95.0, changes=changes, cost_bp=COST_BP)
        assert _close(r, gross - changes * COST, 1e-15)
    with pytest.raises(ValueError, match="changes"):
        shadow_return(100.0, bar, held=True, stop=95.0, changes=-1, cost_bp=COST_BP)
    with pytest.raises(ValueError, match="prev_close"):
        shadow_return(0.0, bar, held=True, stop=95.0, changes=0, cost_bp=COST_BP)


def _trend_values(t_open: object, *, fast: float, slow: float, atr: float = 1.0) -> IndicatorValues:
    return IndicatorValues(t_open, {35: fast, 175: slow}, atr, None, None, None)  # type: ignore[arg-type]


def _run_shadow(playbook: Playbook, steps: list[tuple[float, float, float, float]]) -> list[object]:
    """The shadow over consecutive template bars, each (close, low, fast EMA, slow EMA)."""
    from quant.decide.edge import ShadowState, shadow_step

    state = ShadowState.cold()
    out = []
    for i, (close, low, fast, slow) in enumerate(steps):
        bar = priced(i, close, low=low, high=close + 0.5)
        step = shadow_step(
            playbook, state, _trend_values(bar.t_open, fast=fast, slow=slow), bar, cost_bp=COST_BP
        )
        out.append(step)
        state = step.state
    return out


@pytest.mark.spec("C12", "SPEC§9", "SPEC§7")
def test_a_shadow_round_trip_charges_entry_and_exit_on_held_bars() -> None:
    calm = r1_playbook(Label.CALM_UP)  # enter: fast > slow and close > fast; exit: fast < slow
    steps = _run_shadow(
        calm,
        [
            (100.0, 99.5, 99.0, 100.0),  # 0: first bar: no return, fast < slow, flat
            (101.0, 100.5, 100.5, 100.0),  # 1: entry signal: enters at 101, not held yet
            (102.0, 101.5, 101.0, 100.2),  # 2: first held bar: gross 102/101 - 1, entry cost
            (103.0, 102.5, 101.5, 100.4),  # 3: held, no cost
            (102.5, 102.0, 100.0, 100.6),  # 4: exit signal (fast < slow): gross, exit cost
            (102.0, 101.5, 99.0, 100.8),  # 5: flat: 0
        ],
    )
    held = [s.held for s in steps]  # type: ignore[attr-defined]
    rets = [s.ret for s in steps]  # type: ignore[attr-defined]
    reasons = [s.reason for s in steps]  # type: ignore[attr-defined]
    assert held == [False, False, True, True, True, False]
    assert reasons[1] is Reason.ENTER
    assert reasons[4] is Reason.EXIT
    assert rets[0] is None
    assert rets[1] == 0.0
    assert _close(rets[2], 102.0 / 101.0 - 1 - COST, 1e-15)  # type: ignore[arg-type]
    assert _close(rets[3], 103.0 / 102.0 - 1, 1e-15)  # type: ignore[arg-type]
    assert _close(rets[4], 102.5 / 103.0 - 1 - COST, 1e-15)  # type: ignore[arg-type]
    assert rets[5] == 0.0


@pytest.mark.spec("C12", "SPEC§9", "SPEC§7")
def test_a_shadow_stopped_out_exits_at_the_stop_and_pays_its_exit() -> None:
    from quant.decide.edge import ShadowState, shadow_step

    calm = r1_playbook(Label.CALM_UP)  # stop 4 x ATR(35); ATR here is 1.0, so the stop is 97
    state = ShadowState(
        trade=TradeState(
            in_position=True,
            armed=True,
            entry_price=101.0,
            stop=97.0,
            take_profit=None,
            bars_in_trade=3,
            peak_close=101.0,
        ),
        prev_close=100.0,
        entered=False,
    )
    bar = replace(priced(20, 98.0, low=96.0, high=99.0), open=99.0)
    step = shadow_step(
        calm, state, _trend_values(bar.t_open, fast=100.5, slow=100.0), bar, cost_bp=COST_BP
    )
    assert step.held
    assert step.reason is Reason.STOP
    assert step.ret is not None
    assert _close(step.ret, 97.0 / 100.0 - 1 - COST, 1e-15)
    assert not step.state.trade.in_position
    assert step.state.prev_close == 98.0
    assert not step.state.entered


@pytest.mark.spec("C12", "SPEC§9")
def test_a_shadow_ignores_invalidation_and_blocks() -> None:
    from quant.decide.edge import ShadowState, shadow_step

    # CHOP's invalidation exits after 21 bars in a trade; a shadow keeps holding.
    chop = r1_playbook(Label.CHOP)
    state = ShadowState(
        trade=TradeState(
            in_position=True,
            armed=True,
            entry_price=100.0,
            stop=90.0,
            take_profit=None,
            bars_in_trade=30,
            peak_close=100.0,
        ),
        prev_close=100.0,
        entered=False,
    )
    bar = priced(30, 100.5, low=100.0, high=101.0)
    values = IndicatorValues(bar.t_open, {}, 1.0, 101.0, 1.0, -0.5)
    step = shadow_step(chop, state, values, bar, cost_bp=COST_BP)
    assert step.held
    assert step.reason is Reason.HOLD
    assert step.state.trade.in_position
    assert step.state.trade.bars_in_trade == 31
    assert step.ret is not None
    assert _close(step.ret, 100.5 / 100.0 - 1, 1e-15)


@pytest.mark.spec("C12")
def test_a_flat_playbook_shadow_never_holds() -> None:
    from quant.decide.edge import ShadowState, shadow_step

    stress = r1_playbook(Label.STRESS)
    state = ShadowState.cold()
    for i, close in enumerate((100.0, 101.0, 99.0)):
        bar = priced(i, close)
        step = shadow_step(
            stress, state, IndicatorValues(bar.t_open, {}, None, None, None, None), bar, cost_bp=2.0
        )
        assert not step.held
        assert step.ret in (None, 0.0)
        state = step.state
    assert not state.trade.in_position


# --- counting under the label active at t - 1 (through decide) --------------------------------


@pytest.mark.spec("C12", "SPEC§9", "ARCH§4.3")
def test_decide_counts_a_shadow_return_only_under_the_label_active_before_the_bar() -> None:
    from quant.decide.edge import ShadowState
    from tests.support.decide import DecideFixture

    fx = DecideFixture.get()
    held = TradeState(
        in_position=True,
        armed=True,
        entry_price=fx.price_before * 0.99,
        stop=fx.price_before * 0.5,
        take_profit=None,
        bars_in_trade=2,
        peak_close=fx.price_before,
    )
    shadow = ShadowState(trade=held, prev_close=fx.price_before, entered=False)
    # Held across the bar; the stop is far below; an exit signal on the bar charges its cost.
    values = fx.indicators(Label.CALM_UP)
    exits = values.ema[35] < values.ema[175]
    expected = fx.bar.close / fx.price_before - 1 - (COST if exits else 0.0)
    for active, counts in ((Label.CALM_UP, True), (Label.CHOP, False), (None, False)):
        state = fx.warm_state(active=active, shadows={Label.CALM_UP: shadow})
        before = state.edge.stats[Label.CALM_UP]
        out = fx.decide(state)
        after = out.state.edge.stats[Label.CALM_UP]
        if counts:
            assert after.n == before.n + 1, active
            assert _close(after.mean * after.n - before.mean * before.n, expected, 1e-9)
        else:
            assert after == before, active
        # Another label's shadow, flat across the bar, never counts.
        assert out.state.edge.stats[Label.CHOP] == state.edge.stats[Label.CHOP]
