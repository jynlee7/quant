"""B8a: the primitives, their ATR stops, take-profit, re-arm and invalidation (spec §7, PLAN §5,
D9).

The signals are checked at their edges (strict and inclusive comparisons exactly as written).
The lifecycle runs r1's own CALM_UP and CHOP playbooks over hand-built bars: the stop is fixed at
entry from the entry bar's ATR and never trails; a stop or take-profit leaves the primitive
disarmed until it sees its own exit condition (design call 7); each invalidation kind fires
exactly past its threshold.
"""

from __future__ import annotations

from typing import Any

import pytest

from quant.core.canonical import canonical_dumps
from quant.core.regime import Label
from tests.support.decide import REPO, priced

RED = pytest.mark.xfail(strict=True, reason="red:B8", raises=NotImplementedError)


def values(
    i: int, *, ema: dict[int, float] | None = None, atr: float | None = 2.0, z: float | None = None
) -> Any:
    """IndicatorValues for template bar `i`."""
    from quant.decide.indicators import IndicatorValues

    t_open = priced(i, 100.0).t_open
    return IndicatorValues(t_open=t_open, ema=ema or {}, atr=atr, mean=None, sd=None, z=z)


def trend(i: int, fast: float, slow: float, atr: float | None = 2.0) -> Any:
    return values(i, ema={35: fast, 175: slow}, atr=atr)


def playbook(label: Label, **changes: Any) -> Any:
    """r1's playbook for `label`, with top-level `changes`."""
    from quant.decide.playbook import PLAYBOOK, load_release

    original = load_release(REPO).playbook(label)
    if not changes:
        return original
    data = original.model_dump(mode="json")
    data.update(changes)
    return PLAYBOOK.validate_json(canonical_dumps(data))


def flat_state() -> Any:
    from quant.decide.primitives import TradeState

    return TradeState.flat()


def in_trade(
    entry: float,
    stop: float,
    *,
    bars: int = 0,
    peak: float | None = None,
    take_profit: float | None = None,
) -> Any:
    from quant.decide.primitives import TradeState

    return TradeState(
        in_position=True,
        armed=True,
        entry_price=entry,
        stop=stop,
        take_profit=take_profit,
        bars_in_trade=bars,
        peak_close=entry if peak is None else peak,
    )


# --- signals --------------------------------------------------------------------------------


@RED
@pytest.mark.spec("SPEC§7", "PLAN§5", "D9")
@pytest.mark.parametrize(
    ("fast", "slow", "close", "entry", "exit_"),
    [
        (101.0, 100.0, 101.5, True, False),  # fast > slow, close > fast
        (101.0, 100.0, 101.0, False, False),  # close == fast: not above it
        (101.0, 100.0, 100.5, False, False),  # close below fast: no entry, no exit
        (100.0, 100.0, 105.0, False, False),  # fast == slow: neither
        (99.0, 100.0, 105.0, False, True),  # fast < slow: exit
    ],
)
def test_trend_following_signal(
    fast: float, slow: float, close: float, entry: bool, exit_: bool
) -> None:
    from quant.decide.playbook import TrendParams
    from quant.decide.primitives import trend_following

    params = TrendParams(fast=35, slow=175, atr_bars=35)
    got = trend_following(params, trend(0, fast, slow), close)
    assert (got.entry, got.exit, got.defined) == (entry, exit_, True)


@RED
@pytest.mark.spec("SPEC§7", "PLAN§5", "D9")
@pytest.mark.parametrize(
    ("z", "entry", "exit_"),
    [
        (-1.5, True, False),  # z <= -1.5 enters (inclusive)
        (-2.7, True, False),
        (-1.4999999999999998, False, False),
        (-0.5, False, False),
        (0.0, False, True),  # z >= -0 exits (inclusive)
        (-1e-12, False, False),
        (1.3, False, True),
    ],
)
def test_mean_reversion_signal(z: float, entry: bool, exit_: bool) -> None:
    from quant.decide.playbook import MeanReversionParams
    from quant.decide.primitives import mean_reversion

    params = MeanReversionParams(n=35, entry_z=1.5, exit_z=0.0, atr_bars=35)
    got = mean_reversion(params, values(0, z=z))
    assert (got.entry, got.exit, got.defined) == (entry, exit_, True)


@RED
@pytest.mark.spec("SPEC§7")
def test_mean_reversion_without_z_has_no_signal() -> None:
    from quant.decide.playbook import MeanReversionParams
    from quant.decide.primitives import mean_reversion

    params = MeanReversionParams(n=35, entry_z=1.5, exit_z=0.0, atr_bars=35)
    got = mean_reversion(params, values(0, z=None))
    assert (got.entry, got.exit, got.defined) == (False, False, False)


@RED
@pytest.mark.spec("SPEC§7", "D8")
def test_flat_never_enters() -> None:
    from quant.decide.primitives import flat

    got = flat()
    assert (got.entry, got.exit, got.defined) == (False, False, True)


# --- the lifecycle --------------------------------------------------------------------------


@RED
@pytest.mark.spec("SPEC§7", "PLAN§5")
def test_an_entry_fixes_the_stop_from_the_entry_bars_atr() -> None:
    from quant.decide.primitives import Reason, step

    calm = playbook(Label.CALM_UP)
    got = step(calm, flat_state(), trend(0, 101.0, 100.0, atr=1.25), priced(0, 102.0))
    assert got.reason == Reason.ENTER
    assert got.in_position
    s = got.state
    assert (s.entry_price, s.stop, s.take_profit) == (102.0, 102.0 - 4.0 * 1.25, None)
    assert (s.bars_in_trade, s.peak_close) == (0, 102.0)

    chop = playbook(Label.CHOP)
    got = step(chop, flat_state(), values(0, z=-1.6, atr=2.0), priced(0, 98.0))
    assert got.reason == Reason.ENTER
    assert got.state.stop == 98.0 - 3.0 * 2.0


@RED
@pytest.mark.spec("SPEC§7", "PLAN§5")
def test_the_stop_never_trails() -> None:
    from quant.decide.primitives import Reason, step

    calm = playbook(Label.CALM_UP)
    state = step(calm, flat_state(), trend(0, 101.0, 100.0, atr=1.0), priced(0, 102.0)).state
    for i, close in enumerate([104.0, 107.0, 111.0, 109.0], start=1):
        got = step(calm, state, trend(i, 105.0, 100.0, atr=3.0), priced(i, close, low=close - 1))
        assert got.reason == Reason.HOLD
        state = got.state
        assert state.stop == 98.0  # 102 - 4 x 1.0, whatever later ATRs and prices do
        assert state.bars_in_trade == i
    assert state.peak_close == 111.0


@RED
@pytest.mark.spec("SPEC§7", "PLAN§5")
@pytest.mark.parametrize(
    ("low", "stopped"), [(98.0, True), (97.0, True), (98.00000000000001, False)]
)
def test_the_stop_is_hit_when_the_low_reaches_it(low: float, stopped: bool) -> None:
    from quant.decide.primitives import Reason, step

    calm = playbook(Label.CALM_UP)
    got = step(calm, in_trade(102.0, 98.0), trend(1, 105.0, 100.0), priced(1, 106.0, low=low))
    assert got.reason == (Reason.STOP if stopped else Reason.HOLD)
    assert got.in_position is not stopped


@RED
@pytest.mark.spec("SPEC§7", "PLAN§5")
def test_take_profit_exits_on_the_close() -> None:
    from quant.decide.primitives import Reason, step

    calm = playbook(Label.CALM_UP, take_profit=2.0)
    entered = step(calm, flat_state(), trend(0, 101.0, 100.0, atr=1.5), priced(0, 102.0)).state
    assert entered.take_profit == 102.0 + 2.0 * 1.5
    # The high touching it doesn't count; the decision's close does.
    held = step(calm, entered, trend(1, 103.0, 100.0), priced(1, 104.0, high=110.0))
    assert held.reason == Reason.HOLD
    hit = step(calm, held.state, trend(2, 103.0, 100.0), priced(2, 105.0))
    assert hit.reason == Reason.TAKE_PROFIT
    assert not hit.in_position
    assert not hit.state.armed


@RED
@pytest.mark.spec("SPEC§7")
def test_the_stop_comes_before_take_profit_and_the_exit_signal() -> None:
    from quant.decide.primitives import Reason, step

    calm = playbook(Label.CALM_UP, take_profit=2.0)
    state = in_trade(102.0, 98.0, take_profit=104.0)
    # Low through the stop, close above the take-profit, fast below slow: STOP.
    got = step(calm, state, trend(1, 99.0, 100.0), priced(1, 105.0, low=97.0))
    assert got.reason == Reason.STOP


@RED
@pytest.mark.spec("SPEC§7", "PLAN§5")
def test_a_stopped_primitive_rearms_only_on_its_exit_condition() -> None:
    # Design call 7: a stop-out doesn't re-enter on the next bar while the entry still holds.
    from quant.decide.primitives import Reason, step

    calm = playbook(Label.CALM_UP)
    stopped = step(calm, in_trade(102.0, 98.0), trend(1, 103.0, 100.0), priced(1, 104.0, low=97.5))
    assert stopped.reason == Reason.STOP
    assert not stopped.state.armed
    still = step(calm, stopped.state, trend(2, 103.0, 100.0), priced(2, 104.0))
    assert still.reason == Reason.DISARMED  # the entry condition holds, but no entry
    assert not still.in_position
    crossed = step(calm, still.state, trend(3, 99.0, 100.0), priced(3, 98.5))
    assert crossed.reason == Reason.REARM
    assert crossed.state.armed
    assert not crossed.in_position
    again = step(calm, crossed.state, trend(4, 101.0, 100.0), priced(4, 102.0))
    assert again.reason == Reason.ENTER


@RED
@pytest.mark.spec("SPEC§7")
def test_a_stop_bar_that_shows_the_exit_condition_rearms_on_it() -> None:
    from quant.decide.primitives import Reason, step

    chop = playbook(Label.CHOP)
    state = in_trade(98.0, 92.0)
    got = step(chop, state, values(1, z=0.2), priced(1, 99.0, low=91.0))  # z >= 0 too
    assert got.reason == Reason.STOP
    assert got.state.armed
    again = step(chop, got.state, values(2, z=-1.7), priced(2, 96.0))
    assert again.reason == Reason.ENTER


@RED
@pytest.mark.spec("SPEC§7")
def test_a_signal_exit_stays_armed() -> None:
    from quant.decide.primitives import Reason, step

    chop = playbook(Label.CHOP)
    out = step(chop, in_trade(98.0, 92.0), values(1, z=0.0), priced(1, 100.0))
    assert out.reason == Reason.EXIT
    assert out.state.armed
    assert not out.in_position
    assert (out.state.entry_price, out.state.stop, out.state.peak_close) == (None, None, None)
    assert out.state.bars_in_trade == 0
    back = step(chop, out.state, values(2, z=-1.5), priced(2, 97.0))
    assert back.reason == Reason.ENTER


@RED
@pytest.mark.spec("SPEC§7", "SPEC§8")
def test_blocked_entries_wait_and_a_blocked_trade_is_not_forced_out() -> None:
    from quant.decide.primitives import Reason, step

    calm = playbook(Label.CALM_UP)
    got = step(calm, flat_state(), trend(0, 101.0, 100.0), priced(0, 102.0), entries_allowed=False)
    assert got.reason == Reason.BLOCKED
    assert not got.in_position
    held = step(
        calm, in_trade(102.0, 98.0), trend(1, 103.0, 100.0), priced(1, 104.0), entries_allowed=False
    )
    assert held.reason == Reason.HOLD
    assert held.in_position


@RED
@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize("atr", [None, 0.0])
def test_no_entry_without_a_positive_atr(atr: float | None) -> None:
    from quant.decide.primitives import Reason, step

    calm = playbook(Label.CALM_UP)
    got = step(calm, flat_state(), trend(0, 101.0, 100.0, atr=atr), priced(0, 102.0))
    assert got.reason == Reason.UNDEFINED
    assert not got.in_position


@RED
@pytest.mark.spec("SPEC§7", "D8", "D9")
@pytest.mark.parametrize("label", [Label.STRESS, Label.CRASH])
def test_a_flat_playbook_never_holds(label: Label) -> None:
    from quant.decide.primitives import Reason, TradeState, step

    flat = playbook(label)
    got = step(flat, flat_state(), values(0, z=-3.0), priced(0, 90.0))
    assert got.reason == Reason.FLAT
    assert got.state == TradeState.flat()


@RED
@pytest.mark.spec("SPEC§7")
def test_the_cold_trade_state_is_flat_and_armed() -> None:
    from quant.decide.primitives import TradeState, close_trade

    cold = TradeState.flat()
    assert (cold.in_position, cold.armed, cold.bars_in_trade) == (False, True, 0)
    assert (cold.entry_price, cold.stop, cold.take_profit, cold.peak_close) == (None,) * 4
    assert close_trade(in_trade(102.0, 98.0, bars=5, peak=110.0)) == cold


@RED
@pytest.mark.spec("SPEC§7")
def test_step_refuses_values_of_another_bar() -> None:
    from quant.decide.primitives import step

    with pytest.raises(ValueError, match="t_open"):
        step(playbook(Label.CALM_UP), flat_state(), trend(1, 101.0, 100.0), priced(0, 102.0))


@RED
@pytest.mark.spec("SPEC§7")
def test_stop_and_take_profit_levels() -> None:
    from quant.decide.primitives import stop_level, take_profit_level

    assert stop_level(100.0, 2.5, 4.0) == 90.0
    assert take_profit_level(100.0, 2.5, 4.0) == 110.0


# --- invalidation ---------------------------------------------------------------------------


def _check(pb: Any, state: Any, i: int, close: float, label_prob: float, **ind: Any) -> Any:
    from quant.decide.primitives import check_invalidation

    return check_invalidation(pb, state, values(i, **ind), priced(i, close), label_prob=label_prob)


@RED
@pytest.mark.spec("SPEC§7", "PLAN§5")
@pytest.mark.parametrize(
    ("label_prob", "hit"), [(0.29999999999999993, True), (0.30, False), (0.9, False), (0.0, True)]
)
def test_label_prob_below_fires_strictly_below_its_threshold(label_prob: float, hit: bool) -> None:
    calm = playbook(Label.CALM_UP)
    got = _check(calm, in_trade(102.0, 98.0, bars=3), 4, 103.0, label_prob)
    assert (got is not None) == hit
    if hit:
        assert got.condition.kind == "label_prob_below"
        assert got.reason


@RED
@pytest.mark.spec("SPEC§7", "PLAN§5")
@pytest.mark.parametrize(("bars", "hit"), [(21, False), (22, True)])
def test_bars_in_trade_exceeds_fires_past_its_count(bars: int, hit: bool) -> None:
    chop = playbook(Label.CHOP)
    got = _check(chop, in_trade(98.0, 92.0, bars=bars), 5, 98.5, 0.8, z=-0.4)
    assert (got is not None) == hit
    if hit:
        assert got.condition.kind == "bars_in_trade_exceeds"


@RED
@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize(("close", "hit"), [(104.0, False), (103.99999999999999, True)])
def test_close_below_ema_fires_strictly_below_the_ema(close: float, hit: bool) -> None:
    calm = playbook(Label.CALM_UP, invalidation=[{"kind": "close_below_ema", "span": 50}])
    got = _check(calm, in_trade(102.0, 98.0), 3, close, 0.9, ema={35: 105.0, 50: 104.0, 175: 100.0})
    assert (got is not None) == hit


@RED
@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize(("close", "hit"), [(90.0, False), (89.99, True), (100.0, False)])
def test_trade_drawdown_exceeds_fires_past_its_fraction(close: float, hit: bool) -> None:
    calm = playbook(
        Label.CALM_UP, invalidation=[{"kind": "trade_drawdown_exceeds", "fraction": 0.1}]
    )
    state = in_trade(95.0, 80.0, bars=4, peak=100.0)
    got = _check(calm, state, 5, close, 0.9, ema={35: 101.0, 175: 100.0})
    assert (got is not None) == hit


@RED
@pytest.mark.spec("SPEC§7")
def test_the_first_listed_condition_is_the_hit_and_flat_has_none() -> None:
    chop = playbook(Label.CHOP)  # bars_in_trade_exceeds 21, then label_prob_below 0.30
    got = _check(chop, in_trade(98.0, 92.0, bars=30), 6, 97.0, 0.1, z=-0.5)
    assert got is not None
    assert got.condition.kind == "bars_in_trade_exceeds"
    assert _check(chop, flat_state(), 6, 97.0, 0.1, z=-0.5) is None
