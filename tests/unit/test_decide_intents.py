"""B8b: intents (ARCHITECTURE §4.2 step 10, A3, D9, spec §10, design call 11;
quant.decide.intents).

At most 0.5x of equity per order, BUY and SELL alike; whole shares, and no intent for 0 shares;
the overnight cap on the last fillable decision, 15:46 ET and 12:46 on a half day (2018 FIT
dates, as tests.support.risk uses them); the closing-bar decision only reduces, as a reduce-only
SELL that executes at the next open; a BUY carries the trade's stop below the price; and the
intent is `core.orders.OrderIntent` exactly as B9's RiskEngine reads it, approved on a happy
path. The desk trades $100,000 at $100 a share: 1,000 shares are 1.0x.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import pytest

from quant.core.calendar import NEW_YORK
from quant.core.orders import OrderIntent, Side
from quant.core.regime import Label
from quant.risk.engine import RiskApprovedOrder
from tests.support.decide import r1_caps
from tests.support.risk import (
    CAL,
    DAY,
    EQUITY,
    HALF_DAY,
    NEXT_DAY,
    PRICE,
    closing,
    decisions,
    desk,
    last_fillable,
)

RED = pytest.mark.xfail(strict=True, reason="red:B8", raises=NotImplementedError)
STOP = 95.0


def _order(**overrides: object) -> OrderIntent | None:
    from quant.decide.intents import order

    fields: dict[str, object] = {
        "desired": 1.0,
        "shares": 0,
        "equity": EQUITY,
        "price": PRICE,
        "caps": r1_caps(),
        "label": Label.CALM_UP,
        "uncertain": False,
        "forced": False,
        "stop": STOP,
    }
    decision = overrides.pop("decision", decisions()[0])
    fields.update(overrides)
    return order(decision, **fields)  # type: ignore[arg-type]


# --- one order, whole shares -----------------------------------------------------------------


@RED
@pytest.mark.spec("ARCH§4.2", "SPEC§10")
def test_a_full_entry_scales_in_at_half_of_equity_per_order() -> None:
    first = _order()
    assert first is not None
    assert (first.side, first.quantity, first.target_exposure) == (Side.BUY, 500, 1.0)
    assert first.quantity * PRICE <= 0.5 * EQUITY
    second = _order(shares=500)
    assert second is not None
    assert (second.side, second.quantity) == (Side.BUY, 500)
    assert _order(shares=1000) is None  # there


@RED
@pytest.mark.spec("ARCH§4.2", "SPEC§10", "D9")
def test_a_full_exit_also_takes_two_orders_and_never_sells_more_than_is_held() -> None:
    first = _order(desired=0.0, shares=1000)
    assert first is not None
    assert (first.side, first.quantity, first.target_exposure) == (Side.SELL, 500, 0.0)
    assert first.stop_price is None
    assert not first.reduce_only
    last = _order(desired=0.0, shares=37)
    assert last is not None
    assert (last.side, last.quantity) == (Side.SELL, 37)


@RED
@pytest.mark.spec("ARCH§4.2", "SPEC§10")
def test_shares_are_whole_and_floored_on_decimal_values() -> None:
    from quant.decide.intents import target_shares

    assert 0.29 * EQUITY / PRICE < 290  # floats would floor this to 289
    assert target_shares(0.29, equity=EQUITY, price=PRICE) == 290
    assert target_shares(0.2999, equity=EQUITY, price=PRICE) == 299
    assert target_shares(0.0, equity=EQUITY, price=PRICE) == 0
    assert target_shares(1.0, equity=100_000.0, price=227.2) == 440
    buy = _order(desired=0.29)
    assert buy is not None
    assert buy.quantity == 290
    for bad in ({"desired": 1.2}, {"desired": -0.1}, {"price": 0.0}, {"equity": -1.0}):
        kwargs = {"desired": 0.5, "equity": EQUITY, "price": PRICE, **bad}
        with pytest.raises(ValueError, match=r"desired|equity|price"):
            target_shares(**kwargs)  # type: ignore[arg-type]


@RED
@pytest.mark.spec("ARCH§4.2")
def test_no_intent_when_no_whole_share_moves() -> None:
    assert _order(desired=0.0005) is None  # half a share
    assert _order(desired=0.5, shares=500) is None
    assert _order(desired=0.0, shares=0) is None
    with pytest.raises(ValueError, match="shares"):
        _order(shares=-1)


# --- the decision's times ----------------------------------------------------------------------


@RED
@pytest.mark.spec("A3", "ARCH§5", "SPEC§10")
@pytest.mark.parametrize(
    ("day", "et"), [(DAY, (15, 46)), (HALF_DAY, (12, 46))], ids=["full_day", "half_day"]
)
def test_the_overnight_cap_applies_on_the_last_fillable_decision(
    day: date, et: tuple[int, int]
) -> None:
    from quant.decide.intents import ceiling

    d = last_fillable(day)
    assert d.tau == datetime.combine(day, time(*et), NEW_YORK).astimezone(UTC)
    caps = r1_caps()
    assert ceiling(d, caps, Label.CALM_UP) == 0.5
    assert ceiling(d, caps, Label.CHOP) == 0.5
    assert ceiling(d, caps, Label.STRESS) == 0.25
    assert ceiling(d, caps, None) == 0.0
    earlier = next(x for x in decisions(day) if not x.last_fillable and not x.reduce_only)
    assert ceiling(earlier, caps, Label.CALM_UP) == 1.0


@RED
@pytest.mark.spec("A3", "ARCH§5", "SPEC§9")
def test_the_overnight_cut_is_a_forced_reduce_only_sell() -> None:
    from quant.decide.intents import ceiling
    from quant.decide.sizing import rebalance

    d = last_fillable(DAY)
    # 0.55x held at 15:46 under CALM_UP, which still targets 0.6x: down to 0.5x, deadband or not.
    plan = rebalance(
        0.60,
        0.55,
        ceiling=ceiling(d, r1_caps(), Label.CALM_UP),
        deadband=0.10,
        rule_forced=False,
        halted=False,
        frozen=False,
        locked=False,
        reduce_only=d.reduce_only,
    )
    assert (plan.desired, plan.move, plan.forced) == (0.5, True, True)
    cut = _order(decision=d, desired=plan.desired, shares=550, forced=plan.forced)
    assert cut is not None
    assert (cut.side, cut.quantity, cut.reduce_only, cut.target_exposure) == (
        Side.SELL,
        50,
        True,
        0.5,
    )


@RED
@pytest.mark.spec("A3", "ARCH§5", "SPEC§9")
@pytest.mark.parametrize("day", [DAY, HALF_DAY], ids=["full_day", "half_day"])
def test_the_closing_bar_decision_only_reduces_and_executes_at_the_next_open(day: date) -> None:
    d = closing(day)
    assert d.reduce_only
    assert not d.last_fillable
    assert d.executes_at == CAL.session(CAL.next_session(day)).open
    sell = _order(decision=d, desired=0.2, shares=400)
    assert sell is not None
    assert (sell.side, sell.quantity, sell.reduce_only, sell.tau) == (Side.SELL, 200, True, d.tau)
    with pytest.raises(ValueError, match="reduce_only"):
        _order(decision=d, desired=0.6, shares=400)  # a BUY on a closing bar


# --- the intent as the RiskEngine reads it ----------------------------------------------------


@RED
@pytest.mark.spec("ARCH§4.2", "ARCH§4.4", "SPEC§7", "SPEC§10")
def test_a_buy_carries_the_trades_stop_below_the_price() -> None:
    buy = _order(stop=STOP)
    assert buy is not None
    assert buy.stop_price == STOP < PRICE
    assert not buy.reduce_only
    # A scale-in carries the same stop.
    more = _order(shares=500, stop=STOP)
    assert more is not None
    assert more.stop_price == STOP
    for bad in (None, PRICE, 101.0, 0.0):
        with pytest.raises(ValueError, match="stop"):
            _order(stop=bad)


@RED
@pytest.mark.spec("ARCH§4.2", "SPEC§8")
def test_the_intent_names_its_decision_label_and_rule_five() -> None:
    from quant.core.ids import decision_id

    d = decisions()[2]
    buy = _order(decision=d, label=Label.CHOP, desired=0.3, uncertain=True)
    assert buy is not None
    assert (buy.decision_id, buy.tau, buy.leg) == (decision_id(d.tau), d.tau, 1)
    assert (buy.label, buy.uncertain) == (Label.CHOP, True)
    forced = _order(decision=d, desired=0.1, shares=400, forced=True, label=None)
    assert forced is not None
    assert forced.reduce_only
    assert forced.label is None
    # It is exactly the OrderIntent B9's engine reads.
    assert OrderIntent.model_validate(buy.model_dump()) == buy


@RED
@pytest.mark.spec("ARCH§4.2", "ARCH§4.4", "I4", "SPEC§10")
def test_a_happy_path_intent_passes_the_risk_engines_checks() -> None:
    d = decisions()[0]
    buy = _order(decision=d)
    assert buy is not None
    trading = desk()
    verdict = trading.check(buy)
    assert isinstance(verdict, RiskApprovedOrder), verdict
    assert verdict.quantity == 500
    # The second half the next bar, holding the first.
    d2 = decisions()[1]
    second = _order(decision=d2, shares=500)
    assert second is not None
    trading.clock.advance_to(d2.tau)
    verdict2 = trading.check(second, position=500)
    assert isinstance(verdict2, RiskApprovedOrder), verdict2
    # And the exit at the closing bar, reduce-only, the next morning.
    d3 = closing(DAY)
    exit_ = _order(decision=d3, desired=0.0, shares=1000)
    assert exit_ is not None
    assert exit_.reduce_only
    morning = desk(at=d3.executes_at, day=NEXT_DAY)
    sell = morning.check(exit_, position=1000)
    assert isinstance(sell, RiskApprovedOrder), sell
