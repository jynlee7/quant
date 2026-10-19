"""B8b: decision properties (PLAN B8; ARCHITECTURE §13's property tests; spec §9, §10, D9, A3).

decide() on the synthetic world's DECISION_DAY (tests.support.decide), from the state each
decision of that session starts from, with the filter's posterior, the active label, the live
trade, the position, the calibration and the flags drawn: it never goes short, never holds more
than the bar's cap after a BUY, never moves more than 0.5x of equity in one order, its label
probabilities sum to 1, and it sizes 0 in CRASH and when uncertain or halted, and buys nothing
while frozen or locked. The pure sizing and intent functions get the same properties over
plain numbers. Small example counts (the fast profile derandomizes them) keep the fast tier
fast.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant.core.orders import Side
from quant.core.regime import Label
from tests.support.calibration import sizing_calibration
from tests.support.decide import DecideFixture, day_states, engine_config, r1_caps, r1_playbook

EQUITY = 100_000.0


def _x(v: float) -> Fraction:
    return Fraction(repr(float(v)))


@st.composite
def _cases(draw: st.DrawFn) -> dict[str, Any]:
    weights = draw(st.lists(st.floats(1e-6, 1.0), min_size=2, max_size=2))
    return {
        "k": draw(st.integers(1, 6)),
        "posterior": [w / math.fsum(weights) for w in weights],
        "active": draw(st.sampled_from([None, *Label])),
        "in_trade": draw(st.booleans()),
        "position": draw(st.integers(0, 1600)),
        "halted": draw(st.booleans()),
        "frozen": draw(st.booleans()),
        "locked": draw(st.booleans()),
        "calibration": draw(st.sampled_from([None, "passing", "failed"])),
    }


def _decide(case: dict[str, Any]) -> tuple[Any, Any]:
    from quant.decide.core import Account, Flags
    from quant.decide.primitives import TradeState

    fx = DecideFixture.get()
    d, base = day_states()[case["k"]]
    assert base.filter is not None
    logs = [math.log(p) for p in case["posterior"]]
    state = base.model_copy(update={"filter": base.filter.model_copy(update={"log_probs": logs})})
    active = case["active"]
    trade = fx.in_trade() if case["in_trade"] else None
    switching = state.switching.model_copy(
        update={"active": active, "challenger": None, "count": 0, "blocked": False, "reconfirm": 0}
    )
    state = state.model_copy(
        update={"switching": switching, "trade": trade if trade is not None else TradeState.flat()}
    )
    calibration = {
        None: None,
        "passing": sizing_calibration(),
        "failed": sizing_calibration(outcome_passes=False),
    }[case["calibration"]]
    out = fx.decide(
        state,
        decision=d,
        calibration=calibration,
        account=Account(equity=EQUITY, position=case["position"]),
        flags=Flags(
            halted=case["halted"], drift_frozen=case["frozen"], daily_loss_lock=case["locked"]
        ),
    )
    return d, out


@pytest.mark.spec("D9", "SPEC§9", "SPEC§10", "A3", "ARCH§13", "SPEC§8", "C5")
@settings(max_examples=25)
@given(_cases())
def test_decide_never_goes_short_or_over_a_cap_and_sizes_zero_where_it_must(
    case: dict[str, Any],
) -> None:
    d, out = _decide(case)
    record = out.record
    caps = r1_caps()
    position = case["position"]
    assert len(out.intents) <= 1
    if record.filter is not None:
        assert math.isclose(math.fsum(record.filter.label_probs.values()), 1.0, abs_tol=1e-12)
        assert math.isclose(math.fsum(record.filter.next_label_probs.values()), 1.0, abs_tol=1e-12)
    price = record.target.price
    for intent in out.intents:
        # Never over 0.5x of equity in one order, BUY or SELL.
        assert intent.quantity * _x(price) <= _x(caps.single_order_cap) * _x(EQUITY)
        if intent.side is Side.SELL:
            assert intent.quantity <= position  # never short (D9)
            continue
        # A BUY: never on a closing bar, never past the bar's cap, always with its stop.
        assert not d.reduce_only
        assert not intent.reduce_only
        assert intent.stop_price is not None
        assert 0 < intent.stop_price < price
        ceiling = min(caps.max_position, caps.cap(record.active))
        if d.last_fillable:
            ceiling = min(ceiling, caps.overnight_cap)
        assert (position + intent.quantity) * _x(price) <= _x(ceiling) * _x(EQUITY)
        assert record.active not in (None, Label.CRASH, Label.STRESS)
        assert not case["halted"]
        assert not case["frozen"]
        assert not case["locked"]
        assert not any(t.fired for t in record.trace if t.rule == "R5")
    # Size 0 in CRASH, when uncertain and while halted.
    if record.active is Label.CRASH or case["halted"]:
        assert record.target.value == 0.0
    if any(t.fired for t in record.trace if t.rule == "R5"):
        assert record.target.value == 0.0


@pytest.mark.spec("D9", "SPEC§10", "ARCH§13", "ARCH§4.2")
@settings(max_examples=50)
@given(
    desired=st.floats(0.0, 1.0),
    shares=st.integers(0, 3000),
    price=st.floats(1.0, 5000.0),
    equity=st.floats(1_000.0, 1e7),
    forced=st.booleans(),
    index=st.integers(0, 6),
)
def test_an_order_never_shorts_or_moves_more_than_half_of_equity(
    desired: float, shares: int, price: float, equity: float, forced: bool, index: int
) -> None:
    from quant.decide.intents import order
    from tests.support.risk import decisions

    d = decisions()[index]
    if d.reduce_only:
        desired = min(desired, shares * price / equity)
    intent = order(
        d,
        desired=desired,
        shares=shares,
        equity=equity,
        price=price,
        caps=r1_caps(),
        label=Label.CALM_UP,
        uncertain=False,
        forced=forced,
        stop=price * 0.95,
    )
    if intent is None:
        return
    assert intent.quantity * _x(price) <= _x(0.5) * _x(equity)
    if intent.side is Side.SELL:
        assert intent.quantity <= shares
    else:
        assert (shares + intent.quantity) * _x(price) <= _x(desired) * _x(equity)
        assert not d.reduce_only


@pytest.mark.spec("SPEC§9", "SPEC§10", "ARCH§13", "C5")
@settings(max_examples=50)
@given(
    target=st.floats(0.0, 1.0),
    current=st.floats(0.0, 1.5),
    ceiling=st.sampled_from([0.0, 0.25, 0.5, 1.0]),
    rule_forced=st.booleans(),
    halted=st.booleans(),
    frozen=st.booleans(),
    locked=st.booleans(),
    reduce_only=st.booleans(),
)
def test_rebalancing_never_passes_the_ceiling_and_never_raises_when_it_may_not(
    target: float,
    current: float,
    ceiling: float,
    rule_forced: bool,
    halted: bool,
    frozen: bool,
    locked: bool,
    reduce_only: bool,
) -> None:
    from quant.decide.sizing import rebalance

    plan = rebalance(
        target,
        current,
        ceiling=ceiling,
        deadband=engine_config().sizing.deadband,
        rule_forced=rule_forced,
        halted=halted,
        frozen=frozen,
        locked=locked,
        reduce_only=reduce_only,
    )
    assert plan.desired <= ceiling
    if frozen or locked or reduce_only:
        assert plan.desired <= current
    if current > ceiling:
        assert plan.move
        assert plan.forced


@pytest.mark.spec("SPEC§9", "SPEC§7", "ARCH§13")
@settings(max_examples=50)
@given(
    label=st.sampled_from(list(Label)),
    f_star=st.floats(0.0, 1e4),
    p=st.floats(0.0, 1.0),
    entropy=st.floats(0.0, 1.0),
    m_rule4=st.sampled_from([0.5, 1.0]),
    m_rule5=st.sampled_from([0.0, 1.0]),
    calibrated=st.booleans(),
)
def test_the_target_never_exceeds_the_cap_or_max_size_and_is_zero_in_crash(
    label: Label,
    f_star: float,
    p: float,
    entropy: float,
    m_rule4: float,
    m_rule5: float,
    calibrated: bool,
) -> None:
    from quant.decide.sizing import size

    caps = r1_caps()
    s = size(
        label,
        playbook=r1_playbook(label),
        cap=caps.cap(label),
        calibration=sizing_calibration() if calibrated else None,
        f_star=f_star,
        label_prob=p,
        entropy=entropy,
        m_rule4=m_rule4,
        m_rule5=m_rule5,
        config=engine_config().sizing,
    )
    assert 0.0 <= s.value <= min(caps.cap(label), r1_playbook(label).max_size)
    if label is Label.CRASH or m_rule5 == 0.0 or f_star == 0.0:
        assert s.value == 0.0
