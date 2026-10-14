"""B9c: the fourth approved type, RiskApprovedStop (your call of 2026-10-14; I4, K5, D9).

A protective stop reaches the broker only through a RiskApprovedStop that the engine built and
registered, and `spend` honours it once, as it does the other three types. The engine approves
a stop-market SELL only below the current bid, and only for shares no other SELL covers: the
working SELL orders on the snapshot, the SELLs and stops spent since it was taken, and the SELL
and stop approvals still outstanding. So a stop never sells a share twice and never takes the
position short. Forgeries and reuse of every type, stops included, are in test_risk_approvals.
"""

from __future__ import annotations

import copy
import math
from datetime import timedelta
from typing import Any

import pytest

from quant.core.errors import ApprovalError
from quant.core.orders import OrderKind, Side
from quant.risk.engine import RiskApprovedOrder, RiskApprovedStop, Veto
from tests.support.risk import STOP, desk, intent, order

RED = pytest.mark.xfail(strict=True, reason="red:B9", raises=NotImplementedError)
S = timedelta(seconds=1)
US = timedelta(microseconds=1)


@RED
@pytest.mark.spec("I4", "K5", "ARCH§4.4")
def test_a_stop_carries_what_was_approved_and_is_sealed_like_the_others() -> None:
    d = desk()
    now = d.clock.now()
    stop = d.stop(300, stop_price=95.0, reason="protect")
    assert type(stop) is RiskApprovedStop
    assert (stop.symbol, stop.quantity, stop.stop_price, stop.reason) == (
        "SPY",
        300,
        95.0,
        "protect",
    )
    assert stop.approval_id.startswith("S")
    assert stop.issued_at == now
    assert stop.expires_at == now + 5 * S  # the snapshot's snapshot_max_age_s
    with pytest.raises(ApprovalError):
        RiskApprovedStop(
            token=object(),
            approval_id="S1-0",
            symbol="SPY",
            quantity=1,
            stop_price=95.0,
            reason="r",
            issued_at=now,
            expires_at=now,
        )
    with pytest.raises(AttributeError):
        stop.quantity = 3_000  # type: ignore[misc]
    with pytest.raises(TypeError):
        copy.copy(stop)
    with pytest.raises(TypeError):
        copy.deepcopy(stop)
    assert "token" not in repr(stop).lower()
    assert d.engine.spend(stop) is stop


@RED
@pytest.mark.spec("I4", "ARCH§4.4", "ARCH§14")
def test_a_stop_from_before_a_restart_is_refused() -> None:
    before = desk()
    left_over = before.stop(300)
    after = desk(state=before.engine.state, at=before.clock.now())
    with pytest.raises(ApprovalError):
        after.engine.spend(left_over)
    fresh = after.stop(300)
    assert fresh.approval_id != left_over.approval_id
    assert after.engine.spend(fresh) is fresh


REFUSED: dict[str, dict[str, Any]] = {
    "more shares than are held": {"quantity": 301},
    "a flat account": {"quantity": 1, "position": 0},
    "a short account": {"quantity": 1, "position": -100},
    "no shares": {"quantity": 0},
    "a bool for shares": {"quantity": True},
    "a float for shares": {"quantity": 1.5},
    "a stop at the bid": {"stop_price": 100.0},
    "a stop above the bid": {"stop_price": 100.5},
    "a stop at zero": {"stop_price": 0.0},
    "a stop that isn't finite": {"stop_price": math.nan},
    "a stale snapshot": {"snapshot_age": 5 * S + US},
    "a snapshot from the future": {"snapshot_age": -S},
    "a stale quote": {"quote_age": 5 * S + US},
    "a quote from the future": {"quote_age": -S},
    "a crossed quote": {"bid": 100.0, "ask": 99.9, "stop_price": 99.0},
    "shares a working exit sells": {
        "quantity": 201,
        "open_orders": (order("x-1", Side.SELL, OrderKind.LIMIT, 100, 99.9),),
    },
    "shares a working stop covers": {
        "quantity": 1,
        "open_orders": (order("stop-1", Side.SELL, OrderKind.STOP, 300, STOP),),
    },
}


@RED
@pytest.mark.spec("I4", "D9", "SPEC§10")
@pytest.mark.parametrize("kwargs", REFUSED.values(), ids=REFUSED.keys())
def test_a_stop_is_refused_unless_it_sits_below_the_bid_over_shares_no_sell_covers(
    kwargs: dict[str, Any],
) -> None:
    d = desk()
    before = d.engine.state
    with pytest.raises(ApprovalError):
        d.stop(**{"quantity": 300, **kwargs})
    assert d.engine.state == before  # a refusal numbers no id
    assert d.engine.spend(d.stop(300)) is not None  # the shares are still free


@RED
@pytest.mark.spec("I4", "D9", "SPEC§10")
def test_the_boundaries_pass() -> None:
    d = desk()
    exit_ = order("x-1", Side.SELL, OrderKind.LIMIT, 100, 99.9)
    assert d.stop(200, open_orders=(exit_,)).quantity == 200  # exactly what is left
    d2 = desk()
    assert d2.stop(1, stop_price=99.99, snapshot_age=5 * S, quote_age=5 * S).stop_price == 99.99


@RED
@pytest.mark.spec("I4", "D9", "ARCH§8.1")
def test_sells_and_stops_reserve_shares_from_one_another_until_a_snapshot_shows_them() -> None:
    d = desk()
    sell = d.check(intent(Side.SELL, 100), position=300)
    assert isinstance(sell, RiskApprovedOrder)
    with pytest.raises(ApprovalError):
        d.stop(201)  # the outstanding SELL holds 100 of the 300
    stop = d.stop(200)
    with pytest.raises(ApprovalError):
        d.stop(1)
    veto = d.check(intent(Side.SELL, 1, leg=2), position=300)
    assert isinstance(veto, Veto)  # the outstanding stop holds the rest
    d.engine.spend(stop)
    d.engine.spend(sell)
    with pytest.raises(ApprovalError):
        d.stop(1)  # both are in flight: a snapshot taken now can't show them yet
    d.clock.advance(S)
    shown = (
        order("x-1", Side.SELL, OrderKind.LIMIT, 100, 99.9),
        order("kill-stop-1", Side.SELL, OrderKind.STOP, 200, STOP),
    )
    with pytest.raises(ApprovalError):
        d.stop(1, open_orders=shown)
    assert d.stop(100, position=400, open_orders=shown).quantity == 100


@RED
@pytest.mark.spec("I4", "ARCH§4.4")
def test_a_stop_expires_with_its_snapshot_and_frees_its_shares() -> None:
    d = desk()
    stop = d.stop(300)
    d.clock.advance(5 * S)
    with pytest.raises(ApprovalError):
        d.engine.spend(stop)
    assert d.stop(300).quantity == 300


@RED
@pytest.mark.spec("I4", "D9", "ARCH§8.1", "ARCH§8.3")
@pytest.mark.parametrize("revoke", ["a spent flatten", "a kill", "fewer shares"])
def test_an_outstanding_stop_is_revoked_when_its_shares_may_be_gone(revoke: str) -> None:
    d = desk()
    stop = d.stop(300)
    if revoke == "a spent flatten":
        d.engine.spend(d.engine.approve_flatten(reason="kill"))
    elif revoke == "a kill":
        d.engine.revoke_orders("kill switch")
    else:  # the next check sees 100 shares, fewer than the stop reserves
        d.check(intent(Side.SELL, 50), position=100)
    with pytest.raises(ApprovalError):
        d.engine.spend(stop)


@RED
@pytest.mark.spec("I4", "C3", "SPEC§10")
def test_a_stop_is_issued_and_spent_while_halted_and_is_never_a_submission() -> None:
    d = desk()
    assert d.engine.halt("drawdown")
    stop = d.stop(300)
    assert d.engine.spend(stop) is stop
    assert d.engine.state.submissions == 0
    with pytest.raises(ApprovalError):
        d.engine.spend(stop)
