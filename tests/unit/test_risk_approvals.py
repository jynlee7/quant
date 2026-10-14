"""B9: approvals are built only by the risk engine, and each is honoured once (I4, K5).

The K5 scan already stops `src/` from constructing or subclassing the approved types (B0). These
tests forge at runtime, outside quant.risk.engine, the way B10's tests forge TrialTickets: plain
objects, subclasses, objects built without the token or with a stolen one, approvals changed
after issue, other engines' approvals and approvals from before a restart. `RiskEngine.spend`
must refuse every one, and also reused and expired approvals, while genuine approvals still work.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from quant.core.canonical import canonical_dumps
from quant.core.errors import ApprovalError
from quant.core.orders import Side
from quant.risk import engine as engine_module
from quant.risk.engine import (
    MemorySpentApprovals,
    RiskApprovedCancel,
    RiskApprovedFlatten,
    RiskApprovedOrder,
    RiskApprovedStop,
    verdict_record,
)
from tests.support.risk import Desk, desk, intent

Approved = RiskApprovedOrder | RiskApprovedCancel | RiskApprovedFlatten | RiskApprovedStop
Issue = Callable[[Desk], Approved]


def _order(d: Desk) -> RiskApprovedOrder:
    verdict = d.check(intent(Side.BUY, 100))
    assert type(verdict) is RiskApprovedOrder
    return verdict


def _cancel(d: Desk) -> RiskApprovedCancel:
    return d.engine.approve_cancel("eng1-d20180306T1546Z-1-1", reason="re-quote")


def _flatten(d: Desk) -> RiskApprovedFlatten:
    return d.engine.approve_flatten(reason="kill")


def _stop(d: Desk) -> RiskApprovedStop:
    return d.stop(100)  # a third of the 300 shares the snapshot holds, so three fit


KINDS: dict[str, Issue] = {"order": _order, "cancel": _cancel, "flatten": _flatten, "stop": _stop}
# One field per kind that a forger would widen after issue.
WIDENED: dict[type, tuple[str, object]] = {
    RiskApprovedOrder: ("quantity", 1_000_000),
    RiskApprovedCancel: ("client_order_id", "someone-elses-order"),
    RiskApprovedFlatten: ("symbol", "QQQ"),
    RiskApprovedStop: ("quantity", 1_000_000),
}


def fields_of(approval: Approved) -> dict[str, Any]:
    names = ("approval_id", "issued_at", *type(approval).__slots__)
    return {name: getattr(approval, name) for name in names}


# Each forgery is built outside quant.risk.engine from a genuine approval issued by `d`.
Forgery = Callable[[Desk, Approved, Issue], object]


def _plain(d: Desk, real: Approved, issue: Issue) -> object:
    return SimpleNamespace(**fields_of(real), _token=engine_module._TOKEN)


def _subclass(d: Desk, real: Approved, issue: Issue) -> object:
    def init(self: object, **fields: Any) -> None:
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    forged_type = type("Forged", (type(real),), {"__init__": init})
    return forged_type(**fields_of(real), _token=engine_module._TOKEN)


def _without_token(d: Desk, real: Approved, issue: Issue) -> object:
    forged = object.__new__(type(real))
    for name, value in fields_of(real).items():
        object.__setattr__(forged, name, value)
    return forged


def _other_token(d: Desk, real: Approved, issue: Issue) -> object:
    forged = _without_token(d, real, issue)
    object.__setattr__(forged, "_token", object())
    return forged


def _stolen_token(d: Desk, real: Approved, issue: Issue) -> object:
    return type(real)(token=engine_module._TOKEN, **fields_of(real))  # never issued


def _changed(d: Desk, real: Approved, issue: Issue) -> object:
    other = issue(d)
    name, value = WIDENED[type(other)]
    object.__setattr__(other, name, value)
    return other


def _other_engine(d: Desk, real: Approved, issue: Issue) -> object:
    return issue(desk())


FORGERIES: dict[str, Forgery] = {
    "plain object": _plain,
    "subclass": _subclass,
    "built without the token": _without_token,
    "another token": _other_token,
    "stolen token, never issued": _stolen_token,
    "changed after issue": _changed,
    "another engine": _other_engine,
}


@pytest.mark.spec("I4", "K5", "ARCH§4.4")
@pytest.mark.parametrize("issue", KINDS.values(), ids=KINDS.keys())
@pytest.mark.parametrize("forge", FORGERIES.values(), ids=FORGERIES.keys())
def test_i4_approvals_built_outside_the_engine_are_refused(issue: Issue, forge: Forgery) -> None:
    d = desk()
    real = issue(d)
    forged = forge(d, real, issue)
    with pytest.raises(ApprovalError):
        d.engine.spend(forged)
    genuine = issue(d)  # the engine still honours what it issues
    assert d.engine.spend(genuine) is genuine


@pytest.mark.spec("I4", "K5", "ARCH§4.4")
def test_approved_types_demand_the_token_and_are_immutable_and_uncopyable() -> None:
    d = desk()
    real = _order(d)
    with pytest.raises(ApprovalError):
        RiskApprovedOrder(token=object(), **fields_of(real))
    with pytest.raises(ApprovalError):
        RiskApprovedCancel(
            token=None,
            approval_id="C1-0",
            client_order_id="x",
            reason="r",
            issued_at=real.issued_at,
        )
    with pytest.raises(ApprovalError):
        RiskApprovedFlatten(
            token="token", approval_id="F1-0", symbol="SPY", reason="r", issued_at=real.issued_at
        )
    for approval in (real, _cancel(d), _flatten(d)):
        with pytest.raises(AttributeError):
            approval.approval_id = "other"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            del approval.approval_id
        with pytest.raises(TypeError):
            copy.copy(approval)
        with pytest.raises(TypeError):
            copy.deepcopy(approval)
        assert "token" not in repr(approval).lower()
    assert canonical_dumps(verdict_record(real))  # a record has a canonical form, token-free
    assert "_token" not in verdict_record(real)


@pytest.mark.spec("I4", "ARCH§4.4")
@pytest.mark.parametrize("issue", KINDS.values(), ids=KINDS.keys())
def test_each_approval_is_spendable_once(issue: Issue) -> None:
    d = desk()
    approval = issue(d)
    assert d.engine.spend(approval) is approval
    with pytest.raises(ApprovalError):
        d.engine.spend(approval)


@pytest.mark.spec("I4", "ARCH§4.4", "D7")
def test_an_order_approval_expires_at_the_end_of_its_bar() -> None:
    # The 10:46 ET decision executes in the 10:30-11:30 bar, so its approval expires at 11:30.
    d = desk()
    early = _order(d)
    late = d.check(intent(Side.BUY, 100, leg=2))  # another leg: a re-check would replace early
    assert type(late) is RiskApprovedOrder
    assert early.expires_at == late.expires_at
    d.clock.advance_to(early.expires_at - timedelta(microseconds=1))
    assert d.engine.spend(early) is early
    d.clock.advance_to(late.expires_at)
    with pytest.raises(ApprovalError):
        d.engine.spend(late)


@pytest.mark.spec("I4", "ARCH§4.4")
def test_spent_ids_go_to_the_store_and_the_store_refuses_them_again() -> None:
    spent = MemorySpentApprovals()
    d = desk(spent=spent)
    first = _order(d)
    d.engine.spend(first)
    assert spent.is_spent(first.approval_id)
    assert not spent.spend(first.approval_id, d.clock.now())  # a second record changes nothing

    # An id another process already recorded (a restart racing the adapter) is refused.
    second = _order(d)
    assert spent.spend(second.approval_id, d.clock.now())
    with pytest.raises(ApprovalError):
        d.engine.spend(second)


@pytest.mark.spec("I4", "ARCH§4.4", "ARCH§14")
def test_a_restarted_engine_issues_fresh_ids_that_its_store_accepts() -> None:
    spent = MemorySpentApprovals()
    before = desk(spent=spent)
    spent_before = [_order(before), _cancel(before), _flatten(before)]
    for approval in spent_before:
        before.engine.spend(approval)
    unspent = [_order(before), _cancel(before), _flatten(before)]  # left over at the restart
    after = desk(spent=spent, state=before.engine.state, at=before.clock.now())
    new = [_order(after), _cancel(after), _flatten(after)]
    old_ids = {a.approval_id for a in spent_before + unspent}
    assert old_ids.isdisjoint(a.approval_id for a in new)  # numbered on from the saved state
    for approval in new:
        assert after.engine.spend(approval) is approval
    for approval in unspent:  # the new engine object never issued them
        with pytest.raises(ApprovalError):
            after.engine.spend(approval)
