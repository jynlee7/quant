"""B9a review: approvals that are issued but not yet spent are still risk (spec §10, D9, I4).

An approval is leave to send an order later, so until it is spent, released or expired it holds
its exposure and its shares. These tests issue several approvals before spending any, and change
what the engine knows between issue and spend:

- outstanding BUYs count toward gross exposure, the label cap and the overnight cap, and a
  re-check of the same decision and leg replaces the earlier approval;
- outstanding SELLs reserve shares, so together they never sell more than is held;
- `spend` refuses a BUY once an entry lock, a halt or a known block holds, and anything whose
  intent changed after issue;
- the inclusive monetary boundaries (2%, 10%, the caps, the ±10 bp collar) hold at their exact
  decimal values, not only at round numbers.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from quant.core.errors import ApprovalError
from quant.core.orders import Side, TradingMode
from quant.core.regime import Label
from quant.risk.engine import PendingApproval, RiskApprovedOrder, RiskCheck, RiskVerdict, Veto
from tests.support.risk import HALF_DAY, decisions, desk, intent, last_fillable


def approved(verdict: RiskVerdict) -> RiskApprovedOrder:
    assert type(verdict) is RiskApprovedOrder, verdict
    return verdict


def vetoed(verdict: RiskVerdict, check: RiskCheck) -> Veto:
    assert isinstance(verdict, Veto), verdict
    assert verdict.check is check, verdict
    return verdict


# --- R1/C1: outstanding BUY approvals hold their exposure -----------------------------------------


@pytest.mark.spec("SPEC§10", "ARCH§8.1", "D13", "I4")
def test_outstanding_buys_count_toward_gross_exposure() -> None:
    d = desk()
    first = approved(d.check(intent(Side.BUY, 500)))
    approved(d.check(intent(Side.BUY, 500, leg=2)))  # exactly 1.0x with the first
    vetoed(d.check(intent(Side.BUY, 1, leg=3)), RiskCheck.LIMITS)
    d.engine.spend(first)  # spent: the broker's snapshot carries it from now on
    d.clock.advance(timedelta(seconds=1))
    vetoed(d.check(intent(Side.BUY, 1, leg=4), position=500), RiskCheck.LIMITS)


@pytest.mark.spec("SPEC§10", "ARCH§8.1", "D13")
def test_outstanding_buys_count_toward_the_label_cap() -> None:
    d = desk()
    approved(d.check(intent(Side.BUY, 300, label=Label.CHOP)))
    vetoed(d.check(intent(Side.BUY, 201, label=Label.CHOP, leg=2)), RiskCheck.LIMITS)
    approved(d.check(intent(Side.BUY, 200, label=Label.CHOP, leg=3)))  # exactly 0.5x


@pytest.mark.spec("A3", "SPEC§10", "ARCH§8.1")
def test_outstanding_buys_count_toward_the_overnight_cap() -> None:
    last = last_fillable(HALF_DAY)
    d = desk(at=last.tau + timedelta(seconds=30), day=HALF_DAY)
    approved(d.check(intent(Side.BUY, 300, tau=last.tau)))
    vetoed(d.check(intent(Side.BUY, 201, tau=last.tau, leg=2)), RiskCheck.LIMITS)


@pytest.mark.spec("D7", "SPEC§10", "ARCH§8.1", "ARCH§8.4")
def test_approving_pending_orders_counts_the_ones_already_approved() -> None:
    # Two $30,000 CHOP entries wait for you; approving both would make 0.6x against a 0.5x cap.
    d = desk(mode=TradingMode.LIVE)
    first = d.check(intent(Side.BUY, 300, label=Label.CHOP))
    second = d.check(intent(Side.BUY, 300, label=Label.CHOP, leg=2))
    assert type(first) is PendingApproval
    assert type(second) is PendingApproval
    approved(d.engine.approve_pending(first, d.ctx()))
    vetoed(d.engine.approve_pending(second, d.ctx()), RiskCheck.LIMITS)


@pytest.mark.spec("SPEC§10", "ARCH§8.1", "ARCH§8.2", "I4")
def test_a_recheck_of_the_same_leg_replaces_its_earlier_approval() -> None:
    # A re-quote is checked again and gets a new approval; the old one can no longer be sent.
    d = desk()
    old = approved(d.check(intent(Side.BUY, 500, label=Label.CHOP)))
    new = approved(d.check(intent(Side.BUY, 500, label=Label.CHOP)))  # not blocked by the old
    with pytest.raises(ApprovalError):
        d.engine.spend(old)
    assert d.engine.spend(new) is new


@pytest.mark.spec("SPEC§10", "ARCH§8.1", "I4")
def test_a_released_approval_frees_its_exposure_and_cannot_be_spent() -> None:
    d = desk()
    held = approved(d.check(intent(Side.BUY, 500, label=Label.CHOP)))
    vetoed(d.check(intent(Side.BUY, 1, label=Label.CHOP, leg=2)), RiskCheck.LIMITS)
    d.engine.release(held)
    d.engine.release(held)  # releasing twice changes nothing
    approved(d.check(intent(Side.BUY, 500, label=Label.CHOP, leg=3)))
    with pytest.raises(ApprovalError):
        d.engine.spend(held)


@pytest.mark.spec("SPEC§10", "ARCH§8.1", "I4")
def test_an_expired_approval_holds_no_exposure() -> None:
    first, second = decisions()[0], decisions()[1]
    d = desk()
    old = approved(d.check(intent(Side.BUY, 500, label=Label.CHOP, tau=first.tau)))
    d.clock.advance_to(second.tau + timedelta(seconds=30))
    assert d.clock.now() >= old.expires_at
    approved(d.check(intent(Side.BUY, 500, label=Label.CHOP, tau=second.tau)))


@pytest.mark.spec("SPEC§10", "ARCH§8.1", "I4")
def test_a_spent_buy_counts_until_a_snapshot_taken_after_the_spend() -> None:
    d = desk()
    d.engine.spend(approved(d.check(intent(Side.BUY, 500, label=Label.CHOP))))
    d.clock.advance(timedelta(seconds=1))
    # The snapshot predates the spend, so it can't show the order yet.
    vetoed(
        d.check(intent(Side.BUY, 1, label=Label.CHOP, leg=2), snapshot_age=timedelta(seconds=2)),
        RiskCheck.LIMITS,
    )


# --- C3: outstanding SELL approvals reserve shares -------------------------------------------------


@pytest.mark.spec("D9", "SPEC§10", "ARCH§8.1")
def test_outstanding_sells_reserve_their_shares() -> None:
    d = desk()
    approved(d.check(intent(Side.SELL, 60), position=100))
    assert approved(d.check(intent(Side.SELL, 100, leg=2), position=100)).quantity == 40
    vetoed(d.check(intent(Side.SELL, 1, leg=3), position=100), RiskCheck.LIMITS)


@pytest.mark.spec("D9", "SPEC§10", "ARCH§8.1", "I4")
def test_two_sell_approvals_never_sell_more_than_is_held() -> None:
    d = desk()
    sells = [d.check(intent(Side.SELL, 100, leg=leg), position=100) for leg in (1, 2)]
    sent = 0
    for verdict in sells:
        if isinstance(verdict, RiskApprovedOrder):
            d.engine.spend(verdict)
            sent += verdict.quantity
    assert sent <= 100


@pytest.mark.spec("D9", "SPEC§10", "ARCH§8.1", "I4")
def test_a_sell_is_revoked_when_a_snapshot_shows_fewer_shares_than_it_reserves() -> None:
    # The broker-side stop fired after the SELL was approved: nothing is left to sell.
    d = desk()
    sell = approved(d.check(intent(Side.SELL, 100), position=100))
    vetoed(d.check(intent(Side.SELL, 100, leg=2), position=0), RiskCheck.LIMITS)
    with pytest.raises(ApprovalError):
        d.engine.spend(sell)


@pytest.mark.spec("D9", "ARCH§8.3", "I4")
def test_a_spent_flatten_revokes_every_outstanding_order() -> None:
    d = desk()
    sell = approved(d.check(intent(Side.SELL, 100), position=100))
    buy = approved(d.check(intent(Side.BUY, 100, leg=2), position=100))
    d.engine.spend(d.engine.approve_flatten(reason="kill"))
    for approval in (sell, buy):
        with pytest.raises(ApprovalError):
            d.engine.spend(approval)


# --- R3/C2: spend re-checks what can change after issue --------------------------------------------


@pytest.mark.spec("SPEC§10", "C8", "ARCH§8.1", "I4")
def test_spend_refuses_a_buy_once_the_daily_loss_lock_sets() -> None:
    d = desk()
    buy = approved(d.check(intent(Side.BUY, 100)))
    sell = approved(d.check(intent(Side.SELL, 100, leg=2), position=100))
    assert d.engine.mark(98_000.0).locked
    with pytest.raises(ApprovalError):
        d.engine.spend(buy)
    assert d.engine.spend(sell) is sell  # reductions still go


@pytest.mark.spec("SPEC§10", "SPEC§8", "C5", "ARCH§8.1", "I4")
@pytest.mark.parametrize(
    "block",
    [{"halted": True}, {"drift_frozen": True}, {"healthy": False}],
    ids=["HALTED in control.db", "drift freeze", "rule 6"],
)
def test_spend_refuses_a_buy_once_a_check_learns_of_a_block(block: dict[str, bool]) -> None:
    d = desk()
    buy = approved(d.check(intent(Side.BUY, 100)))
    sell = approved(d.check(intent(Side.SELL, 100, leg=2), position=100))
    assert isinstance(d.check(intent(Side.BUY, 100, leg=3), position=100, **block), Veto)  # type: ignore[arg-type]
    with pytest.raises(ApprovalError):
        d.engine.spend(buy)
    assert d.engine.spend(sell) is sell


# --- R2: an approval's intent cannot change after issue --------------------------------------------


@pytest.mark.spec("I4", "K5", "ARCH§4.4")
@pytest.mark.parametrize(
    ("name", "value"), [("side", Side.BUY), ("quantity", 1_000_000)], ids=["side", "quantity"]
)
def test_spend_refuses_an_approval_whose_intent_changed(name: str, value: object) -> None:
    d = desk()
    approval = approved(d.check(intent(Side.SELL, 1_500), position=1_500))
    object.__setattr__(approval.intent, name, value)
    with pytest.raises(ApprovalError):
        d.engine.spend(approval)


# --- R4, R5/C4: monetary boundaries at their exact decimal values -----------------------------------


@pytest.mark.spec("SPEC§10", "C8", "D13")
def test_the_daily_loss_lock_sets_at_exactly_two_percent_of_any_equity() -> None:
    inside = desk(equity=100_000.50)
    assert not inside.engine.mark(98_000.50).locked
    d = desk(equity=100_000.50)
    outcome = d.engine.mark(98_000.49)  # 100,000.50 - 2,000.01: exactly 2%
    assert outcome.locked
    assert outcome.flatten


@pytest.mark.spec("SPEC§10", "ARCH§8.3", "C8", "D13")
def test_the_drawdown_trigger_fires_at_exactly_ten_percent_of_any_peak() -> None:
    inside = desk(equity=100_000.30)
    assert not inside.engine.mark(90_000.28).kill
    d = desk(equity=100_000.30)
    assert d.engine.mark(90_000.27).kill  # 100,000.30 - 10,000.03: exactly 10%


@pytest.mark.spec("SPEC§10", "ARCH§8.1", "D13")
@pytest.mark.parametrize("label", [Label.CALM_UP, Label.CHOP], ids=["single order", "CHOP"])
def test_the_caps_hold_at_exactly_their_dollar_value(label: Label) -> None:
    # 100 shares at $512.94 are $51,294: exactly 0.5x of $102,588, the single-order cap and
    # CHOP's cap. Float multiplication puts the product a hair above the cap.
    price, equity = 512.94, 102_588.0
    d = desk(equity=equity)
    ctx = {"equity": equity, "bid": price, "ask": price, "limit": price}
    approved(d.check(intent(Side.BUY, 100, label=label, stop=500.0), **ctx))  # type: ignore[arg-type]
    vetoed(d.check(intent(Side.BUY, 101, label=label, stop=500.0, leg=2), **ctx), RiskCheck.LIMITS)  # type: ignore[arg-type]


@pytest.mark.spec("ARCH§8.1", "ARCH§8.2", "SPEC§10")
def test_the_collar_passes_exactly_ten_basis_points_from_the_quote() -> None:
    # ARCHITECTURE §8.2 buys at the ask + 10 bp and sells at the bid - 10 bp. For quotes on a
    # whole ten dollars those are whole cents, so execution sends them exactly.
    vetoes = []
    for dollars in range(50, 710, 10):
        quote = Decimal(dollars)
        band = quote * 10 / 10_000
        for side, limit in ((Side.BUY, quote + band), (Side.SELL, quote - band)):
            d = desk()
            verdict = d.check(
                intent(side, 1, stop=float(quote) * 0.9),
                position=1,
                bid=float(quote),
                ask=float(quote),
                limit=float(limit),
            )
            if not isinstance(verdict, RiskApprovedOrder):
                vetoes.append((side.value, str(limit)))
    assert vetoes == []
