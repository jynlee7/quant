"""B9: RiskEngine.check runs ARCHITECTURE §8.1's seven checks in order (spec §10, D7, D13, C3).

Each check is shown vetoing at its boundary and passing just inside it. The desk trades a
$100,000 account at $100 a share (tests.support.risk), so 1,000 shares are exactly 1.0x and
each cap lands on a whole number of shares. Float boundaries (ages, the collar) are probed one
microsecond or one part in a billion either side.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta

import pytest

from quant.core.canonical import sha256_hex
from quant.core.errors import ApprovalError, RiskError
from quant.core.orders import Label, OrderKind, Side, TradingMode
from quant.risk.engine import (
    CHECK_ORDER,
    MarkOutcome,
    PendingApproval,
    RiskApprovedOrder,
    RiskCheck,
    RiskState,
    RiskVerdict,
    Veto,
    verdict_record,
)
from tests.support.risk import (
    DAY,
    EQUITY,
    HALF_DAY,
    NEXT_DAY,
    PRICE,
    SHARES_PER_EQUITY,
    Desk,
    closing,
    config,
    decisions,
    desk,
    intent,
    last_fillable,
    order,
)

US = timedelta(microseconds=1)
FIVE_S = timedelta(seconds=5)


def approved(verdict: RiskVerdict) -> RiskApprovedOrder:
    assert type(verdict) is RiskApprovedOrder, verdict
    return verdict


def vetoed(verdict: RiskVerdict, check: RiskCheck, *, halts: bool = False) -> Veto:
    assert isinstance(verdict, Veto), verdict
    assert verdict.check is check, verdict
    assert verdict.halts is halts, verdict
    assert verdict.reason
    return verdict


# --- the precondition: a scheduled decision, inside its window, this session -----------------------


@pytest.mark.spec("ARCH§8.1", "ARCH§8.2", "SPEC§10")
def test_an_intent_must_be_a_scheduled_decision_inside_its_window() -> None:
    first, second = decisions()[0], decisions()[1]
    d = desk(at=first.tau)
    approved(d.check(intent(tau=first.tau)))  # at tau exactly
    vetoed(d.check(intent(tau=first.tau - timedelta(minutes=1))), RiskCheck.INTENT)  # no decision
    vetoed(d.check(intent(tau=second.tau)), RiskCheck.INTENT)  # before its time

    # The 10:46 ET decision executes in the 10:30-11:30 bar and expires at its close.
    expires = first.bar.t_close + timedelta(hours=1)
    assert expires == second.bar.t_close
    d.clock.advance_to(expires - US)
    approved(d.check(intent(tau=first.tau, leg=2)))
    d.clock.advance_to(expires)
    vetoed(d.check(intent(tau=first.tau, leg=3)), RiskCheck.INTENT)  # never sent late


@pytest.mark.spec("ARCH§8.1", "SPEC§10")
def test_an_intent_needs_its_session_started() -> None:
    d = desk(day=None)
    vetoed(d.check(intent()), RiskCheck.INTENT)
    d.engine.start_session(DAY, equity=EQUITY)
    approved(d.check(intent()))


@pytest.mark.spec("A3", "ARCH§5", "ARCH§8.1", "D9")
def test_the_closing_decision_executes_at_the_next_open_and_only_reduces() -> None:
    close = closing()
    assert close.reduce_only
    d = desk(at=close.tau + timedelta(seconds=30))
    vetoed(d.check(intent(Side.SELL, 100, tau=close.tau), position=300), RiskCheck.INTENT)

    next_open = close.executes_at
    d.clock.advance_to(next_open + timedelta(seconds=1))
    d.engine.start_session(NEXT_DAY, equity=EQUITY)
    sell = approved(d.check(intent(Side.SELL, 100, tau=close.tau), position=300))
    assert sell.quantity == 100
    assert sell.expires_at == next_open + timedelta(hours=1)  # the end of the first bar
    vetoed(d.check(intent(Side.BUY, 100, tau=close.tau, leg=2)), RiskCheck.LIMITS)


# --- 1. HALTED -------------------------------------------------------------------------------------


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "D9")
def test_1_halted_passes_only_risk_reducing_intents_clamped_to_the_position() -> None:
    d = desk()
    approved(d.check(intent(Side.BUY, 100)))  # not halted
    vetoed(d.check(intent(Side.BUY, 100, leg=2), halted=True), RiskCheck.HALTED)
    sell = approved(d.check(intent(Side.SELL, 500, leg=3), position=300, halted=True))
    assert sell.quantity == 300  # clamped: never short

    state = d.engine.state.model_copy(update={"halted": True, "halt_reason": "test"})
    restored = desk(state=state, at=d.clock.now())
    vetoed(restored.check(intent(Side.BUY, 100, leg=4)), RiskCheck.HALTED)
    assert approved(restored.check(intent(Side.SELL, 100, leg=5), position=100)).quantity == 100


# --- 2. Reconciliation -----------------------------------------------------------------------------


@pytest.mark.spec("ARCH§8.1", "SPEC§10")
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL], ids=str)
def test_2_the_snapshot_may_be_five_seconds_old_and_no_older(side: Side) -> None:
    d = desk()
    approved(d.check(intent(side, 100), position=100, snapshot_age=FIVE_S))
    vetoed(
        d.check(intent(side, 100), position=100, snapshot_age=FIVE_S + US), RiskCheck.RECONCILIATION
    )
    vetoed(d.check(intent(side, 100), position=100, snapshot_age=-US), RiskCheck.RECONCILIATION)
    assert not d.engine.state.halted  # a stale snapshot is retried, not a halt


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "C2")
@pytest.mark.parametrize(
    "mismatch",
    [
        {"position": 100, "book_position": 101},
        {"position": 100, "open_orders": (order("stop-1"),), "book_orders": ()},
        {
            "position": 100,
            "open_orders": (order("stop-1", quantity=100),),
            "book_orders": (order("stop-1", quantity=99),),
        },
        {"position": -1, "book_position": -1},
    ],
    ids=["position", "an order the book lacks", "a quantity", "a short"],
)
def test_2_a_mismatch_or_a_short_halts_trading(mismatch: dict[str, object]) -> None:
    d = desk()
    vetoed(d.check(intent(Side.SELL, 100), **mismatch), RiskCheck.RECONCILIATION, halts=True)
    assert d.engine.state.halted
    vetoed(d.check(intent(Side.BUY, 100, leg=2)), RiskCheck.HALTED)


@pytest.mark.spec("ARCH§8.1")
def test_2_open_orders_match_in_any_order() -> None:
    orders = (order("stop-1"), order("buy-1", Side.BUY, OrderKind.LIMIT, 50, PRICE))
    d = desk()
    approved(
        d.check(intent(Side.BUY, 100), position=100, open_orders=orders, book_orders=orders[::-1])
    )


# --- 3. Limits -------------------------------------------------------------------------------------


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "D9", "D13")
def test_3_gross_exposure_may_reach_one_times_equity_and_no_more() -> None:
    d = desk()
    approved(d.check(intent(Side.BUY, 400), position=600))  # exactly 1.0x
    vetoed(d.check(intent(Side.BUY, 401, leg=2), position=600), RiskCheck.LIMITS)
    # Working buy orders count as held (a fresh desk: the approval above still holds 400).
    w = desk()
    working = (order("buy-1", Side.BUY, OrderKind.LIMIT, 100, PRICE),)
    approved(w.check(intent(Side.BUY, 400), position=500, open_orders=working))
    vetoed(
        w.check(intent(Side.BUY, 401, leg=2), position=500, open_orders=working), RiskCheck.LIMITS
    )


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "D13")
@pytest.mark.parametrize("label", [Label.CHOP, Label.STRESS], ids=str)
def test_3_each_label_cap_holds_at_its_boundary(label: Label) -> None:
    cap = round(config().label_caps.cap(label) * SHARES_PER_EQUITY)  # 500 and 250 shares
    d = desk()
    approved(d.check(intent(Side.BUY, cap - 100, label=label), position=100))
    vetoed(d.check(intent(Side.BUY, cap - 99, label=label, leg=2), position=100), RiskCheck.LIMITS)


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "D13", "SPEC§8")
@pytest.mark.parametrize(
    "kwargs",
    [{"label": Label.CRASH}, {"label": None}, {"label": Label.CALM_UP, "uncertain": True}],
    ids=["CRASH", "no active label", "uncertain"],
)
def test_3_crash_no_label_and_uncertainty_cap_buys_at_zero(kwargs: dict[str, object]) -> None:
    d = desk()
    vetoed(d.check(intent(Side.BUY, 1, **kwargs)), RiskCheck.LIMITS)  # type: ignore[arg-type]


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "D13")
def test_3_one_order_may_buy_half_of_equity_and_no_more() -> None:
    d = desk()
    approved(d.check(intent(Side.BUY, 500)))
    vetoed(d.check(intent(Side.BUY, 501, leg=2)), RiskCheck.LIMITS)


@pytest.mark.spec("A3", "ARCH§8.1", "SPEC§10")
@pytest.mark.parametrize("day", [DAY, HALF_DAY], ids=["15:46 ET", "12:46 ET on a half day"])
def test_3_the_overnight_cap_binds_on_the_last_fillable_decision_only(day: object) -> None:
    last = last_fillable(day)  # type: ignore[arg-type]
    d = desk(at=last.tau + timedelta(seconds=30), day=day)  # type: ignore[arg-type]
    approved(d.check(intent(Side.BUY, 400, tau=last.tau), position=100))  # exactly 0.5x
    vetoed(d.check(intent(Side.BUY, 401, tau=last.tau, leg=2), position=100), RiskCheck.LIMITS)

    earlier = decisions(day)[last.bar.index - 1]  # type: ignore[arg-type]
    e = desk(at=earlier.tau + timedelta(seconds=30), day=day)  # type: ignore[arg-type]
    approved(e.check(intent(Side.BUY, 401, tau=earlier.tau), position=100))


@pytest.mark.spec("D9", "ARCH§8.1", "SPEC§10")
def test_3_sells_are_clamped_so_they_never_go_short() -> None:
    # A fresh desk for each case: an outstanding SELL approval reserves its shares.
    vetoed(desk().check(intent(Side.SELL, 100)), RiskCheck.LIMITS)  # flat: nothing to sell
    assert approved(desk().check(intent(Side.SELL, 500), position=300)).quantity == 300
    working = (order("sell-1", Side.SELL, OrderKind.LIMIT, 100, PRICE),)
    assert (
        approved(desk().check(intent(Side.SELL, 500), position=300, open_orders=working)).quantity
        == 200
    )
    stop = (order("stop-1", Side.SELL, OrderKind.STOP, 300),)  # cancelled before any sale
    assert (
        approved(desk().check(intent(Side.SELL, 500), position=300, open_orders=stop)).quantity
        == 300
    )
    working_all = (order("sell-2", Side.SELL, OrderKind.LIMIT, 300, PRICE),)
    vetoed(
        desk().check(intent(Side.SELL, 1), position=300, open_orders=working_all),
        RiskCheck.LIMITS,
    )


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "D7")
def test_risk_reducing_intents_skip_the_caps_locks_and_approval() -> None:
    d = desk(mode=TradingMode.LIVE)
    d.engine.mark(EQUITY * 0.97)  # the daily-loss lock
    sell = approved(
        d.check(
            intent(Side.SELL, 1_500, label=Label.CRASH, uncertain=True),
            position=1_500,  # 1.5x, over every cap
            equity=EQUITY * 0.97,
            healthy=False,
            drift_frozen=True,
        )
    )
    assert sell.quantity == 1_500
    assert sell.notional == 150_000.0
    assert not sell.flagged


# --- 4. Locks --------------------------------------------------------------------------------------


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "C5", "SPEC§8")
@pytest.mark.parametrize(
    "lock", [{"drift_frozen": True}, {"healthy": False}], ids=["drift freeze", "rule 6"]
)
def test_4_a_drift_freeze_and_rule_6_veto_entries_only(lock: dict[str, bool]) -> None:
    d = desk()
    vetoed(d.check(intent(Side.BUY, 100), **lock), RiskCheck.LOCKS)  # type: ignore[arg-type]
    approved(d.check(intent(Side.SELL, 100, leg=2), position=100, **lock))  # type: ignore[arg-type]
    approved(d.check(intent(Side.BUY, 100, leg=3)))


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "C8", "D13")
def test_4_the_daily_loss_lock_sets_at_exactly_two_percent() -> None:
    inside = desk()
    outcome = inside.engine.mark(98_000.01)
    assert not outcome.locked
    assert not outcome.flatten
    approved(inside.check(intent(Side.BUY, 100)))

    d = desk()
    outcome = d.engine.mark(98_000.0)
    assert outcome == MarkOutcome(
        equity=98_000.0, day_loss=0.02, drawdown=0.02, locked=True, flatten=True, kill=False
    )
    assert d.engine.state.daily_loss_locked
    vetoed(d.check(intent(Side.BUY, 100), equity=98_000.0), RiskCheck.LOCKS)
    approved(d.check(intent(Side.SELL, 100, leg=2), position=100, equity=98_000.0))
    again = d.engine.mark(97_000.0)
    assert again.locked
    assert not again.flatten

    d.clock.advance_to(decisions(NEXT_DAY)[0].tau + timedelta(seconds=30))
    d.engine.start_session(NEXT_DAY, equity=97_000.0)  # the lock lasts until the next session
    assert not d.engine.state.daily_loss_locked
    approved(d.check(intent(Side.BUY, 100, tau=decisions(NEXT_DAY)[0].tau), equity=97_000.0))


@pytest.mark.spec("ARCH§8.3", "SPEC§10", "C8", "D13")
def test_the_drawdown_trigger_fires_at_exactly_ten_percent_from_the_peak() -> None:
    d = desk()
    d.engine.mark(105_000.0)  # a new peak
    d.clock.advance_to(decisions(NEXT_DAY)[0].tau)
    d.engine.start_session(NEXT_DAY, equity=95_000.0)
    assert d.engine.state.peak_equity == 105_000.0
    inside = d.engine.mark(94_500.01)
    assert not inside.kill
    assert not d.engine.state.halted
    outcome = d.engine.mark(94_500.0)  # exactly 10% below 105,000, but only 0.53% today
    assert outcome.kill
    assert not outcome.locked
    assert outcome.drawdown == pytest.approx(0.10)
    state = d.engine.state
    assert state.halted
    assert state.killed
    assert state.halt_reason
    assert not d.engine.mark(90_000.0).kill  # it fires once
    vetoed(d.check(intent(Side.BUY, 1, tau=decisions(NEXT_DAY)[0].tau)), RiskCheck.HALTED)


@pytest.mark.spec("SPEC§10", "C8")
def test_sessions_and_marks_refuse_misuse() -> None:
    d = desk(day=None)
    with pytest.raises(RiskError):
        d.engine.mark(EQUITY)  # no session yet
    with pytest.raises(RiskError):
        d.engine.start_session(DAY.replace(day=10), equity=EQUITY)  # a Saturday
    with pytest.raises(RiskError):
        d.engine.start_session(DAY, equity=float("nan"))
    d.engine.start_session(DAY, equity=EQUITY)
    with pytest.raises(RiskError):
        d.engine.start_session(DAY, equity=EQUITY)  # not later than the current one
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(RiskError):
            d.engine.mark(bad)


# --- 5. Rate (C3) ----------------------------------------------------------------------------------


@pytest.mark.spec("C3", "ARCH§8.1", "SPEC§10")
def test_5_the_twenty_first_intent_in_a_session_halts_trading() -> None:
    d = desk()
    for leg in range(1, 21):
        approved(d.check(intent(Side.BUY, 1, leg=leg)))
    approved(d.check(intent(Side.BUY, 1, leg=7)))  # a re-quote of a counted intent
    assert len(d.engine.state.intents) == 20
    assert not d.engine.state.halted

    vetoed(d.check(intent(Side.BUY, 1, leg=21)), RiskCheck.RATE, halts=True)
    state = d.engine.state
    assert state.halted
    assert len(state.intents) == 21
    vetoed(d.check(intent(Side.BUY, 1, leg=22)), RiskCheck.HALTED)
    approved(d.check(intent(Side.SELL, 1, leg=23), position=20))  # reductions still pass

    d.clock.advance_to(decisions(NEXT_DAY)[0].tau)
    d.engine.start_session(NEXT_DAY, equity=EQUITY)
    assert d.engine.state.intents == ()
    assert d.engine.state.halted


@pytest.mark.spec("C3", "ARCH§8.1")
def test_5_a_risk_reducing_twenty_first_intent_halts_and_still_passes() -> None:
    d = desk()
    for leg in range(1, 21):
        approved(d.check(intent(Side.SELL, 1, leg=leg), position=100))
    assert approved(d.check(intent(Side.SELL, 1, leg=21), position=100)).quantity == 1
    assert d.engine.state.halted


@pytest.mark.spec("C3", "ARCH§4.4", "SPEC§10")
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL], ids=str)
def test_the_sixtieth_raw_submission_halts_trading(side: Side) -> None:
    d = desk()
    order_intent = intent(side, 1)
    for _ in range(100):  # cancels and flattens are not submissions
        d.engine.spend(d.engine.approve_cancel("eng1-x", reason="re-quote"))
    # One intent re-quoted 60 times: each re-check replaces the approval before it, so each
    # attempt is spent before the next is checked.
    for _ in range(59):
        d.engine.spend(approved(d.check(order_intent, position=100)))
    assert d.engine.state.submissions == 59
    assert not d.engine.state.halted
    last = approved(d.check(order_intent, position=100))
    if side is Side.BUY:
        with pytest.raises(ApprovalError):
            d.engine.spend(last)
        assert d.engine.state.submissions == 59
    else:
        assert d.engine.spend(last) is last
        assert d.engine.state.submissions == 60
    assert d.engine.state.halted


@pytest.mark.spec("ARCH§8.1", "ARCH§8.3", "I4")
def test_a_halt_stops_buys_already_approved_but_never_reductions() -> None:
    d = desk()
    buy = approved(d.check(intent(Side.BUY, 100)))
    sell = approved(d.check(intent(Side.SELL, 100, leg=2), position=100))
    vetoed(
        d.check(intent(Side.SELL, 100, leg=3), position=100, book_position=99),
        RiskCheck.RECONCILIATION,
        halts=True,
    )
    with pytest.raises(ApprovalError):
        d.engine.spend(buy)
    assert d.engine.spend(sell) is sell
    cancel = d.engine.approve_cancel("eng1-y", reason="kill")
    flatten = d.engine.approve_flatten(reason="kill")
    assert d.engine.spend(cancel) is cancel
    assert d.engine.spend(flatten) is flatten


# --- 6. Price --------------------------------------------------------------------------------------


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "C8")
def test_6_the_quote_may_be_five_seconds_old_and_no_older() -> None:
    d = desk()
    approved(d.check(intent(Side.BUY, 100), quote_age=FIVE_S))
    vetoed(d.check(intent(Side.BUY, 100), quote_age=FIVE_S + US), RiskCheck.PRICE)
    vetoed(d.check(intent(Side.BUY, 100), quote_age=-US), RiskCheck.PRICE)
    vetoed(d.check(intent(Side.BUY, 100), bid=PRICE + 0.01), RiskCheck.PRICE)  # crossed


@pytest.mark.spec("ARCH§8.1", "SPEC§10")
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL], ids=str)
@pytest.mark.parametrize("direction", [1.0, -1.0], ids=["above", "below"])
def test_6_the_limit_sits_within_ten_basis_points_of_its_side_of_the_quote(
    side: Side, direction: float
) -> None:
    band = PRICE * config().collar_bp / 10_000
    inside = PRICE + direction * band * (1 - 1e-9)
    outside = PRICE + direction * band * (1 + 1e-9)
    stop = PRICE * 0.9  # below either limit
    d = desk()
    approved(d.check(intent(side, 100, stop=stop), position=100, limit=inside))
    vetoed(d.check(intent(side, 100, stop=stop), position=100, limit=outside), RiskCheck.PRICE)


@pytest.mark.spec("ARCH§8.1", "SPEC§10", "SPEC§7")
@pytest.mark.parametrize(
    "stop", [None, PRICE, PRICE + 1.0], ids=["no stop", "at the limit", "above"]
)
def test_6_every_entry_carries_a_stop_below_its_limit(stop: float | None) -> None:
    d = desk()
    vetoed(d.check(intent(Side.BUY, 100, stop=stop)), RiskCheck.PRICE)
    approved(d.check(intent(Side.BUY, 100, stop=PRICE - 0.01)))


# --- 7. Approval (D7) ------------------------------------------------------------------------------


@pytest.mark.spec("D7", "ARCH§8.1", "ARCH§8.4", "SPEC§10")
def test_7_live_orders_above_25000_wait_for_approval() -> None:
    d = desk(mode=TradingMode.LIVE)
    at_threshold = approved(d.check(intent(Side.BUY, 250)))  # exactly $25,000
    assert not at_threshold.flagged
    pending = d.check(intent(Side.BUY, 251, leg=2))
    assert type(pending) is PendingApproval
    assert (pending.quantity, pending.notional) == (251, 25_100.0)
    assert pending.expires_at == at_threshold.expires_at  # the end of the bar
    with pytest.raises(ApprovalError):
        d.engine.spend(pending)  # not an approval

    order_ = d.engine.approve_pending(pending, d.ctx())
    assert type(order_) is RiskApprovedOrder
    assert order_.flagged
    assert order_.quantity == 251
    assert d.engine.spend(order_) is order_
    with pytest.raises(ApprovalError):
        d.engine.approve_pending(pending, d.ctx())  # approved once
    assert len(d.engine.state.intents) == 2  # approving didn't count the intent again


@pytest.mark.spec("D7", "ARCH§8.1")
def test_7_paper_flags_large_orders_and_never_blocks_them() -> None:
    d = desk(mode=TradingMode.PAPER)
    assert not approved(d.check(intent(Side.BUY, 250))).flagged
    flagged = approved(d.check(intent(Side.BUY, 251, leg=2)))
    assert flagged.flagged
    assert flagged.quantity == 251
    assert d.engine.spend(flagged) is flagged


@pytest.mark.spec("D7", "ARCH§8.4", "I4")
def test_7_a_pending_approval_expires_rechecks_and_cannot_be_forged() -> None:
    d = desk(mode=TradingMode.LIVE)
    pending = d.check(intent(Side.BUY, 300))
    assert type(pending) is PendingApproval
    with pytest.raises(ApprovalError):
        d.engine.approve_pending(replace(pending), d.ctx())  # a copy was never issued
    other = desk(mode=TradingMode.LIVE).check(intent(Side.BUY, 300))
    assert type(other) is PendingApproval
    with pytest.raises(ApprovalError):
        d.engine.approve_pending(other, d.ctx())  # another engine's
    vetoed(d.engine.approve_pending(pending, d.ctx(halted=True)), RiskCheck.HALTED)
    d.clock.advance_to(pending.expires_at)
    with pytest.raises(ApprovalError):
        d.engine.approve_pending(pending, d.ctx())


# --- the order of the checks -----------------------------------------------------------------------

Case = Callable[[], tuple[RiskVerdict, RiskCheck]]


def _twenty_intents(d: Desk) -> None:
    for leg in range(1, 21):
        approved(d.check(intent(Side.BUY, 1, leg=leg)))


def _intent_before_halted() -> tuple[RiskVerdict, RiskCheck]:
    d = desk()
    late = intent(tau=decisions()[0].tau - timedelta(minutes=1))
    return d.check(late, halted=True), RiskCheck.INTENT


def _halted_before_reconciliation() -> tuple[RiskVerdict, RiskCheck]:
    d = desk()
    return d.check(intent(), halted=True, snapshot_age=FIVE_S + US), RiskCheck.HALTED


def _reconciliation_before_limits() -> tuple[RiskVerdict, RiskCheck]:
    d = desk()
    return d.check(intent(Side.BUY, 501), snapshot_age=FIVE_S + US), RiskCheck.RECONCILIATION


def _limits_before_locks() -> tuple[RiskVerdict, RiskCheck]:
    d = desk()
    return d.check(intent(Side.BUY, 501), drift_frozen=True), RiskCheck.LIMITS


def _locks_before_rate() -> tuple[RiskVerdict, RiskCheck]:
    d = desk()
    _twenty_intents(d)
    verdict = d.check(intent(Side.BUY, 1, leg=21), healthy=False)
    assert len(d.engine.state.intents) == 20
    assert not d.engine.state.halted
    return verdict, RiskCheck.LOCKS


def _rate_before_price() -> tuple[RiskVerdict, RiskCheck]:
    d = desk()
    _twenty_intents(d)
    return d.check(intent(Side.BUY, 1, leg=21), quote_age=FIVE_S + US), RiskCheck.RATE


def _price_before_approval() -> tuple[RiskVerdict, RiskCheck]:
    d = desk(mode=TradingMode.LIVE)
    return d.check(intent(Side.BUY, 300), quote_age=FIVE_S + US), RiskCheck.PRICE


ORDER_CASES: dict[str, Case] = {
    "intent, then halted": _intent_before_halted,
    "halted, then reconciliation": _halted_before_reconciliation,
    "reconciliation, then limits": _reconciliation_before_limits,
    "limits, then locks": _limits_before_locks,
    "locks, then rate": _locks_before_rate,
    "rate, then price": _rate_before_price,
    "price, then approval": _price_before_approval,
}


@pytest.mark.spec("ARCH§8.1")
@pytest.mark.parametrize("case", ORDER_CASES.values(), ids=ORDER_CASES.keys())
def test_the_first_failing_check_decides(case: Case) -> None:
    assert tuple(RiskCheck) == CHECK_ORDER
    verdict, first = case()
    vetoed(verdict, first, halts=first is RiskCheck.RATE)


# --- records ---------------------------------------------------------------------------------------


@pytest.mark.spec("ARCH§4.3", "ARCH§14", "ARCH§8.1")
def test_verdict_records_are_canonical_and_identical_across_runs() -> None:
    def run() -> list[str]:
        d = desk(mode=TradingMode.LIVE)
        verdicts = [
            d.check(intent(Side.BUY, 100)),
            d.check(intent(Side.BUY, 300, leg=2)),
            d.check(intent(Side.BUY, 501, leg=3)),
        ]
        return [sha256_hex(verdict_record(v)) for v in verdicts]

    first = run()
    assert first == run()
    d = desk(mode=TradingMode.LIVE)
    kinds = [
        verdict_record(d.check(intent(Side.BUY, 100)))["verdict"],
        verdict_record(d.check(intent(Side.BUY, 300, leg=2)))["verdict"],
        verdict_record(d.check(intent(Side.BUY, 501, leg=3)))["verdict"],
    ]
    assert kinds == ["approved", "pending", "veto"]


@pytest.mark.spec("ARCH§4.3", "SPEC§10")
def test_state_restores_and_continues_the_count() -> None:
    d = desk()
    for leg in range(1, 6):
        approved(d.check(intent(Side.BUY, 1, leg=leg)))
    state = d.engine.state
    assert isinstance(state, RiskState)
    assert len(state.intents) == 5
    restored = desk(state=RiskState.model_validate(state.model_dump()), at=d.clock.now())
    for leg in range(6, 21):
        approved(restored.check(intent(Side.BUY, 1, leg=leg)))
    vetoed(restored.check(intent(Side.BUY, 1, leg=21)), RiskCheck.RATE, halts=True)
