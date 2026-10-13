"""B9a review 2: every halt is counted, signalled, persisted and resumable (spec §10, C3, D7).

Round 2 of the B9a review found the engine's halts and counters leaking:

- R6: an intent vetoed before check 5 was never counted, so 21 vetoed intents didn't halt;
- C6: a halt on a pass was silent (the 21st intent as a SELL, the 60th submission);
- C7: `spend`, `mark` and `check` changed RiskState with no way to persist it before returning;
- C8: nothing cleared a halt or the drawdown trigger, and no contract said what a resume resets;
- C9: the 60th raw submission was refused instead of sent, so at most 59 BUYs went out;
- C10: a later halt overwrote the reason of the first, a drawdown kill's included;
- C11: approving a pending order re-priced it without bound;
- cleanup: the engine kept every approval it ever issued.
"""

from __future__ import annotations

import gc
import weakref
from datetime import timedelta

import pytest

from quant.core.errors import RiskError, SubmissionLimitError
from quant.core.orders import Side, TradingMode
from quant.risk.engine import (
    PendingApproval,
    RiskApprovedOrder,
    RiskCheck,
    RiskEngine,
    RiskState,
    RiskVerdict,
    Veto,
    verdict_record,
)
from tests.support.risk import CAL, DAY, EQUITY, PRICE, config, decisions, desk, intent

US = timedelta(microseconds=1)
FIVE_S = timedelta(seconds=5)


def approved(verdict: RiskVerdict) -> RiskApprovedOrder:
    assert type(verdict) is RiskApprovedOrder, verdict
    return verdict


def vetoed(verdict: RiskVerdict, check: RiskCheck, *, halts: bool = False) -> Veto:
    assert isinstance(verdict, Veto), verdict
    assert verdict.check is check, verdict
    assert verdict.halts is halts, verdict
    return verdict


# --- R6: every intent counts toward C3's limit, vetoed or not -------------------------------------


@pytest.mark.spec("C3", "SPEC§10", "ARCH§8.1")
@pytest.mark.parametrize(
    ("quantity", "context", "check"),
    [
        (501, {}, RiskCheck.LIMITS),  # over the single-order cap
        (1, {"snapshot_age": FIVE_S + US}, RiskCheck.RECONCILIATION),  # a stale snapshot
    ],
    ids=["limits", "reconciliation"],
)
def test_vetoed_intents_count_and_the_twenty_first_halts(
    quantity: int, context: dict[str, object], check: RiskCheck
) -> None:
    d = desk()
    for leg in range(1, 21):
        vetoed(d.check(intent(Side.BUY, quantity, leg=leg), **context), check)
    assert len(d.engine.state.intents) == 20
    assert not d.engine.state.halted

    vetoed(d.check(intent(Side.BUY, quantity, leg=21), **context), check, halts=True)
    state = d.engine.state
    assert len(state.intents) == 21
    assert state.halted
    assert state.halt_reason == "daily intent limit exceeded"
    vetoed(d.check(intent(Side.BUY, 1, leg=22)), RiskCheck.HALTED)  # halted, so no second halt


# --- C6: a halt on a pass is signalled ------------------------------------------------------------


@pytest.mark.spec("C3", "ARCH§8.1")
def test_a_twenty_first_intent_that_passes_says_it_halted() -> None:
    d = desk()
    for leg in range(1, 21):
        assert approved(d.check(intent(Side.SELL, 1, leg=leg), position=100)).halts is False
    last = approved(d.check(intent(Side.SELL, 1, leg=21), position=100))
    assert last.halts is True
    assert last.quantity == 1
    assert d.engine.state.halted
    assert verdict_record(last)["halts"] is True


@pytest.mark.spec("C3", "ARCH§4.4", "SPEC§10")
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL], ids=str)
def test_the_sixtieth_submission_publishes_its_halt_before_spend_returns(side: Side) -> None:
    seen: list[RiskState] = []
    d = desk(on_change=seen.append)
    order_intent = intent(side, 1)
    for _ in range(59):
        d.engine.spend(approved(d.check(order_intent, position=100)))
    assert not seen[-1].halted
    last = approved(d.check(order_intent, position=100))
    assert d.engine.spend(last) is last
    assert seen[-1].submissions == 60
    assert seen[-1].halted
    assert seen[-1].halt_reason == "raw submission limit reached"


@pytest.mark.spec("C3", "ARCH§4.4", "ARCH§8.3")
def test_a_buy_past_the_submission_limit_is_refused_with_its_own_error() -> None:
    d = desk()
    sell = intent(Side.SELL, 1)
    for _ in range(60):
        d.engine.spend(approved(d.check(sell, position=100)))
    assert d.engine.state.halted
    d.engine.resume(equity=EQUITY)  # a resume keeps the session's counts
    assert not d.engine.state.halted

    buy = approved(d.check(intent(Side.BUY, 1, leg=2), position=100))
    with pytest.raises(SubmissionLimitError):
        d.engine.spend(buy)
    state = d.engine.state
    assert state.submissions == 60
    assert state.halted
    assert state.halt_reason == "raw submission limit reached"
    # A reduction is never refused here: it is counted, and keeps trading halted.
    reduce = approved(d.check(intent(Side.SELL, 1, leg=3), position=100))
    assert d.engine.spend(reduce) is reduce
    assert d.engine.state.submissions == 61


# --- C7: every change to RiskState is published before the call returns ----------------------------


@pytest.mark.spec("ARCH§4.3", "SPEC§10", "C3", "C8")
def test_every_state_change_is_published_before_the_call_returns() -> None:
    seen: list[RiskState] = []
    d = desk(on_change=seen.append)
    assert seen[-1] == d.engine.state
    assert seen[-1].session == DAY

    buy = approved(d.check(intent(Side.BUY, 100)))
    assert seen[-1] == d.engine.state
    assert seen[-1].intents == (f"{buy.intent.decision_id}#1",)
    d.engine.spend(buy)
    assert seen[-1].submissions == 1
    count = len(seen)
    d.engine.release(buy)  # already spent: nothing changes, so nothing is published
    assert len(seen) == count
    d.engine.approve_cancel("eng1-x", reason="re-quote")  # numbers an id
    assert seen[-1] == d.engine.state
    d.engine.mark(98_000.0)
    assert seen[-1].daily_loss_locked
    d.engine.mark(89_000.0)
    assert seen[-1].killed
    assert seen[-1].halted
    assert seen[-1] == d.engine.state

    restored = RiskEngine(
        config(), clock=d.clock, mode=TradingMode.PAPER, calendar=CAL, state=seen[-1]
    )
    assert restored.state == d.engine.state


@pytest.mark.spec("ARCH§4.3", "ARCH§4.4", "C3")
def test_a_state_that_cannot_be_persisted_stops_the_call() -> None:
    def persist(state: RiskState) -> None:
        if state.submissions or len(state.intents) > 1:
            raise OSError("disk full")

    d = desk(on_change=persist)
    buy = approved(d.check(intent(Side.BUY, 100)))
    with pytest.raises(OSError, match="disk full"):
        d.engine.spend(buy)  # the adapter sends nothing unless spend returns
    d2 = desk(on_change=persist)
    approved(d2.check(intent(Side.BUY, 100)))
    with pytest.raises(OSError, match="disk full"):
        d2.check(intent(Side.BUY, 100, leg=2))  # neither an approval nor a Veto


# --- C8: what a resume resets ----------------------------------------------------------------------


@pytest.mark.spec("ARCH§8.3", "SPEC§10")
def test_a_resume_clears_the_halt_and_keeps_the_day_s_locks_and_counts() -> None:
    d = desk()
    d.engine.mark(98_000.0)  # the daily-loss lock
    vetoed(
        d.check(intent(Side.SELL, 1), position=100, book_position=99, equity=98_000.0),
        RiskCheck.RECONCILIATION,
        halts=True,
    )
    assert d.engine.state.halted
    before = d.engine.state
    d.engine.resume(equity=98_000.0)
    state = d.engine.state
    assert not state.halted
    assert state.halt_reason is None
    assert state.daily_loss_locked  # until the next session, as always
    assert state.intents == before.intents
    assert state.submissions == before.submissions
    vetoed(d.check(intent(Side.BUY, 1, leg=2), equity=98_000.0), RiskCheck.LOCKS)

    clean = desk()
    vetoed(
        clean.check(intent(Side.SELL, 1), position=100, book_position=99),
        RiskCheck.RECONCILIATION,
        halts=True,
    )
    clean.engine.resume(equity=EQUITY)
    approved(clean.check(intent(Side.BUY, 1, leg=2)))


@pytest.mark.spec("ARCH§8.3", "SPEC§10", "C8")
def test_a_resume_after_the_drawdown_trigger_keeps_the_peak() -> None:
    d = desk()
    d.engine.mark(105_000.0)
    assert d.engine.mark(94_500.0).kill
    with pytest.raises(RiskError):
        d.engine.resume(equity=94_500.0)  # still 10% below the peak: it would fire at once
    assert d.engine.state.killed
    assert d.engine.state.halted
    for bad in (0.0, float("nan")):
        with pytest.raises(RiskError):
            d.engine.resume(equity=bad)

    d.engine.resume(equity=94_500.01)
    state = d.engine.state
    assert not state.killed
    assert not state.halted
    assert state.peak_equity == 105_000.0  # resetting the peak is the owner's call, not B9's
    assert not d.engine.mark(94_500.01).kill
    assert d.engine.mark(94_500.0).kill  # the trigger is armed again

    with pytest.raises(RiskError):
        desk(day=None).engine.resume(equity=EQUITY)  # no session yet


# --- C10: the first halt's reason survives ---------------------------------------------------------


@pytest.mark.spec("ARCH§8.3", "SPEC§10")
def test_a_later_halt_keeps_the_first_reason() -> None:
    d = desk()
    d.engine.mark(105_000.0)
    assert d.engine.mark(94_500.0).kill
    first = d.engine.state.halt_reason
    assert first == "maximum drawdown reached"
    vetoed(
        d.check(intent(Side.SELL, 1), position=100, book_position=99, equity=94_500.0),
        RiskCheck.RECONCILIATION,
        halts=True,
    )
    assert d.engine.state.halt_reason == first


# --- C11: approving a pending order may not re-price it past the collar ------------------------------


@pytest.mark.spec("D7", "ARCH§8.4", "SPEC§10")
def test_approving_a_pending_order_refuses_a_price_moved_past_the_collar() -> None:
    d = desk(mode=TradingMode.LIVE)
    pending = d.check(intent(Side.BUY, 251))
    assert type(pending) is PendingApproval
    assert pending.limit_price == PRICE
    moved = 100.2  # 20 bp above the price you were asked about, inside the new quote's collar
    vetoed(
        d.engine.approve_pending(pending, d.ctx(bid=moved, ask=moved, limit=moved)),
        RiskCheck.APPROVAL,
    )
    cheaper = approved(d.engine.approve_pending(pending, d.ctx(bid=99.0, ask=99.0, limit=99.0)))
    assert cheaper.flagged

    other = d.check(intent(Side.BUY, 251, leg=2))
    assert type(other) is PendingApproval
    edge = 100.1  # exactly the collar above
    assert approved(d.engine.approve_pending(other, d.ctx(bid=edge, ask=edge, limit=edge))).flagged


# --- cleanup: the engine lets go of approvals once they expire --------------------------------------


@pytest.mark.spec("ARCH§8.1", "ARCH§4.4")
def test_the_engine_lets_go_of_approvals_once_they_expire() -> None:
    d = desk(mode=TradingMode.LIVE)
    refs: list[weakref.ref[object]] = []

    def issue() -> None:
        spent = intent(Side.BUY, 100)
        d.engine.spend(approved(d.check(spent)))
        replaced = intent(Side.SELL, 10, leg=2)
        approved(d.check(replaced, position=100))
        replacement = intent(Side.SELL, 10, leg=2)
        approved(d.check(replacement, position=100))  # revokes the one it replaces
        waiting = intent(Side.BUY, 300, leg=3)
        assert type(d.check(waiting)) is PendingApproval
        released = intent(Side.BUY, 10, leg=4)
        d.engine.release(approved(d.check(released)))
        refs.extend(weakref.ref(i) for i in (spent, replaced, replacement, waiting, released))

    issue()
    later = decisions()[2]
    d.clock.advance_to(later.tau + timedelta(seconds=30))  # every approval above has expired
    approved(d.check(intent(Side.BUY, 1, tau=later.tau), position=100))
    gc.collect()
    assert [ref() for ref in refs] == [None] * len(refs)
