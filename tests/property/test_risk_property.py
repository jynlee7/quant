"""B9: no sequence of intents, approvals and fills produces a short or breaches a cap (spec §10,
D9, A3).

Hypothesis draws a session of intents: any side and size, any label, uncertainty and health,
at any of the session's decisions in order, including the closing one that executes at the next
open. Some are re-quotes, checked again under an earlier decision and leg. Each intent goes
through `RiskEngine.check`, and approvals pile up unspent: after each check the oldest few are
spent, each filled in part and the rest cancelled, and the position moves by the fill. Whatever
the intents ask for, every approval the engine issues must leave, counting the approvals still
outstanding (B9a review, R1 and C3):

- the SELLs together selling no more shares than are held, so the position never goes below 0
  (D9);
- the BUYs together holding no more than the smallest cap in force: 1.0x gross, the label's cap
  (0 when uncertain or with no label) and, on the last fillable decision, 0.5x overnight (A3);
- no BUY of more than 0.5x equity in one order;
- no BUY on the closing decision, which can only reduce, or on an unhealthy bar;
- nothing at all on a run whose intents exceeded C3's limit, except reductions.

An approval outstanding here is one issued, not yet spent, not expired, and not replaced by a
re-quote of its leg or dropped because a later check saw an unhealthy bar (a BUY). In paper,
PendingApproval never appears (D7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant.core.errors import ApprovalError
from quant.core.orders import Label, Side, TradingMode
from quant.risk.engine import PendingApproval, RiskApprovedOrder, Veto
from tests.support.risk import EQUITY, NEXT_DAY, config, decisions, desk, intent

LAST = len(decisions()) - 1  # the closing decision


@dataclass(frozen=True)
class Step:
    decision: int  # an index into DAY's decisions; steps run in order of it
    side: Side
    quantity: int
    label: Label | None
    uncertain: bool
    healthy: bool
    price: float
    requote: bool  # check the decision's last leg again instead of a new one
    spend: int  # how many of the oldest outstanding approvals to spend after this check
    fill: float  # the share of a spent order that fills


steps = st.lists(
    st.builds(
        Step,
        decision=st.one_of(st.just(0), st.integers(0, LAST)),  # crowd some on one decision
        side=st.sampled_from(Side),
        quantity=st.one_of(st.integers(1, 300), st.integers(1, 1_500)),
        label=st.sampled_from([*Label, None]),
        uncertain=st.sampled_from([False, False, False, True]),  # mostly approvable
        healthy=st.sampled_from([True, True, True, False]),
        price=st.floats(50.0, 200.0, allow_nan=False, allow_infinity=False),
        requote=st.booleans(),
        spend=st.sampled_from([0, 0, 1, 3]),  # mostly leave approvals outstanding
        fill=st.sampled_from([0.0, 0.5, 1.0]),
    ),
    max_size=30,
)


def exact(value: float) -> Decimal:
    return Decimal(repr(value))


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)
@pytest.mark.spec("SPEC§10", "D9", "A3", "C3", "D7", "ARCH§8.1", "I4")
@given(st.integers(0, 400), steps)
def test_no_sequence_of_intents_and_fills_goes_short_or_breaches_a_cap(
    held_at_start: int, drawn: list[Step]
) -> None:
    limits = config()
    d = desk(mode=TradingMode.PAPER)
    position = held_at_start
    legs: dict[int, int] = {}
    outstanding: list[tuple[tuple[int, int], RiskApprovedOrder]] = []
    session_rolled = False

    def spend_oldest(count: int, fill: float) -> None:
        nonlocal position
        for _ in range(min(count, len(outstanding))):
            _, approval = outstanding.pop(0)
            try:
                d.engine.spend(approval)
            except ApprovalError:
                continue  # expired, halted or locked since it was issued
            filled = int(approval.quantity * fill)
            position += filled if approval.intent.side is Side.BUY else -filled
            assert position >= 0

    for step in sorted(drawn, key=lambda s: s.decision):
        decision = decisions()[step.decision]
        if step.decision == LAST and not session_rolled:
            d.clock.advance_to(decision.executes_at + timedelta(seconds=1))
            d.engine.start_session(NEXT_DAY, equity=EQUITY)
            session_rolled = True
        else:  # a second on, so each snapshot postdates every earlier spend
            start = decision.tau if not session_rolled else decision.executes_at
            d.clock.advance_to(max(d.clock.now(), start) + timedelta(seconds=1))
        if not (step.requote and step.decision in legs):
            legs[step.decision] = legs.get(step.decision, 0) + 1
        key = (step.decision, legs[step.decision])
        now = d.clock.now()
        outstanding[:] = [
            (k, a)
            for k, a in outstanding
            if k != key and now < a.expires_at and (step.healthy or a.intent.side is Side.SELL)
        ]
        price = round(step.price, 2)
        verdict = d.check(
            intent(
                step.side,
                step.quantity,
                tau=decision.tau,
                leg=key[1],
                label=step.label,
                uncertain=step.uncertain,
                stop=price * 0.95,
            ),
            position=position,
            bid=price,
            ask=price,
            limit=price,
            healthy=step.healthy,
        )
        assert not isinstance(verdict, PendingApproval)  # paper never blocks
        if isinstance(verdict, RiskApprovedOrder):
            same_side = sum(a.quantity for _, a in outstanding if a.intent.side is step.side)
            if step.side is Side.BUY:
                cap = min(
                    limits.max_gross_exposure,
                    limits.uncertain_cap
                    if step.uncertain or step.label is None
                    else limits.label_caps.cap(step.label),
                )
                if decision.last_fillable:
                    cap = min(cap, limits.overnight_cap)
                held = position + same_side + verdict.quantity
                assert not decision.reduce_only
                assert step.healthy
                assert held * exact(price) <= exact(cap) * exact(EQUITY)
                assert verdict.quantity * exact(price) <= (
                    exact(limits.single_order_cap) * exact(EQUITY)
                )
                assert verdict.quantity == step.quantity  # a BUY is never resized
                assert not d.engine.state.halted
            else:
                assert 0 < same_side + verdict.quantity <= position
            outstanding.append((key, verdict))
        else:
            assert isinstance(verdict, Veto)
        spend_oldest(step.spend, step.fill)
    spend_oldest(len(outstanding), 1.0)
