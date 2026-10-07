"""B9: no sequence of intents and fills produces a short or breaches a cap (spec §10, D9, A3).

Hypothesis draws a session of intents: any side and size, any label, uncertainty and health,
at any of the session's decisions in order, including the closing one that executes at the next
open. Each intent goes through `RiskEngine.check`; an approval is spent and then filled in part,
the rest cancelled, and the position moves by the fill. Whatever the intents ask for, the engine
must never approve:

- a SELL of more shares than are held, so the position never goes below 0 (D9);
- a BUY that leaves more held than the smallest cap in force: 1.0x gross, the label's cap (0
  when uncertain or with no label) and, on the last fillable decision, 0.5x overnight (A3);
- a BUY of more than 0.5x equity in one order;
- a BUY on the closing decision, which can only reduce;
- anything at all on a run whose intents exceeded C3's limit, except reductions.

In paper, PendingApproval never appears (D7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant.core.orders import Label, Side, TradingMode
from quant.risk.engine import PendingApproval, RiskApprovedOrder, Veto
from tests.support.risk import EQUITY, NEXT_DAY, config, decisions, desk, intent

RED = pytest.mark.xfail(strict=True, reason="red:B9", raises=NotImplementedError)
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
    fill: float  # the share of an approved order that fills


steps = st.lists(
    st.builds(
        Step,
        decision=st.integers(0, LAST),
        side=st.sampled_from(Side),
        quantity=st.integers(1, 1_500),
        label=st.sampled_from([*Label, None]),
        uncertain=st.booleans(),
        healthy=st.booleans(),
        price=st.floats(50.0, 200.0, allow_nan=False, allow_infinity=False),
        fill=st.sampled_from([0.0, 0.5, 1.0]),
    ),
    max_size=30,
)


@RED
@pytest.mark.spec("SPEC§10", "D9", "A3", "C3", "D7", "ARCH§8.1", "I4")
@given(steps)
def test_no_sequence_of_intents_and_fills_goes_short_or_breaches_a_cap(drawn: list[Step]) -> None:
    limits = config()
    d = desk(mode=TradingMode.PAPER)
    position = 0
    legs: dict[int, int] = {}
    session_rolled = False
    for step in sorted(drawn, key=lambda s: s.decision):
        decision = decisions()[step.decision]
        if step.decision == LAST and not session_rolled:
            d.clock.advance_to(decision.executes_at + timedelta(seconds=1))
            d.engine.start_session(NEXT_DAY, equity=EQUITY)
            session_rolled = True
        elif not session_rolled:
            d.clock.advance_to(max(d.clock.now(), decision.tau + timedelta(seconds=1)))
        legs[step.decision] = legs.get(step.decision, 0) + 1
        price = round(step.price, 2)
        verdict = d.check(
            intent(
                step.side,
                step.quantity,
                tau=decision.tau,
                leg=legs[step.decision],
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
        if isinstance(verdict, Veto):
            continue
        assert type(verdict) is RiskApprovedOrder
        held_value = (position + verdict.quantity) * price
        if step.side is Side.BUY:
            cap = min(
                limits.max_gross_exposure,
                limits.uncertain_cap
                if step.uncertain or step.label is None
                else limits.label_caps.cap(step.label),
            )
            if decision.last_fillable:
                cap = min(cap, limits.overnight_cap)
            assert not decision.reduce_only
            assert step.healthy
            assert held_value <= cap * EQUITY
            assert verdict.quantity * price <= limits.single_order_cap * EQUITY
            assert verdict.quantity == step.quantity  # a BUY is never resized
            assert not d.engine.state.halted
        else:
            assert 0 < verdict.quantity <= position
        d.engine.spend(verdict)
        filled = int(verdict.quantity * step.fill)
        position += filled if step.side is Side.BUY else -filled
        assert position >= 0
