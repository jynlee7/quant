"""B9b: no sequence of intents, fills, kills and flattens produces a short or breaches a cap
(spec §10, D9, ARCHITECTURE §8.3, I4).

This extends the B9a property test (test_risk_property.py) across the kill switch. Hypothesis
draws a session: intents before the kill (any side and size, any label, at the session's
intraday decisions in order), each spent and filled in part or not at all, then a kill at a
drawn moment, then more intents. The kill runs the real KillExecutor against a FakeBroker that
holds the position the fills built. Whatever was drawn:

- before the kill, every BUY approval leaves the position, counting the BUYs outstanding, inside
  the smallest cap in force and the single-order cap; every SELL approval leaves it long or flat;
- the kill ends FLAT: the broker holds no position and no open orders, trading is halted, and
  the position never went below 0 on the way;
- every approval issued before the kill and still unspent is refused at spend afterwards;
- after the kill, no intent is approved at all: BUYs are halted and there is nothing to sell.
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
from quant.risk.engine import RiskApprovedOrder
from quant.risk.kill import KillExecutor, KillRequest, KillStatus, MemoryKillLock
from quant.risk.triggers import KillTrigger
from tests.support.broker import FakeBroker, RecordingAlerts, RecordingKillStore
from tests.support.risk import CAL, EQUITY, config, decisions, desk, intent

INTRADAY = [d for d in decisions() if not d.reduce_only]  # the decisions that fill today


@dataclass(frozen=True)
class Step:
    decision: int  # an index into INTRADAY; steps run in order of it
    side: Side
    quantity: int
    label: Label | None
    spend: bool
    fill: float  # the share of a spent order that fills


steps = st.lists(
    st.builds(
        Step,
        decision=st.integers(0, len(INTRADAY) - 1),
        side=st.sampled_from(Side),
        quantity=st.one_of(st.integers(1, 300), st.integers(1, 1_200)),
        label=st.sampled_from([*Label, None]),
        spend=st.booleans(),
        fill=st.sampled_from([0.0, 0.5, 1.0]),
    ),
    max_size=12,
)


def exact(value: float) -> Decimal:
    return Decimal(repr(value))


@pytest.mark.spec("SPEC§10", "D9", "ARCH§8.3", "I4")
@given(
    held_at_start=st.integers(0, 400),
    before=steps,
    after=steps,
    trigger=st.sampled_from(KillTrigger),
)
def test_no_sequence_with_a_kill_goes_short_or_breaches_a_cap(
    held_at_start: int, before: list[Step], after: list[Step], trigger: KillTrigger
) -> None:
    lock = MemoryKillLock()  # the executor's lock, built before anything runs
    limits = config()
    d = desk(mode=TradingMode.PAPER)
    position = held_at_start
    legs: dict[int, int] = {}
    unspent: list[RiskApprovedOrder] = []

    def check(step: Step) -> object:
        decision = INTRADAY[step.decision]
        d.clock.advance_to(max(d.clock.now(), decision.tau) + timedelta(seconds=1))
        legs[step.decision] = legs.get(step.decision, 0) + 1
        return d.check(
            intent(
                step.side,
                step.quantity,
                tau=decision.tau,
                leg=legs[step.decision],
                label=step.label,
                stop=95.0,
            ),
            position=position,
        )

    for step in sorted(before, key=lambda s: s.decision):
        verdict = check(step)
        if not isinstance(verdict, RiskApprovedOrder):
            continue
        decision = INTRADAY[step.decision]
        now = d.clock.now()
        unspent[:] = [a for a in unspent if now < a.expires_at]  # the engine lets expired ones go
        if step.side is Side.BUY:
            cap = min(
                limits.max_gross_exposure,
                limits.uncertain_cap if step.label is None else limits.label_caps.cap(step.label),
            )
            if decision.last_fillable:
                cap = min(cap, limits.overnight_cap)
            outstanding = sum(a.quantity for a in unspent if a.intent.side is Side.BUY)
            held = position + outstanding + verdict.quantity
            assert held * exact(100.0) <= exact(cap) * exact(EQUITY)
        else:
            outstanding = sum(a.quantity for a in unspent if a.intent.side is Side.SELL)
            assert 0 < outstanding + verdict.quantity <= position
        if step.spend:
            try:
                d.engine.spend(verdict)
            except ApprovalError:
                continue
            filled = int(verdict.quantity * step.fill)
            position += filled if step.side is Side.BUY else -filled
            assert position >= 0
        else:
            unspent.append(verdict)

    d.clock.advance(timedelta(seconds=1))
    broker = FakeBroker(d.engine, d.clock, position=position)
    report = KillExecutor(
        d.engine,
        broker,
        limits,
        clock=d.clock,
        lock=lock,
        store=RecordingKillStore(),
        alerts=RecordingAlerts(),
        calendar=CAL,
    ).execute(KillRequest(trigger, "engine", "drawn"))

    assert report.status is KillStatus.FLAT
    assert broker.position == 0
    assert broker.orders == {}
    assert d.engine.state.halted
    assert min(broker.flatten_quantities, default=0) >= 0
    position = broker.position

    for approval in unspent:
        with pytest.raises(ApprovalError):
            d.engine.spend(approval)

    for step in sorted(after, key=lambda s: s.decision):
        verdict = check(step)
        assert not isinstance(verdict, RiskApprovedOrder), verdict
