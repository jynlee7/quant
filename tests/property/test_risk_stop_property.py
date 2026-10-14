"""B9c: no sequence that includes protective stops goes short (spec §10, D9, I4; your call of
2026-10-14).

Hypothesis draws an account (a position and the SELL orders already working against it), then
a run of steps in any order: stop approvals and SELL checks, judged on a fresh snapshot or on
one read a few steps before (so orders spent since are in flight), spent at once or later, time passing (so approvals expire), entries filling, every working SELL LIMIT
filling, and the price falling through every stop. The FakeBroker places a stop without asking
whether other SELLs already reserve its shares, so only the engine's own bound stands between a
stop and a share sold twice; it refuses a SELL LIMIT beyond the free shares, as Alpaca does
(ARCHITECTURE §8.2), which is what the SELL clamp leans on for working stops. At the end every
stop triggers and every SELL fills. Whatever was drawn, the signed position never went below 0.

A second property runs the kill executor to a drawn end, with exits that may never fill and
cancels that may never land, then lets every stop trigger and every SELL fill: never short, and
an INCOMPLETE kill leaves every remaining share under a working SELL.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from quant.core.errors import ApprovalError, BrokerError
from quant.core.orders import BrokerSnapshot, InternalBook, OrderKind, Quote, Side, TradingMode
from quant.risk.engine import RiskApprovedOrder, RiskApprovedStop, RiskContext
from quant.risk.kill import KillExecutor, KillRequest, KillStatus, MemoryKillLock
from quant.risk.triggers import KillTrigger
from tests.support.broker import FakeBroker, RecordingAlerts, RecordingKillStore
from tests.support.risk import CAL, DAY, PRICE, config, decisions, desk, intent, order

# Few distinct sizes, so stops and SELLs often claim the same shares.
SHARES = st.sampled_from([1, 100, 200, 300, 301])
# Stops and SELLs come often, so they meet; the repeats weight the draw.
KINDS = st.sampled_from(
    ["stop"] * 4
    + ["sell"] * 2
    + ["spend"] * 2
    + ["look"] * 3
    + ["advance", "entry fills", "sells fill", "stops trigger"]
)
steps = st.lists(st.tuples(KINDS, SHARES, st.booleans()), min_size=2, max_size=14)


@pytest.mark.spec("SPEC§10", "D9", "I4")
# Each run is cheap and the shorts it hunts need a few steps in the right order, so it takes
# more examples than the fast profile's 50 (at least 300, and the full tier's 500 there).
@settings(max_examples=max(300, settings().max_examples))
@given(
    position=st.sampled_from([0, 100, 300, 400]),
    working=st.lists(st.tuples(st.sampled_from(OrderKind), SHARES), max_size=2),
    run=steps,
)
# Found by the full tier: a SELL took the shares under a working stop after a stop was approved.
@example(position=100, working=[(OrderKind.STOP, 1)], run=[("stop", 1, False), ("sell", 100, True)])
# ... and one under a stop spent but not yet on the snapshot.
@example(
    position=300,
    working=[],
    run=[
        ("stop", 100, False),
        ("stop", 200, False),
        ("look", 1, False),
        ("spend", 1, False),
        ("sell", 1, True),
    ],
)
def test_no_sequence_with_stops_goes_short(
    position: int, working: list[tuple[OrderKind, int]], run: list[tuple[str, int, bool]]
) -> None:
    d = desk(mode=TradingMode.PAPER)
    tau = decisions()[0].tau
    orders = []
    reserved = 0
    for n, (order_kind, quantity) in enumerate(working):  # what Alpaca would have let stand
        if reserved + quantity <= position:
            orders.append(order(f"w-{n}", Side.SELL, order_kind, quantity, 99.9))
            reserved += quantity
    broker = FakeBroker(d.engine, d.clock, position=position, orders=orders)
    unspent: list[RiskApprovedOrder | RiskApprovedStop] = []
    legs = 0
    seen = None  # the snapshot "look" read last; stops and SELLs are judged on it while fresh

    def send(approval: RiskApprovedOrder | RiskApprovedStop) -> None:
        try:
            if isinstance(approval, RiskApprovedStop):
                broker.place_stop(approval)
            else:
                broker.submit(approval, client_order_id=f"o-{approval.approval_id}")
        except (ApprovalError, BrokerError):
            pass  # refused: expired, revoked, or no free shares at the broker

    def snapshot() -> BrokerSnapshot:
        return seen if seen is not None else broker.snapshot()

    for kind, number, spend_now in run:
        if kind == "look":
            seen = broker.snapshot()
            continue
        if kind == "stop":
            try:
                approval: RiskApprovedOrder | RiskApprovedStop = d.engine.approve_stop(
                    snapshot(), broker.quote(), quantity=number, stop_price=95.0, reason="p"
                )
            except ApprovalError:
                continue
        elif kind == "sell":
            legs += 1
            snapshot_ = snapshot()
            verdict = d.engine.check(
                intent(Side.SELL, number, tau=tau, leg=legs),
                RiskContext(
                    broker=snapshot_,
                    book=InternalBook(
                        position=snapshot_.position, open_orders=snapshot_.open_orders
                    ),
                    quote=Quote(bid=PRICE, ask=PRICE, at=d.clock.now()),
                    limit_price=PRICE,
                ),
            )
            if not isinstance(verdict, RiskApprovedOrder):
                continue
            approval = verdict
        elif kind == "spend":
            if unspent:
                send(unspent.pop(0))
            continue
        elif kind == "advance":
            d.clock.advance(timedelta(seconds=3))  # two of them outlive a 5 s snapshot
            continue
        elif kind == "entry fills":
            broker.move(number)
            continue
        elif kind == "sells fill":
            broker.fill_sells()
            continue
        else:
            broker.trigger_stops()
            continue
        if spend_now:
            send(approval)
        else:
            unspent.append(approval)

    for approval in unspent:  # whatever is left goes out late, if the engine lets it
        send(approval)
    broker.trigger_stops()
    broker.fill_sells()
    assert min(broker.positions) >= 0, broker.positions


@pytest.mark.spec("SPEC§10", "D9", "ARCH§8.3", "I4")
@given(
    position=st.integers(1, 400),
    covered=st.integers(0, 400),
    fill_at=st.sampled_from([None, 1.0]),
    partial=st.sampled_from([None, 1, 50]),
    cancels_fail=st.sampled_from([0, 3, 10_000]),
    minutes_to_close=st.sampled_from([2, 120]),
)
def test_a_kill_that_cant_finish_never_leaves_shares_to_be_sold_twice(
    position: int,
    covered: int,
    fill_at: float | None,
    partial: int | None,
    cancels_fail: int,
    minutes_to_close: int,
) -> None:
    close = CAL.session(DAY).close
    d = desk(at=close - timedelta(minutes=minutes_to_close))
    orders = []
    if covered:
        orders.append(order("stop-1", Side.SELL, OrderKind.STOP, min(covered, position), 95.0))
    broker = FakeBroker(
        d.engine,
        d.clock,
        position=position,
        orders=orders,
        fill_at=fill_at,
        partial=partial,
        fail={"cancel": cancels_fail},
    )
    report = KillExecutor(
        d.engine,
        broker,
        config(),
        clock=d.clock,
        lock=MemoryKillLock(),
        store=RecordingKillStore(),
        alerts=RecordingAlerts(),
        calendar=CAL,
    ).execute(KillRequest(KillTrigger.MANUAL, "cli", "drawn"))

    assert report.status in (KillStatus.FLAT, KillStatus.INCOMPLETE)
    if report.status is KillStatus.INCOMPLETE and broker.position > 0:
        selling = sum(o.quantity for o in broker.orders.values() if o.side is Side.SELL)
        assert selling >= broker.position  # every remaining share is under a SELL
    broker.trigger_stops()
    broker.fill_sells()
    assert min(broker.positions) >= 0, broker.positions
