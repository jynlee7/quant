"""B9c: no sequence that includes protective stops goes short (spec §10, D9, I4; your call of
2026-10-14).

Hypothesis draws an account (a position and the SELL orders already working against it), then
a run of steps in any order: stop approvals and SELL checks against fresh snapshots, spent at
once or later, time passing (so approvals expire), entries filling, every working SELL LIMIT
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
from hypothesis import given
from hypothesis import strategies as st

from quant.core.errors import ApprovalError, BrokerError
from quant.core.orders import InternalBook, OrderKind, Quote, Side, TradingMode
from quant.risk.engine import RiskApprovedOrder, RiskApprovedStop, RiskContext
from quant.risk.kill import KillExecutor, KillRequest, KillStatus, MemoryKillLock
from quant.risk.triggers import KillTrigger
from tests.support.broker import FakeBroker, RecordingAlerts, RecordingKillStore
from tests.support.risk import CAL, DAY, PRICE, config, decisions, desk, intent, order

RED = pytest.mark.xfail(strict=True, reason="red:B9", raises=NotImplementedError)
RED_KILL = pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)  # no stop yet

steps = st.lists(
    st.one_of(
        st.tuples(st.just("stop"), st.integers(1, 450), st.booleans()),
        st.tuples(st.just("sell"), st.integers(1, 450), st.booleans()),
        st.tuples(st.just("spend"), st.just(0), st.just(False)),
        st.tuples(st.just("advance"), st.sampled_from([1, 3, 6]), st.just(False)),
        st.tuples(st.just("entry fills"), st.integers(1, 100), st.just(False)),
        st.tuples(st.just("sells fill"), st.just(0), st.just(False)),
        st.tuples(st.just("stops trigger"), st.just(0), st.just(False)),
    ),
    max_size=16,
)


@RED
@pytest.mark.spec("SPEC§10", "D9", "I4")
@given(
    position=st.integers(0, 400),
    working=st.lists(st.tuples(st.sampled_from(OrderKind), st.integers(1, 200)), max_size=3),
    run=steps,
)
def test_no_sequence_with_stops_goes_short(
    position: int, working: list[tuple[OrderKind, int]], run: list[tuple[str, int, bool]]
) -> None:
    d = desk(mode=TradingMode.PAPER)
    tau = decisions()[0].tau
    orders = []
    reserved = 0
    for n, (kind, quantity) in enumerate(working):  # what Alpaca would have let stand
        if reserved + quantity <= position:
            orders.append(order(f"w-{n}", Side.SELL, kind, quantity, 99.9))
            reserved += quantity
    broker = FakeBroker(d.engine, d.clock, position=position, orders=orders)
    unspent: list[RiskApprovedOrder | RiskApprovedStop] = []
    legs = 0

    def send(approval: RiskApprovedOrder | RiskApprovedStop) -> None:
        try:
            if isinstance(approval, RiskApprovedStop):
                broker.place_stop(approval)
            else:
                broker.submit(approval, client_order_id=f"o-{approval.approval_id}")
        except (ApprovalError, BrokerError):
            pass  # refused: expired, revoked, or no free shares at the broker

    for kind, number, spend_now in run:
        if kind == "stop":
            try:
                approval: RiskApprovedOrder | RiskApprovedStop = d.engine.approve_stop(
                    broker.snapshot(), broker.quote(), quantity=number, stop_price=95.0, reason="p"
                )
            except ApprovalError:
                continue
        elif kind == "sell":
            legs += 1
            snapshot = broker.snapshot()
            verdict = d.engine.check(
                intent(Side.SELL, number, tau=tau, leg=legs),
                RiskContext(
                    broker=snapshot,
                    book=InternalBook(position=snapshot.position, open_orders=snapshot.open_orders),
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
            d.clock.advance(timedelta(seconds=number))
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


@RED_KILL
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
