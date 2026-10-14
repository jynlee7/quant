"""B9c review: stops sized on what a kill already sent, never left bare, and saves that can't
lose a halt (spec §10, ARCHITECTURE §8.1, §8.3 and §16, I4, D9, C3).

The B9c review (2026-10-14) found these, each proven here by a test that failed first:

- R1/C4: a flatten spent after a snapshot was taken wasn't counted against it, so a stop (or a
  SELL) could be approved over the shares that flatten sells;
- R2/C1: at the deadline the kill sent its cancels and sized the stop at once, from a snapshot
  that still showed the exits, so the shares were bare once the cancels landed;
- C2: outside regular hours a QUEUED kill with no stop placed none (your call of 2026-10-14:
  it must place one under the shares no working sell covers);
- C3: the kill asked for a stop over every bare share, so any share another approval held made
  the engine refuse it outright, leaving even the free shares bare;
- R3: a SpentApprovals store that failed to record a spend stopped risk-reducing approvals too;
- R4/C5: a SELL that tripped a halt went out on a save that failed, so a restart from the last
  saved state lost the halt;
- R5: a gap between exits restarted the stop-missing clock instead of keeping the exits' 5
  minutes;
- C6: a stop approval dropped unsent couldn't be released;
- C7: a save failure inside a step that then failed never reached the kill's report.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from quant.core.errors import ApprovalError, BrokerError
from quant.core.orders import BrokerSnapshot, OrderKind, Quote, Side
from quant.risk import engine as risk_engine
from quant.risk.engine import (
    RiskApprovedCancel,
    RiskApprovedFlatten,
    RiskApprovedOrder,
    RiskState,
    Veto,
)
from quant.risk.kill import KillStatus
from quant.risk.triggers import KillTrigger
from tests.support.broker import FakeBroker
from tests.support.risk import DAY, EQUITY, PRICE, STOP, desk, intent, order
from tests.unit.test_risk_kill import ENTRY, EXIT, Rig, et
from tests.unit.test_risk_kill import STOP as STOP_300
from tests.unit.test_risk_monitor import SHARES, Watch

S = timedelta(seconds=1)


def snapshot_at(at: datetime, position: int = 300, orders: tuple[Any, ...] = ()) -> BrokerSnapshot:
    return BrokerSnapshot(taken_at=at, position=position, equity=EQUITY, open_orders=orders)


def quote_at(at: datetime) -> Quote:
    return Quote(bid=PRICE, ask=PRICE, at=at)


def stopped_shares(broker: FakeBroker) -> int:
    return sum(
        o.quantity
        for o in broker.orders.values()
        if o.side is Side.SELL and o.kind is OrderKind.STOP
    )


# --- R1/C4: a flatten spent since the snapshot is in flight ---------------------------------------


@pytest.mark.xfail(strict=True, reason="red:B9", raises=pytest.fail.Exception)
@pytest.mark.spec("SPEC§10", "D9", "I4", "ARCH§8.1")
def test_a_stop_is_refused_over_shares_a_flatten_spent_since_its_snapshot() -> None:
    d = desk()
    before = snapshot_at(d.clock.now())
    d.clock.advance(S)
    d.engine.spend(d.engine.approve_flatten(reason="kill"))
    with pytest.raises(ApprovalError):
        d.engine.approve_stop(
            before, quote_at(d.clock.now()), quantity=300, stop_price=STOP, reason="protect"
        )
    # A snapshot taken after the flatten shows what it did: here it failed at the broker.
    d.clock.advance(S)
    after = snapshot_at(d.clock.now())
    stop = d.engine.approve_stop(
        after, quote_at(d.clock.now()), quantity=300, stop_price=STOP, reason="protect"
    )
    assert stop.quantity == 300


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)
@pytest.mark.spec("SPEC§10", "D9", "I4", "ARCH§8.1")
def test_a_sell_is_vetoed_over_shares_a_flatten_spent_since_its_snapshot() -> None:
    d = desk()
    before = d.ctx(position=300)
    d.clock.advance(S)
    d.engine.spend(d.engine.approve_flatten(reason="kill"))
    assert isinstance(d.engine.check(intent(Side.SELL, 300), before), Veto)


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)
@pytest.mark.spec("SPEC§10", "D9", "I4")
def test_a_stop_approved_on_a_snapshot_from_before_a_kill_exit_never_goes_short() -> None:
    # R1's scenario through the fake broker: the exit fills, then the stop triggers.
    d = desk()
    broker = FakeBroker(d.engine, d.clock, position=300)
    before = broker.snapshot()
    d.clock.advance(S)
    broker.flatten(d.engine.approve_flatten(reason="kill"), limit_price=99.9)
    try:
        stop = d.engine.approve_stop(
            before, broker.quote(), quantity=300, stop_price=STOP, reason="protect"
        )
    except ApprovalError:
        pass
    else:
        broker.place_stop(stop)
        broker.trigger_stops()
    assert min(broker.positions) >= 0


# --- R2/C1: at the deadline the kill waits for its cancels before sizing the stop -----------------


class SlowCancelBroker(FakeBroker):
    """A FakeBroker whose cancels land `cancel_s` seconds after the broker accepts them; until
    then the order still shows and still reserves its shares."""

    cancel_s = 1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.landing: dict[str, datetime] = {}

    def cancel(self, approval: RiskApprovedCancel) -> None:
        self._spend(approval)
        self._maybe_fail("cancel")
        if approval.client_order_id in self.orders:
            self.landing.setdefault(
                approval.client_order_id, self.clock.now() + timedelta(seconds=self.cancel_s)
            )
        self.events.append(("cancel", approval.client_order_id))

    def land(self) -> None:
        for client_order_id, at in list(self.landing.items()):
            if self.clock.now() >= at:
                self.orders.pop(client_order_id, None)
                del self.landing[client_order_id]

    def snapshot(self) -> BrokerSnapshot:
        self.land()
        return super().snapshot()

    def flatten(self, approval: RiskApprovedFlatten, *, limit_price: float) -> str:
        self.land()
        return super().flatten(approval, limit_price=limit_price)


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)
@pytest.mark.spec("SPEC§10", "ARCH§8.3", "ARCH§16")
def test_at_the_deadline_the_stop_waits_for_the_exit_s_cancel_to_land() -> None:
    rig = Rig(fill_at=1.0, broker_type=SlowCancelBroker)  # no exit ever fills
    report = rig.run()
    assert report.status is KillStatus.INCOMPLETE
    broker = rig.broker
    assert isinstance(broker, SlowCancelBroker)
    rig.clock.advance(60 * S)
    broker.land()  # every cancel sent has landed by now
    assert broker.position == 300
    assert stopped_shares(broker) == 300  # no share is left bare
    broker.trigger_stops()
    assert min(broker.positions) >= 0


# --- C2: an outside-hours kill with shares no stop covers places one ------------------------------


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)
@pytest.mark.spec("SPEC§10", "ARCH§8.3", "ARCH§16", "I4")
@pytest.mark.parametrize(
    ("orders", "placed"),
    [((ENTRY, EXIT), 300), ((order("stop-1", Side.SELL, OrderKind.STOP, 100, 95.0), EXIT), 200)],
    ids=["no stop", "a stop over a third"],
)
def test_outside_hours_a_queued_kill_places_a_stop_on_the_bare_shares(
    orders: tuple[Any, ...], placed: int
) -> None:
    rig = Rig(at=et(DAY, 18), day=None, orders=orders)
    report = rig.run()
    assert report.status is KillStatus.QUEUED
    assert report.queued_exit_at is not None
    assert [(s.quantity, s.stop_price) for s in report.stops] == [(placed, 99.00)]
    assert stopped_shares(rig.broker) == 300
    assert "flatten" not in rig.kinds()
    assert rig.kinds().index("place_stop") < rig.kinds().index("alert")
    rig.broker.trigger_stops()
    assert min(rig.broker.positions) >= 0


@pytest.mark.spec("SPEC§10", "ARCH§8.3")
def test_outside_hours_a_queued_kill_whose_stop_covers_everything_places_none() -> None:
    rig = Rig(at=et(DAY, 18), day=None, orders=(STOP_300, EXIT))
    report = rig.run()
    assert report.status is KillStatus.QUEUED
    assert report.stops == ()
    assert report.kept == ("stop-1",)


# --- C3/C9: the kill's stop takes the shares that are free --------------------------------------


class ReservingBroker(FakeBroker):
    """Execution holds a 10-share stop approval it hasn't sent when the kill asks for a quote."""

    held: object = None

    def quote(self) -> Quote:
        result = super().quote()
        if self.held is None:
            self.held = self.engine.approve_stop(
                self.snapshot(), result, quantity=10, stop_price=STOP, reason="execution"
            )
        return result


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)
@pytest.mark.spec("SPEC§10", "ARCH§8.3", "D9")
def test_the_kill_s_stop_covers_the_free_shares_when_another_approval_holds_some() -> None:
    # Outside hours, an exit for 50 that never cancels: 250 shares bare, 10 of them reserved.
    rig = Rig(
        at=et(DAY, 18),
        day=None,
        orders=(EXIT,),
        fail={"cancel": 10_000},
        broker_type=ReservingBroker,
    )
    report = rig.run()
    assert report.status is KillStatus.INCOMPLETE
    covered = sum(s.quantity for s in report.stops)
    assert 240 <= covered <= 250
    rig.broker.trigger_stops()
    rig.broker.fill_sells()
    assert min(rig.broker.positions) >= 0


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AttributeError)
@pytest.mark.spec("ARCH§8.1", "D9")
def test_the_engine_says_how_many_shares_are_free_for_a_stop() -> None:
    d = desk()
    working = (order("x-1", Side.SELL, OrderKind.LIMIT, 50, 99.9),)
    held = d.stop(100, open_orders=working)
    free = d.engine.free_for_stop(snapshot_at(d.clock.now(), orders=working))  # type: ignore[attr-defined]
    assert free == 150
    assert held.quantity == 100


# --- R3: a failing SpentApprovals store doesn't stop risk-reducing approvals ----------------------


class FullSpentStore:
    """A SpentApprovals store on a full ops.db: every write raises, reads still work."""

    def __init__(self) -> None:
        self.full = True
        self.ids: dict[str, datetime] = {}

    def is_spent(self, approval_id: str) -> bool:
        return approval_id in self.ids

    def spend(self, approval_id: str, at: datetime) -> bool:
        if self.full:
            raise OSError("ops.db is full")
        if approval_id in self.ids:
            return False
        self.ids[approval_id] = at
        return True


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)
@pytest.mark.spec("SPEC§10", "ARCH§8.1", "ARCH§8.3", "I4")
def test_a_spent_store_that_can_t_record_doesn_t_stop_the_kill() -> None:
    rig = Rig(orders=())
    store = FullSpentStore()
    rig.engine._spent = store  # the store B13 will back with ops.db
    report = rig.run()
    assert report.status is KillStatus.FLAT
    assert rig.broker.position == 0
    assert "OSError: ops.db is full" in report.errors


@pytest.mark.xfail(strict=True, reason="red:B9", raises=OSError)
@pytest.mark.spec("ARCH§8.1", "I4", "ARCH§4.4")
def test_a_reducing_spend_survives_the_spent_store_but_is_still_spent_once() -> None:
    store = FullSpentStore()
    d = desk(spent=store)
    flatten = d.engine.approve_flatten(reason="kill")
    assert d.engine.spend(flatten) is flatten
    assert isinstance(d.engine.unsaved, OSError)
    with pytest.raises(ApprovalError):
        d.engine.spend(flatten)
    sell = d.check(intent(Side.SELL, 100), position=300)
    assert isinstance(sell, RiskApprovedOrder)
    assert d.engine.spend(sell) is sell
    buy = d.check(intent(Side.BUY, 100, leg=2), position=300)
    assert isinstance(buy, RiskApprovedOrder)
    with pytest.raises(OSError, match="is full"):
        d.engine.spend(buy)  # a BUY still needs its record
    store.full = False
    d.engine.spend(d.engine.approve_cancel("eng1-x", reason="kill"))  # the next call catches up
    assert d.engine.unsaved is None
    assert {flatten.approval_id, sell.approval_id} <= set(store.ids)


# --- R4/C5: a SELL that trips a halt needs its save -----------------------------------------------


class Disk:
    """An `on_change` that saves every state it is given, and raises OSError while `full`."""

    def __init__(self) -> None:
        self.full = False
        self.saved: list[RiskState] = []

    def __call__(self, state: RiskState) -> None:
        if self.full:
            raise OSError("disk full")
        self.saved.append(state)


def started(**changes: Any) -> RiskState:
    return RiskState(session=DAY, start_equity=EQUITY, peak_equity=EQUITY, **changes)


@pytest.mark.xfail(strict=True, reason="red:B9", raises=pytest.fail.Exception)
@pytest.mark.spec("C3", "ARCH§8.1", "SPEC§10")
def test_the_sell_that_reaches_the_submission_limit_isn_t_sent_on_a_failed_save() -> None:
    disk = Disk()
    d = desk(state=started(submissions=59), on_change=disk)
    sell = d.check(intent(Side.SELL, 100), position=300)
    assert isinstance(sell, RiskApprovedOrder)
    disk.full = True
    with pytest.raises(OSError, match="disk full"):
        d.engine.spend(sell)  # the 60th halts: the adapter sends nothing unless it is saved
    assert d.engine.state.halted  # the halt holds in memory
    with pytest.raises(OSError, match="disk full"):
        d.check(intent(Side.SELL, 100, leg=2), position=300)  # and holds SELLs until saved
    disk.full = False
    assert isinstance(d.check(intent(Side.SELL, 100, leg=2), position=300), RiskApprovedOrder)
    assert disk.saved[-1].halted


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)
@pytest.mark.spec("C3", "ARCH§8.1", "SPEC§10")
def test_a_restart_never_lets_more_than_60_submissions_out() -> None:
    # R4's scenario: 59 saved; the 60th is a SELL while the save fails; then a restart from the
    # last saved state, keeping the spent store, approves and spends a BUY.
    disk = Disk()
    spent = risk_engine.MemorySpentApprovals()
    d = desk(state=started(submissions=59), on_change=disk, spent=spent)
    sell = d.check(intent(Side.SELL, 100), position=300)
    assert isinstance(sell, RiskApprovedOrder)
    disk.full = True
    sent = 59
    try:
        d.engine.spend(sell)
        sent += 1
    except OSError:
        pass
    restarted = desk(state=disk.saved[-1], spent=spent, at=d.clock.now())
    buy = restarted.check(intent(Side.BUY, 100, leg=2), position=200)
    if isinstance(buy, RiskApprovedOrder):
        restarted.engine.spend(buy)
        sent += 1
    assert sent <= 60
    assert sent == 60 or restarted.engine.state.halted


@pytest.mark.xfail(strict=True, reason="red:B9", raises=pytest.fail.Exception)
@pytest.mark.spec("C3", "ARCH§8.1", "SPEC§10")
def test_the_sell_past_the_intent_limit_isn_t_handed_out_on_a_failed_save() -> None:
    disk = Disk()
    keys = tuple(f"d{n}#1" for n in range(20))
    d = desk(state=started(intents=keys), on_change=disk)
    disk.full = True
    with pytest.raises(OSError, match="disk full"):
        d.check(intent(Side.SELL, 100), position=300)  # the 21st halts
    with pytest.raises(OSError, match="disk full"):
        d.check(intent(Side.SELL, 100), position=300)  # a re-check while the halt is unsaved
    disk.full = False
    sell = d.check(intent(Side.SELL, 100), position=300)
    assert isinstance(sell, RiskApprovedOrder)
    assert disk.saved[-1].halted


@pytest.mark.spec("ARCH§8.1", "ARCH§8.3")
def test_without_a_halt_sells_still_go_out_while_the_save_fails() -> None:
    # Your call of 2026-10-14, kept: only a halt waiting for its save holds SELLs back.
    disk = Disk()
    d = desk(on_change=disk)
    disk.full = True
    sell = d.check(intent(Side.SELL, 100), position=300)
    assert isinstance(sell, RiskApprovedOrder)
    assert d.engine.spend(sell) is sell


# --- R5: a gap between exits keeps the exits' 5 minutes ------------------------------------------


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)
@pytest.mark.spec("SPEC§10", "ARCH§16", "ARCH§8.3")
def test_a_gap_between_exits_doesn_t_extend_their_5_minutes() -> None:
    exit_ = order("x-1", Side.SELL, OrderKind.LIMIT, SHARES, 99.9)
    w = Watch()
    first = w.tick(100.0, orders=(exit_,))
    assert first.exit_covered
    w.advance(299)
    gap = w.tick(100.0, orders=())  # cancelled for the re-quote; the new one isn't out yet
    assert gap.replace_stop
    assert gap.kill == ()
    w.advance(1)
    late = w.tick(100.0, orders=())
    assert late.kill == (KillTrigger.STOP_MISSING,)


# --- C6: a stop approval dropped unsent can be released --------------------------------------------


@pytest.mark.xfail(strict=True, reason="red:B9", raises=ApprovalError)
@pytest.mark.spec("I4", "ARCH§8.1", "ARCH§4.4")
def test_a_stop_approval_dropped_unsent_can_be_released() -> None:
    d = desk()
    stop = d.stop(300)
    d.engine.release(stop)  # type: ignore[arg-type]
    sell = d.check(intent(Side.SELL, 100), position=300)
    assert isinstance(sell, RiskApprovedOrder)
    with pytest.raises(ApprovalError):
        d.engine.spend(stop)


# --- C7: a save failure inside a failed step still reaches the report -----------------------------


class FreshFailures:
    """An `on_change` that raises a new OSError on every call while `failing`."""

    def __init__(self) -> None:
        self.failing = False

    def __call__(self, state: RiskState) -> None:
        if self.failing:
            raise OSError("ops.db is full")


class OtherThreadBroker(FakeBroker):
    """Before the first exit is spent, another thread's call fails to save; the spend's retry
    fails too, and then the broker refuses the exit. Saves work again afterwards."""

    persist: FreshFailures

    def flatten(self, approval: RiskApprovedFlatten, *, limit_price: float) -> str:
        if not getattr(self, "interfered", False):
            self.interfered = True
            self.persist.failing = True
            self.engine.approve_cancel("eng1-other", reason="another thread")
            self._spend(approval)
            self.persist.failing = False
            raise BrokerError("flatten refused (fake)")
        return super().flatten(approval, limit_price=limit_price)


@pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)
@pytest.mark.spec("SPEC§10", "ARCH§8.3", "ARCH§8.1")
def test_a_save_failure_inside_a_failed_step_reaches_the_report() -> None:
    persist = FreshFailures()
    rig = Rig(orders=(), on_change=persist, broker_type=OtherThreadBroker)
    broker = rig.broker
    assert isinstance(broker, OtherThreadBroker)
    broker.persist = persist
    report = rig.run()
    assert report.status is KillStatus.FLAT
    assert "OSError: ops.db is full" in report.errors


# --- C8, not a defect: execution re-places its stop through approve_stop -------------------------


@pytest.mark.spec("D9", "I4", "ARCH§8.1", "ARCH§8.2")
def test_a_stop_approved_beside_a_sell_never_oversells_when_execution_re_places() -> None:
    # The review's scenario: 100 shares under a 40-share stop; a SELL of 40, then a stop
    # approval for the 20 left. Execution cancels the old stop, sells, and re-places the stop
    # for what is left. It can only do that through approve_stop, which sees the 20 working.
    d = desk()
    broker = FakeBroker(
        d.engine, d.clock, position=100, orders=(order("stop-1", Side.SELL, OrderKind.STOP, 40),)
    )
    sell = d.check(intent(Side.SELL, 40), position=100, open_orders=tuple(broker.orders.values()))
    assert isinstance(sell, RiskApprovedOrder)
    broker.place_stop(
        d.engine.approve_stop(
            broker.snapshot(), broker.quote(), quantity=20, stop_price=STOP, reason="protect"
        )
    )
    broker.cancel(d.engine.approve_cancel("stop-1", reason="exit"))
    broker.submit(sell, client_order_id="x-1")
    broker.fill_sells()
    d.clock.advance(S)
    after = broker.snapshot()
    with pytest.raises(ApprovalError):
        d.engine.approve_stop(
            after, broker.quote(), quantity=60, stop_price=STOP, reason="re-place"
        )
    broker.place_stop(
        d.engine.approve_stop(
            after, broker.quote(), quantity=40, stop_price=STOP, reason="re-place"
        )
    )
    broker.trigger_stops()
    assert broker.positions == [100, 60, 40, 0]  # the 20 and the 40 trigger, never short
