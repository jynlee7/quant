"""B9b: the kill executor halts, cancels, flattens, verifies and alerts; outside regular hours it
keeps the stop and queues the exit (spec §10, ARCHITECTURE §8.2 and §8.3, I4).

Every run here goes against tests.support.broker.FakeBroker on a SimClock, so the 30 s re-quotes
and the 300 s deadline take no wall time. DAY (2018-03-06) is a full session, 09:30-16:00 ET
(14:30-21:00 UTC); the account holds 300 shares at $100 under a 300-share stop.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest

from quant.core.calendar import NEW_YORK
from quant.core.errors import ApprovalError, RiskError
from quant.core.orders import OrderKind, Quote, Side
from quant.risk.engine import RiskApprovedCancel, RiskApprovedFlatten
from quant.risk.kill import (
    FileKillLock,
    KillExecutor,
    KillReport,
    KillRequest,
    KillStatus,
    MemoryKillLock,
    in_regular_hours,
    kill_collar_bp,
    kill_limit_price,
    next_open,
)
from quant.risk.limits import RiskConfig
from quant.risk.triggers import KillTrigger
from tests.support.broker import Event, FakeBroker, RecordingAlerts, RecordingKillStore
from tests.support.risk import CAL, DAY, HALF_DAY, NEXT_DAY, PRICE, Desk, config, desk, intent
from tests.support.risk import order as open_order

RED = pytest.mark.xfail(strict=True, reason="red:B9", raises=NotImplementedError)
REQUEST = KillRequest(KillTrigger.MANUAL, "cli", "test kill")
STOP = open_order("stop-1", Side.SELL, OrderKind.STOP, 300, 95.0)
ENTRY = open_order("o-1", Side.BUY, OrderKind.LIMIT, 100, 100.10)
EXIT = open_order("x-1", Side.SELL, OrderKind.LIMIT, 50, 99.90)


def et(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), NEW_YORK).astimezone(UTC)


def kill_config(**changes: float) -> RiskConfig:
    limits = config()
    return limits.model_copy(update={"kill": limits.kill.model_copy(update=changes)})


class Rig:
    """A desk, a fake broker sharing its engine, a store, alerts, a lock and an executor."""

    def __init__(
        self,
        *,
        at: datetime | None = None,
        day: date | None = DAY,
        limits: RiskConfig | None = None,
        store_fails: bool = False,
        alerts_fail: bool = False,
        **broker: object,
    ) -> None:
        self.events: list[Event] = []
        self.desk: Desk = desk(at=at, day=day)
        self.engine = self.desk.engine
        self.clock = self.desk.clock
        broker.setdefault("position", 300)
        broker.setdefault("orders", (STOP,))
        self.broker = FakeBroker(self.engine, self.clock, events=self.events, **broker)  # type: ignore[arg-type]
        self.store = RecordingKillStore(self.events, fail_writes=store_fails)
        self.alerts = RecordingAlerts(self.events, fail=alerts_fail)
        self.lock = MemoryKillLock()
        self.limits = config() if limits is None else limits
        self.executor = KillExecutor(
            self.engine,
            self.broker,
            self.limits,
            clock=self.clock,
            lock=self.lock,
            store=self.store,
            alerts=self.alerts,
            calendar=CAL,
        )

    def run(self, request: KillRequest = REQUEST) -> KillReport:
        return self.executor.execute(request)

    def kinds(self) -> list[str]:
        return [event[0] for event in self.events]


# --- regular hours: halt, cancel, flatten, verify, alert -------------------------------------------


@pytest.mark.spec("SPEC§10", "ARCH§8.3", "I4")
@RED
def test_in_regular_hours_the_executor_halts_cancels_flattens_verifies_and_alerts() -> None:
    rig = Rig(orders=(STOP, ENTRY))
    report = rig.run()

    assert report.status is KillStatus.FLAT
    assert report.regular_hours
    # 1. halt, before anything reaches the broker
    assert rig.engine.state.halted
    assert report.halt_reason == "kill switch (manual, cli): test kill"
    assert rig.engine.state.halt_reason == report.halt_reason
    assert rig.events[0] == ("write_halted", report.halt_reason, report.started_at)
    assert rig.broker.halted_at_first_change is True
    # 2. every order cancelled, the stop included, before the exit
    assert sorted(report.cancelled) == ["o-1", "stop-1"]
    first_flatten = rig.kinds().index("flatten")
    assert {e[1] for e in rig.events[:first_flatten] if e[0] == "cancel"} == {"o-1", "stop-1"}
    # 3. one exit at the bid - 10 bp
    assert [(x.limit_price, x.collar_bp) for x in report.exits] == [(99.90, 10.0)]
    assert rig.broker.flatten_quantities == [300]
    # 4. verified flat on a snapshot taken after the exit
    assert report.final is not None
    assert report.final.position == 0
    assert report.final.open_orders == ()
    assert report.final.taken_at >= report.exits[0].sent_at
    assert rig.broker.position == 0
    assert rig.broker.orders == {}
    # 5. alerted, last, with the report returned
    assert rig.alerts.reports == [report]
    assert rig.events[-1] == ("alert", KillStatus.FLAT)
    # every change went through an approval the engine issued and spent (I4)
    assert [type(a) for a in rig.broker.spent] == [
        RiskApprovedCancel,
        RiskApprovedCancel,
        RiskApprovedFlatten,
    ]
    assert report.errors == ()
    assert not rig.lock.held


@pytest.mark.spec("SPEC§10", "ARCH§8.2", "ARCH§8.3")
@RED
def test_an_unfilled_kill_exit_is_cancelled_and_re_quoted_wider_every_30_s() -> None:
    rig = Rig(fill_at=99.75)  # fills only at 99.75 or lower: the third exit, at 30 bp
    report = rig.run()

    assert report.status is KillStatus.FLAT
    assert [(x.limit_price, x.collar_bp) for x in report.exits] == [
        (99.90, 10.0),
        (99.80, 20.0),
        (99.70, 30.0),
    ]
    gaps = [b.sent_at - a.sent_at for a, b in zip(report.exits, report.exits[1:], strict=False)]
    assert all(timedelta(seconds=30) <= gap <= timedelta(seconds=32) for gap in gaps), gaps
    # each unfilled exit is cancelled before the next goes out
    flattens = [e for e in rig.events if e[0] in ("flatten", "cancel") and e[1] != "stop-1"]
    assert [e[:2] for e in flattens] == [
        ("flatten", "kill-1"),
        ("cancel", "kill-1"),
        ("flatten", "kill-2"),
        ("cancel", "kill-2"),
        ("flatten", "kill-3"),
    ]
    flattens_spent = [a for a in rig.broker.spent if isinstance(a, RiskApprovedFlatten)]
    assert len({a.approval_id for a in flattens_spent}) == 3  # a new approval per exit


@pytest.mark.spec("SPEC§10", "ARCH§8.3")
@RED
def test_the_kill_exit_never_widens_past_its_maximum_and_stops_at_the_deadline() -> None:
    rig = Rig(fill_at=1.0, limits=kill_config(widen_bp=40.0))  # never fills
    report = rig.run()

    assert report.status is KillStatus.INCOMPLETE
    assert [x.collar_bp for x in report.exits][:5] == [10.0, 50.0, 90.0, 100.0, 100.0]
    assert max(x.collar_bp for x in report.exits) == 100.0
    assert min(x.limit_price for x in report.exits) == 99.00
    elapsed = report.finished_at - report.started_at
    assert timedelta(seconds=300) <= elapsed <= timedelta(seconds=301)
    assert rig.alerts.reports == [report]
    assert rig.engine.state.halted
    assert not rig.lock.held


@pytest.mark.spec("SPEC§10", "ARCH§8.3")
@RED
def test_after_a_partial_fill_the_next_exit_closes_only_the_remainder() -> None:
    rig = Rig(partial=100)
    report = rig.run()
    assert report.status is KillStatus.FLAT
    assert rig.broker.flatten_quantities == [300, 200]
    assert len(report.exits) == 2


@pytest.mark.spec("SPEC§10", "ARCH§8.3", "ARCH§16")
@RED
def test_broker_errors_are_recorded_and_retried_until_flat() -> None:
    rig = Rig(fail={"cancel": 1, "flatten": 2, "snapshot": 1})
    report = rig.run()
    assert report.status is KillStatus.FLAT
    assert len(report.errors) == 4
    assert all(error.startswith("BrokerError: ") for error in report.errors)
    assert len(report.exits) == 1  # a failed flatten sends nothing to record
    assert rig.broker.position == 0


@pytest.mark.spec("SPEC§10", "ARCH§8.3")
@RED
def test_a_failing_halt_write_or_alert_never_stops_the_kill() -> None:
    rig = Rig(store_fails=True, alerts_fail=True)
    report = rig.run()
    assert report.status is KillStatus.FLAT
    assert rig.engine.state.halted
    assert any(error.startswith("OSError: ") for error in report.errors)
    assert rig.alerts.reports == [report]
    assert rig.broker.position == 0


@pytest.mark.spec("SPEC§10", "ARCH§8.3")
@RED
def test_an_account_already_flat_is_verified_halted_and_alerted_without_an_exit() -> None:
    rig = Rig(position=0, orders=())
    report = rig.run()
    assert report.status is KillStatus.FLAT
    assert report.exits == ()
    assert report.cancelled == ()
    assert rig.engine.state.halted
    assert rig.alerts.reports == [report]


@pytest.mark.spec("SPEC§10", "ARCH§8.3", "ARCH§5")
@RED
def test_a_session_closing_mid_kill_queues_the_exit_for_the_next_open() -> None:
    rig = Rig(at=et(DAY, 15, 59), fill_at=1.0)
    report = rig.run()
    assert report.status is KillStatus.INCOMPLETE
    assert report.queued_exit_at == CAL.session(NEXT_DAY).open
    assert ("queue_exit", CAL.session(NEXT_DAY).open, report.halt_reason) in rig.events
    assert report.finished_at <= CAL.session(DAY).close + timedelta(seconds=2)


@pytest.mark.spec("SPEC§10", "ARCH§8.3")
@RED
def test_a_short_position_is_closed_with_a_buy_at_the_ask_plus_the_collar() -> None:
    rig = Rig(position=-50, orders=())
    report = rig.run()
    assert report.status is KillStatus.FLAT
    assert [x.limit_price for x in report.exits] == [100.10]
    assert rig.broker.position == 0


# --- outside regular hours: keep the stop, queue the exit ------------------------------------------


@pytest.mark.spec("SPEC§10", "ARCH§8.3", "ARCH§5")
@RED
@pytest.mark.parametrize(
    ("at", "opens"),
    [
        (et(DAY, 18), NEXT_DAY),  # after the close
        (et(DAY, 8), DAY),  # before the open
        (et(date(2018, 3, 10), 12), date(2018, 3, 12)),  # Saturday; DST starts on the Sunday
        (et(HALF_DAY, 14), date(2018, 11, 26)),  # after a 13:00 early close
    ],
    ids=["evening", "pre-open", "weekend", "half-day"],
)
def test_outside_regular_hours_it_keeps_the_stop_and_queues_the_exit(
    at: datetime, opens: date
) -> None:
    rig = Rig(at=at, day=None, orders=(STOP, ENTRY, EXIT))
    report = rig.run()

    assert report.status is KillStatus.QUEUED
    assert not report.regular_hours
    assert rig.engine.state.halted
    assert rig.kinds()[0] == "write_halted"
    assert sorted(report.cancelled) == ["o-1", "x-1"]
    assert report.kept == ("stop-1",)
    assert list(rig.broker.orders) == ["stop-1"]  # the stop still protects the position
    assert "flatten" not in rig.kinds()
    assert report.exits == ()
    assert rig.broker.position == 300
    open_at = CAL.session(opens).open
    assert open_at.astimezone(NEW_YORK).time() == time(9, 30)
    assert report.queued_exit_at == open_at
    assert ("queue_exit", open_at, report.halt_reason) in rig.events
    assert rig.alerts.reports == [report]
    assert rig.events[-1] == ("alert", KillStatus.QUEUED)


@pytest.mark.spec("SPEC§10", "ARCH§8.3")
@RED
def test_outside_regular_hours_a_flat_account_drops_its_orders_and_queues_nothing() -> None:
    rig = Rig(at=et(DAY, 18), day=None, position=0, orders=(STOP,))
    report = rig.run()
    assert report.status is KillStatus.FLAT
    assert report.cancelled == ("stop-1",)
    assert report.queued_exit_at is None
    assert "queue_exit" not in rig.kinds()


@pytest.mark.spec("ARCH§8.3", "ARCH§5")
@RED
def test_regular_hours_follow_the_calendar() -> None:
    session = CAL.session(DAY)
    us = timedelta(microseconds=1)
    assert in_regular_hours(session.open, CAL)
    assert not in_regular_hours(session.open - us, CAL)
    assert in_regular_hours(session.close - us, CAL)
    assert not in_regular_hours(session.close, CAL)
    assert not in_regular_hours(et(HALF_DAY, 13), CAL)  # the early close
    assert not in_regular_hours(et(date(2018, 3, 10), 12), CAL)  # a Saturday
    assert next_open(session.open - us, CAL) == session.open
    assert next_open(session.close, CAL) == CAL.session(NEXT_DAY).open


# --- the lock: one copy at a time ------------------------------------------------------------------


@pytest.mark.spec("ARCH§8.3")
@RED
def test_a_second_copy_does_nothing_while_the_lock_is_held() -> None:
    rig = Rig()
    assert rig.lock.acquire()
    report = rig.run()
    assert report.status is KillStatus.BUSY
    assert report.halt_reason == ""
    assert rig.events == []
    assert rig.broker.snapshots == 0
    assert not rig.engine.state.halted
    assert rig.alerts.reports == []
    assert rig.lock.held  # still the first copy's
    rig.lock.release()
    assert rig.run().status is KillStatus.FLAT


@pytest.mark.spec("ARCH§8.3")
@RED
def test_the_file_lock_excludes_every_other_holder(tmp_path: Path) -> None:
    path = tmp_path / "kill.lock"
    first, second = FileKillLock(path), FileKillLock(path)
    assert first.acquire()
    assert not second.acquire()
    with pytest.raises(RiskError):
        first.acquire()  # not re-entrant
    first.release()
    assert second.acquire()
    second.release()
    with pytest.raises(RiskError):
        second.release()
    assert path.exists()
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.spec("ARCH§8.3")
@RED
def test_the_memory_lock_is_held_by_one_caller_at_a_time() -> None:
    lock = MemoryKillLock()
    assert not lock.held
    assert lock.acquire()
    assert lock.held
    assert not lock.acquire()
    lock.release()
    assert not lock.held
    with pytest.raises(RiskError):
        lock.release()


# --- the exit's price ------------------------------------------------------------------------------


@pytest.mark.spec("SPEC§10", "ARCH§8.2", "ARCH§8.3")
@RED
def test_kill_exits_start_at_the_collar_and_widen_to_the_maximum() -> None:
    limits = config()
    quote = Quote(bid=PRICE, ask=PRICE, at=et(DAY, 11))
    widths = [kill_collar_bp(a, collar_bp=limits.collar_bp, kill=limits.kill) for a in range(12)]
    assert widths == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 100.0, 100.0]
    prices = [
        kill_limit_price(quote, 300, a, collar_bp=limits.collar_bp, kill=limits.kill)
        for a in range(12)
    ]
    assert prices == [99.9, 99.8, 99.7, 99.6, 99.5, 99.4, 99.3, 99.2, 99.1, 99.0, 99.0, 99.0]


@pytest.mark.spec("SPEC§10", "ARCH§8.3")
@RED
def test_kill_exit_prices_round_toward_marketable() -> None:
    limits = config()
    quote = Quote(bid=123.45, ask=123.47, at=et(DAY, 11))
    sell = kill_limit_price(quote, 10, 0, collar_bp=limits.collar_bp, kill=limits.kill)
    buy = kill_limit_price(quote, -10, 0, collar_bp=limits.collar_bp, kill=limits.kill)
    assert sell == 123.32  # 123.32655 rounded down
    assert buy == 123.6  # 123.59347 rounded up
    with pytest.raises(ValueError, match="position"):
        kill_limit_price(quote, 0, 0, collar_bp=limits.collar_bp, kill=limits.kill)
    with pytest.raises(ValueError, match="attempt"):
        kill_limit_price(quote, 10, -1, collar_bp=limits.collar_bp, kill=limits.kill)


# --- RiskEngine.halt, the executor's first step ----------------------------------------------------


@pytest.mark.spec("SPEC§10", "ARCH§8.1", "ARCH§8.3")
@RED
def test_halt_halts_keeps_the_first_reason_revokes_buys_and_publishes() -> None:
    seen = []
    d = desk(on_change=seen.append)
    buy = d.check(intent(Side.BUY, 100))
    seen.clear()

    assert d.engine.halt("kill switch (manual, cli): first") is True
    assert d.engine.state.halted
    assert d.engine.state.halt_reason == "kill switch (manual, cli): first"
    assert not d.engine.state.killed
    assert seen == [d.engine.state]
    with pytest.raises(ApprovalError):
        d.engine.spend(buy)  # the halt revoked the outstanding BUY

    assert d.engine.halt("second") is False
    assert d.engine.state.halt_reason == "kill switch (manual, cli): first"
    assert len(seen) == 1
    with pytest.raises(RiskError):
        d.engine.halt("")


@pytest.mark.spec("SPEC§10", "ARCH§8.3")
@RED
def test_halt_works_before_any_session() -> None:
    d = desk(day=None)
    assert d.engine.halt("kill switch (manual, telegram): from the phone")
    assert d.engine.state.halted
