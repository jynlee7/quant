"""B9b: the minute monitor marks daily loss and drawdown on the freshest price, reconciles, and
watches the stop (C8, spec §10, ARCHITECTURE §5 and §16).

The desk starts DAY with $100,000. The account holds 1,000 shares and no cash, so equity is
exactly 1,000 x the price: $98 is the 2% daily-loss line and $90 the 10% drawdown line.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from quant.core.orders import BrokerSnapshot, InternalBook, OpenOrder, OrderKind, Side
from quant.risk.monitor import MinuteInput, MinuteMonitor, MonitorReport
from quant.risk.reconcile import ReconcileStatus
from quant.risk.triggers import KillTrigger
from tests.support.risk import EQUITY, Desk, config, desk
from tests.support.risk import order as open_order

SHARES = 1_000
STOP = open_order("stop-1", Side.SELL, OrderKind.STOP, SHARES, 90.0)


class Watch:
    def __init__(self, d: Desk | None = None) -> None:
        self.desk = desk() if d is None else d
        self.monitor = MinuteMonitor(self.desk.engine, config(), clock=self.desk.clock)

    def tick(
        self,
        price: float = 100.0,
        *,
        price_age: float = 0.0,
        cash: float = 0.0,
        position: int = SHARES,
        orders: tuple[OpenOrder, ...] = (STOP,),
        book_position: int | None = None,
        snapshot_age: float = 0.0,
    ) -> MonitorReport:
        now = self.desk.clock.now()
        return self.monitor.tick(
            MinuteInput(
                price=price,
                price_at=now - timedelta(seconds=price_age),
                cash=cash,
                snapshot=BrokerSnapshot(
                    taken_at=now - timedelta(seconds=snapshot_age),
                    position=position,
                    equity=EQUITY,
                    open_orders=orders,
                ),
                book=InternalBook(
                    position=position if book_position is None else book_position,
                    open_orders=orders,
                ),
            )
        )

    def advance(self, seconds: float) -> None:
        self.desk.clock.advance(timedelta(seconds=seconds))


@pytest.mark.spec("C8", "SPEC§10")
def test_each_minute_marks_equity_on_the_freshest_price() -> None:
    w = Watch()
    report = w.tick(101.0)
    assert report.equity == 101_000.0
    assert report.mark is not None
    assert report.mark.equity == 101_000.0
    assert w.desk.engine.state.peak_equity == 101_000.0
    assert not report.price_stale
    assert report.reconciliation.status is ReconcileStatus.MATCH
    assert report.stop_covered is True
    assert (report.flatten, report.replace_stop, report.kill) == (False, False, ())
    assert report.at == w.desk.clock.now()

    with_cash = w.tick(100.0, cash=500.0, position=SHARES)
    assert with_cash.equity == 100_500.0


@pytest.mark.spec("C8", "SPEC§10")
def test_a_2_percent_daily_loss_between_bars_locks_entries_and_asks_to_flatten() -> None:
    w = Watch()
    report = w.tick(98.0)  # exactly 2% of start-of-day equity
    assert report.flatten
    assert report.mark is not None
    assert report.mark.locked
    assert w.desk.engine.state.daily_loss_locked
    assert report.kill == ()
    assert not w.tick(97.0).flatten  # once per lock


@pytest.mark.spec("C8", "SPEC§10", "ARCH§8.3")
def test_a_10_percent_drawdown_between_bars_fires_the_kill() -> None:
    w = Watch()
    report = w.tick(90.0)  # exactly 10% from the 100,000 peak
    assert report.kill == (KillTrigger.DRAWDOWN,)
    assert w.desk.engine.state.halted
    assert w.desk.engine.state.killed
    assert report.flatten  # 10% is also past the 2% daily loss


@pytest.mark.spec("C8")
def test_a_stale_price_marks_nothing() -> None:
    w = Watch()
    before = w.desk.engine.state
    report = w.tick(50.0, price_age=120.000001)
    assert report.price_stale
    assert report.equity is None
    assert report.mark is None
    assert w.desk.engine.state == before
    assert report.kill == ()

    assert not w.tick(101.0, price_age=120.0).price_stale  # exactly the maximum is fresh
    assert w.tick(50.0, price_age=-1.0).price_stale  # from the future


@pytest.mark.spec("C8", "ARCH§8.1")
def test_a_stale_snapshot_marks_nothing_and_skips_the_stop() -> None:
    w = Watch()
    before = w.desk.engine.state
    report = w.tick(50.0, snapshot_age=6.0, orders=())
    assert report.reconciliation.status is ReconcileStatus.STALE
    assert report.mark is None
    assert not report.price_stale
    assert w.desk.engine.state == before
    assert report.stop_covered is None
    assert not report.replace_stop
    assert report.kill == ()


@pytest.mark.spec("C8")
def test_before_any_session_nothing_is_marked() -> None:
    w = Watch(desk(day=None))
    report = w.tick(50.0)
    assert report.mark is None
    assert report.equity is None
    assert not report.price_stale


@pytest.mark.spec("SPEC§10", "ARCH§8.3", "ARCH§16")
def test_a_reconciliation_mismatch_fires_the_kill() -> None:
    w = Watch()
    report = w.tick(100.0, book_position=900)
    assert report.reconciliation.mismatch
    assert report.kill == (KillTrigger.RECONCILIATION,)


@pytest.mark.spec("C8", "ARCH§8.3")
def test_an_equity_wiped_out_fires_the_drawdown_kill_without_a_mark() -> None:
    w = Watch()
    report = w.tick(100.0, cash=-100_000.0)
    assert report.mark is None
    assert report.kill == (KillTrigger.DRAWDOWN,)


@pytest.mark.spec("SPEC§10", "ARCH§16", "ARCH§5")
def test_a_missing_stop_is_re_placed_and_kills_after_5_minutes() -> None:
    w = Watch()
    start = w.desk.clock.now()
    first = w.tick(100.0, orders=())
    assert first.stop_covered is False
    assert first.replace_stop
    assert first.stop_missing_since == start
    assert first.kill == ()

    w.advance(299)
    assert w.tick(100.0, orders=()).kill == ()
    w.advance(1)
    late = w.tick(100.0, orders=())
    assert late.kill == (KillTrigger.STOP_MISSING,)
    assert late.stop_missing_since == start
    assert w.monitor.stop_missing_since == start

    w.advance(60)
    covered = w.tick(100.0)
    assert covered.stop_covered is True
    assert covered.stop_missing_since is None
    assert not covered.replace_stop


@pytest.mark.spec("SPEC§10", "ARCH§16")
@pytest.mark.parametrize(
    ("position", "orders", "covered"),
    [
        (SHARES, (STOP,), True),
        (
            SHARES,
            (
                open_order("s", Side.SELL, OrderKind.STOP, 600, 90.0),
                open_order("t", Side.SELL, OrderKind.STOP, 400, 89.0),
            ),
            True,
        ),
        (SHARES, (open_order("s", Side.SELL, OrderKind.STOP, 500, 90.0),), False),
        (SHARES, (open_order("x", Side.SELL, OrderKind.LIMIT, 500, 99.0),), False),
        (0, (), True),
    ],
    ids=["one stop", "two stops", "half covered", "half an exit", "flat"],
)
def test_a_stop_covers_the_position_only_with_stop_sells_for_every_share(
    position: int, orders: tuple[OpenOrder, ...], covered: bool
) -> None:
    report = Watch().tick(100.0, position=position, orders=orders, cash=EQUITY - position * 100.0)
    assert report.stop_covered is covered
    assert report.replace_stop is not covered
    assert not report.exit_covered


# --- B9c: a working exit counts as cover, for 5 minutes (C5, your call of 2026-10-14) -----------

EXIT = open_order("x-1", Side.SELL, OrderKind.LIMIT, SHARES, 99.9)
RED = pytest.mark.xfail(strict=True, reason="red:B9", raises=AssertionError)


@RED
@pytest.mark.spec("SPEC§10", "ARCH§16")
def test_a_limit_is_no_stop_but_a_working_exit_for_every_share_covers() -> None:
    # Before B9c this case read "a limit is no stop" and asked to re-place the stop; the
    # owner's C5 call makes a risk-approved exit for the full position count as cover.
    report = Watch().tick(100.0, orders=(EXIT,))
    assert report.stop_covered is False  # a limit is still no stop ...
    assert report.exit_covered  # ... but the exit covers every share
    assert not report.replace_stop
    assert report.stop_missing_since is None
    assert report.kill == ()


@RED
@pytest.mark.spec("SPEC§10", "ARCH§16")
def test_an_exit_with_a_stop_covering_the_rest_covers() -> None:
    orders = (
        open_order("x-1", Side.SELL, OrderKind.LIMIT, 400, 99.9),
        open_order("s", Side.SELL, OrderKind.STOP, 600, 90.0),
    )
    report = Watch().tick(100.0, orders=orders)
    assert report.exit_covered
    assert not report.replace_stop


@pytest.mark.spec("SPEC§10", "ARCH§16", "ARCH§8.1")
def test_an_exit_the_engine_doesnt_know_is_no_cover() -> None:
    w = Watch()
    now = w.desk.clock.now()
    report = w.monitor.tick(
        MinuteInput(
            price=100.0,
            price_at=now,
            cash=0.0,
            snapshot=BrokerSnapshot(
                taken_at=now, position=SHARES, equity=EQUITY, open_orders=(EXIT,)
            ),
            book=InternalBook(position=SHARES, open_orders=()),
        )
    )
    assert not report.exit_covered
    assert report.replace_stop
    assert KillTrigger.RECONCILIATION in report.kill


@RED
@pytest.mark.spec("SPEC§10", "ARCH§16", "ARCH§8.3")
def test_an_exit_still_working_after_5_minutes_fires_the_stop_missing_kill() -> None:
    w = Watch()
    start = w.desk.clock.now()
    first = w.tick(100.0, orders=(EXIT,))
    assert (first.exit_covered, first.exit_since, first.kill) == (True, start, ())
    w.advance(299)
    assert w.tick(100.0, orders=(EXIT,)).kill == ()
    # A re-quote is a new order: the clock runs from the first exit, not this one.
    requote = open_order("x-2", Side.SELL, OrderKind.LIMIT, SHARES, 99.9)
    w.advance(1)
    late = w.tick(100.0, orders=(requote,))
    assert late.kill == (KillTrigger.STOP_MISSING,)
    assert late.exit_since == start
    assert not late.replace_stop
    w.advance(60)
    covered = w.tick(100.0)  # a stop over every share ends the spell
    assert covered.stop_covered
    assert not covered.exit_covered
    assert (covered.exit_since, covered.stop_missing_s, covered.kill) == (None, None, ())


@RED
@pytest.mark.spec("SPEC§10", "ARCH§16")
def test_the_stop_missing_clock_pauses_while_an_exit_works() -> None:
    w = Watch()
    bare = w.tick(100.0, orders=())
    assert (bare.replace_stop, bare.stop_missing_s) == (True, 0.0)
    w.advance(120)
    paused = w.tick(100.0, orders=(EXIT,))  # 120 s bare, then the exit goes out
    assert not paused.replace_stop
    assert paused.stop_missing_s == 120.0
    w.advance(100)
    assert w.tick(100.0, orders=(EXIT,)).stop_missing_s == 120.0  # paused
    resumed = w.tick(100.0, orders=())  # the exit was cancelled: bare again
    assert (resumed.replace_stop, resumed.stop_missing_s, resumed.kill) == (True, 120.0, ())
    w.advance(179)
    assert w.tick(100.0, orders=()).kill == ()
    w.advance(1)
    fired = w.tick(100.0, orders=())
    assert fired.stop_missing_s == 300.0
    assert fired.kill == (KillTrigger.STOP_MISSING,)


# --- B9b review ------------------------------------------------------------------------------------


@pytest.mark.spec("C8", "SPEC§10")
def test_equity_is_summed_exactly_so_the_exact_2_and_10_percent_lines_hold() -> None:
    # B9b review R5: in floats, 18,828.80 + 192 x 412.35 is 98,000.00000000001.
    w = Watch()
    loss = w.tick(412.35, cash=18_828.80, position=192)
    assert loss.equity == 98_000.0
    assert loss.flatten
    assert w.desk.engine.state.daily_loss_locked

    w = Watch()
    drawdown = w.tick(412.35, cash=16_189.35, position=179)
    assert drawdown.equity == 90_000.0
    assert drawdown.kill == (KillTrigger.DRAWDOWN,)


@pytest.mark.spec("C8", "SPEC§10", "ARCH§16")
@pytest.mark.parametrize(
    ("price", "cash"),
    [
        pytest.param(
            100.0,
            math.nan,
            id="cash nan",
        ),
        pytest.param(
            100.0,
            math.inf,
            id="cash inf",
        ),
        pytest.param(
            100.0,
            -math.inf,
            id="cash -inf",
        ),
        pytest.param(
            math.nan,
            0.0,
            id="price nan",
        ),
        pytest.param(
            math.inf,
            0.0,
            id="price inf",
        ),
    ],
)
def test_a_cash_or_price_that_isnt_finite_is_a_data_fault_not_a_kill(
    price: float, cash: float
) -> None:
    # B9b review C4: a garbled broker field is bad data, not a wiped-out account.
    w = Watch()
    before = w.desk.engine.state
    report = w.tick(price, cash=cash)
    assert report.kill == ()
    assert report.mark is None
    assert report.equity is None
    assert w.desk.engine.state == before
    assert report.data_fault is not None
    assert report.stop_covered is True  # the stop is still checked
