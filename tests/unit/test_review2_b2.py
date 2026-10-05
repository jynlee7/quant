"""B2 second review: gaps the second code review found in the store, the view and the policy
(BUILD_LOG, B2)."""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.epoch import MINUTE_US, to_us
from quant.core.errors import ClockError, DataError, LockboxError, PolicyError
from quant.data import market_db
from quant.data.frames import MinuteFrame
from quant.data.market_db import BarStore
from quant.data.policy import DataPolicy, Purpose, Segments
from quant.data.synthetic import synthetic_minutes
from quant.store.sqlite import connect, transaction
from tests.support.bars import Ticket, store_with_days

DAY = date(2019, 7, 2)
DAY_OPEN = datetime(2019, 7, 2, 13, 30, tzinfo=UTC)  # 09:30 in New York
TAU = datetime(2019, 7, 2, 14, 46, tzinfo=UTC)  # the decision for the 09:30 bar
SYMBOL = "SPY"
OPERATE = Purpose.OPERATE
US = timedelta(microseconds=1)
TRIAL = Ticket("t1", date(2019, 1, 1), date(2024, 12, 31))


def red(failure: type[BaseException]) -> pytest.MarkDecorator:
    """Red until the review fix lands; `failure` is how each test fails before it (P2)."""
    return pytest.mark.xfail(strict=True, reason="red:B2", raises=failure)


DID_NOT_RAISE = pytest.fail.Exception


def day_minutes() -> MinuteFrame:
    return synthetic_minutes(TradingCalendar.xnys(), DAY, DAY, seed=9)


def hour(minutes: MinuteFrame, i: int) -> MinuteFrame:
    """The minutes of the day's i-th hourly bar."""
    start = to_us(DAY_OPEN) + i * 60 * MINUTE_US
    return minutes.select((minutes.t >= start) & (minutes.t < start + 60 * MINUTE_US))


def jumped(at: int) -> MinuteFrame:
    """The day's minutes with the price level 6% higher from minute `at` on."""
    m = day_minutes()
    o, h, lo, c = (x.copy() for x in (m.open, m.high, m.low, m.close))
    for column in (o, h, lo, c):
        column[at:] *= 1.06
    return MinuteFrame(m.t, o, h, lo, c, m.volume)


class SteppingClock:
    """A wall clock that NTP can step back."""

    def __init__(self, reading: datetime) -> None:
        self.reading = reading

    def now(self) -> datetime:
        return self.reading

    def sleep_until(self, when: datetime) -> None:
        self.reading = when


# 1. known_at is stamped by the store's clock under the write lock, and so is a live view's as_of.


@red(TypeError)
@pytest.mark.spec("I1", "AD2", "ARCH§6.2", "ARCH§9.1")
def test_an_ingest_is_known_from_when_it_holds_the_write_lock(tmp_path: Path) -> None:
    path = tmp_path / "market.db"
    called = TAU + timedelta(seconds=5)
    clock = SimClock(called)
    store = BarStore.open(path, clock=clock)
    first_bar = hour(day_minutes(), 0)
    blocker = connect(path)
    blocker.execute("BEGIN IMMEDIATE")  # another writer holds the lock
    calling = threading.Event()
    errors: list[BaseException] = []

    def ingest() -> None:
        try:
            calling.set()
            BarStore.open(path, clock=clock).ingest_minutes(
                SYMBOL, first_bar, known_at=market_db.NOW, source="live"
            )
        except BaseException as exc:
            errors.append(exc)

    writer = threading.Thread(target=ingest)
    writer.start()
    assert calling.wait(timeout=10)
    time.sleep(0.2)  # the writer is waiting for the lock
    locked = called + timedelta(seconds=5)
    clock.advance_to(locked)
    blocker.execute("COMMIT")
    blocker.close()
    writer.join()
    assert errors == []
    # Stamped when it took the lock, not when it was called: a live view stamped while it waited
    # can't find its rows known before its as_of.
    assert [v.known_at for v in store.versions(SYMBOL, first_bar.at(0))] == [locked]


@red(TypeError)
@pytest.mark.spec("I1", "ARCH§6.2")
def test_stamps_order_live_views_and_ingests_even_when_the_clock_steps_back() -> None:
    clock = SteppingClock(TAU + timedelta(seconds=30))
    store = BarStore.memory(clock=clock)
    first_bar = hour(day_minutes(), 0)
    # A live view, then an ingest at the same clock reading: the ingest is known after the view.
    live = store.view(TAU, as_of=market_db.NOW)
    store.ingest_minutes(SYMBOL, first_bar, known_at=market_db.NOW, source="live")
    assert live.hourly(SYMBOL, purpose=OPERATE) == []
    assert store.view(TAU, as_of=live.as_of).hourly(SYMBOL, purpose=OPERATE) == []
    # The clock steps back a minute. The next live view still comes after the ingest, and sees it.
    clock.reading -= timedelta(minutes=1)
    again = store.view(TAU, as_of=market_db.NOW)
    assert again.as_of > live.as_of
    assert len(again.hourly(SYMBOL, purpose=OPERATE)) == 1
    with pytest.raises(ClockError):  # a store without a clock can't stamp NOW
        BarStore.memory().view(TAU, as_of=market_db.NOW)


@red(TypeError)
@pytest.mark.spec("I1", "ARCH§6.2")
def test_history_cant_add_what_a_live_view_may_have_read() -> None:
    clock = SimClock(datetime(2019, 7, 3, 14, 46, tzinfo=UTC))
    store = BarStore.memory(clock=clock)
    store.view(clock.now(), as_of=market_db.NOW)  # live trading has read the store
    # As history, 2019-07-02 would be known before that view, which didn't see it.
    with pytest.raises(DataError):
        store.ingest_minutes(SYMBOL, day_minutes(), known_at=market_db.HISTORY, source="backfill")
    with pytest.raises(DataError):
        store.put_dividend(
            SYMBOL,
            ex_date=date(2019, 7, 5),
            amount=1.0,
            declared=date(2019, 7, 1),
            known_at=market_db.HISTORY,
        )
    summary = store.ingest_minutes(SYMBOL, day_minutes(), known_at=market_db.NOW, source="restart")
    assert summary.minutes.inserted == 390


# 2. Move flags compare against stored minutes, and a revision recomputes them.


@red(TypeError)
@pytest.mark.spec("SPEC§2", "ARCH§6.1")
def test_a_jump_against_a_stored_minute_is_flagged() -> None:
    clock = SimClock(TAU)
    store = BarStore.memory(clock=clock)
    m = jumped(60)  # 10:30, the first minute of the second bar
    store.ingest_minutes(SYMBOL, hour(m, 0), known_at=market_db.NOW, source="live")
    clock.advance(timedelta(hours=1))
    summary = store.ingest_minutes(SYMBOL, hour(m, 1), known_at=market_db.NOW, source="live")
    assert [f.t for f in summary.flags] == [m.at(60)]
    assert store.flags(SYMBOL) == list(summary.flags)


@red(TypeError)
@pytest.mark.spec("SPEC§2", "ARCH§6.1")
def test_a_revision_recomputes_the_flags_around_it_and_keeps_the_one_it_clears() -> None:
    clock = SimClock(datetime(2019, 7, 2, 21, tzinfo=UTC))
    store = BarStore.memory(clock=clock)
    m = jumped(60)
    store.ingest_minutes(SYMBOL, m, known_at=market_db.HISTORY, source="backfill")
    assert [f.t for f in store.flags(SYMBOL)] == [m.at(60)]
    # The settle lifts 10:29 to the new level, so the jump comes a minute earlier: into 10:29,
    # not into 10:30, which is in the next bar.
    one = m.select(np.arange(len(m)) == 59)
    settled = MinuteFrame(
        one.t, one.open * 1.06, one.high * 1.06, one.low * 1.06, one.close * 1.06, one.volume
    )
    summary = store.ingest_minutes(SYMBOL, settled, known_at=market_db.NOW, source="settle")
    assert [f.t for f in summary.flags] == [m.at(59)]
    assert [f.t for f in summary.cleared] == [m.at(60)]
    assert [f.t for f in store.flags(SYMBOL)] == [m.at(59)]
    # Clearing appends a version: what was flagged before the settle is still on record.
    assert [f.t for f in store.flags(SYMBOL, as_of=clock.now())] == [m.at(60)]


# 3. Backfilled dividends and splits are history, known from their declaration day.


@red(TypeError)
@pytest.mark.spec("AD2", "SPEC§2", "ARCH§6.2")
def test_a_backfilled_corporate_action_is_known_from_the_end_of_its_declaration_day() -> None:
    store = BarStore.memory()
    ex_date, declared = date(2019, 6, 21), date(2019, 6, 19)
    store.put_dividend(
        SYMBOL, ex_date=ex_date, amount=1.4354, declared=declared, known_at=market_db.HISTORY
    )
    store.put_split(
        SYMBOL,
        ex_date=date(2030, 1, 2),
        ratio=2.0,
        declared=date(2029, 12, 3),
        known_at=market_db.HISTORY,
    )
    midnight = datetime(2019, 6, 20, 4, tzinfo=UTC)  # the end of 2019-06-19 in New York
    june = (date(2019, 6, 1), date(2019, 6, 30))
    assert store.view(midnight).dividends(SYMBOL, *june, purpose=OPERATE) == []
    assert store.view(midnight + US).dividends(SYMBOL, *june, purpose=OPERATE) == [
        (ex_date, 1.4354)
    ]
    later = store.view(datetime(2029, 12, 5, tzinfo=UTC))
    assert later.splits(SYMBOL, date(2030, 1, 1), date(2030, 1, 31), purpose=OPERATE) == [
        (date(2030, 1, 2), 2.0)
    ]
    with pytest.raises(DataError):  # history can't revise an action
        store.put_dividend(
            SYMBOL, ex_date=ex_date, amount=1.44, declared=declared, known_at=market_db.HISTORY
        )
    with pytest.raises(DataError):  # nor declare one after its ex-date
        store.put_dividend(
            SYMBOL,
            ex_date=date(2019, 12, 20),
            amount=1.57,
            declared=date(2019, 12, 23),
            known_at=market_db.HISTORY,
        )


# 4. Calibration and Kelly edge estimates read forward data, and never the lockbox.


@red(AttributeError)
@pytest.mark.spec("C1", "C11", "A2", "ARCH§6.3")
def test_calibration_and_edge_estimates_read_fit_dev_with_a_ticket_and_forward() -> None:
    policy = DataPolicy(Segments(lockbox_end=date(2026, 10, 20)))
    calibrate = Purpose.CALIBRATE
    policy.check(calibrate, date(2016, 1, 4), date(2018, 12, 31))  # the fit-window backtest
    with pytest.raises(PolicyError):  # Kelly's shadow returns on DEV are strategy P&L (C11)
        policy.check(calibrate, date(2019, 1, 2), date(2024, 12, 31))
    policy.check(calibrate, date(2019, 1, 2), date(2024, 12, 31), ticket=TRIAL)
    policy.check(calibrate, date(2026, 10, 21), date(2026, 11, 30))  # then forward (ARCH §6.3)
    with pytest.raises(LockboxError):
        policy.check(calibrate, date(2024, 12, 1), date(2026, 11, 30), ticket=TRIAL)
    with pytest.raises(PolicyError):  # gates still score only FIT and DEV
        policy.check(Purpose.SCORE, date(2026, 10, 21), date(2026, 11, 30))


# 5. A bounded minute read is checked over the days it reads.


@red(LockboxError)
@pytest.mark.spec("A2", "C1", "ARCH§6.3")
def test_a_bounded_minute_read_is_checked_over_the_days_it_reads() -> None:
    store = store_with_days(date(2018, 12, 27), date(2018, 12, 28))
    start = datetime(2018, 12, 27, 14, 30, tzinfo=UTC)
    new_year = datetime(2019, 1, 1, 5, tzinfo=UTC)  # midnight in New York: DEV's first instant
    in_lockbox = store.view(datetime(2025, 3, 3, 15, tzinfo=UTC))
    assert len(in_lockbox.minutes(SYMBOL, start, new_year, purpose=Purpose.SCORE)) == 2 * 390
    # The end is exclusive, so this read stays in FIT and needs no trial ticket.
    in_dev = store.view(datetime(2019, 1, 3, 15, tzinfo=UTC))
    assert len(in_dev.minutes(SYMBOL, start, new_year, purpose=Purpose.SCORE)) == 2 * 390
    with pytest.raises(LockboxError):
        in_lockbox.minutes(SYMBOL, start, datetime(2025, 1, 3, tzinfo=UTC), purpose=Purpose.SCORE)


# 7. The last n bars, for n >= 0.


@red(DID_NOT_RAISE)
@pytest.mark.spec("I1", "ARCH§6.2")
def test_a_negative_count_of_bars_is_refused() -> None:
    view = store_with_days(DAY, DAY).view(datetime(2019, 7, 3, tzinfo=UTC))
    with pytest.raises(ValueError, match="last"):
        view.hourly(SYMBOL, purpose=OPERATE, last=-1)


# 8. A failed transaction raises its own error and leaves the connection usable.


@red(sqlite3.OperationalError)
@pytest.mark.spec("AD2", "ARCH§9.1")
def test_a_transaction_sqlite_rolled_back_itself_raises_its_own_error(tmp_path: Path) -> None:
    conn = connect(tmp_path / "x.db")
    conn.execute("CREATE TABLE a (x INTEGER PRIMARY KEY)")

    def insert_twice() -> None:
        with transaction(conn):
            conn.execute("INSERT INTO a VALUES (1)")
            # SQLite ends the transaction itself, as it does on an interrupt or a full disk.
            conn.execute("INSERT OR ROLLBACK INTO a VALUES (1)")

    with pytest.raises(sqlite3.IntegrityError):
        insert_twice()
    assert not conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM a").fetchone() == (0,)


# Found while verifying finding 8: written red first.


@red(AssertionError)
@pytest.mark.spec("AD2", "ARCH§9.1")
def test_a_failed_commit_rolls_back_and_frees_the_connection(tmp_path: Path) -> None:
    conn = connect(tmp_path / "x.db")
    conn.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE c (p INTEGER REFERENCES p (id) DEFERRABLE INITIALLY DEFERRED)")

    def insert_orphan() -> None:
        with transaction(conn):
            conn.execute("INSERT INTO c VALUES (1)")  # its key is checked at COMMIT

    with pytest.raises(sqlite3.IntegrityError):
        insert_orphan()
    assert not conn.in_transaction
    with transaction(conn):
        conn.execute("INSERT INTO p VALUES (1)")
