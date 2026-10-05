"""B2 review: gaps the code review found in the store, the view and the policy (BUILD_LOG, B2)."""

from __future__ import annotations

import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.epoch import to_us
from quant.core.errors import DataError, PolicyError
from quant.data import market_db
from quant.data.frames import MinuteFrame
from quant.data.market_db import BarStore
from quant.data.policy import DataPolicy, Purpose
from quant.data.synthetic import synthetic_minutes
from quant.data.validate import validate_minutes
from quant.store import sqlite as store_sqlite
from tests.support.bars import Ticket, store_with_days

DAY = date(2019, 7, 2)
SYMBOL = "SPY"


def day_minutes() -> MinuteFrame:
    return synthetic_minutes(TradingCalendar.xnys(), DAY, DAY, seed=9)


@pytest.mark.spec("A2", "ARCH§6.3")
def test_every_read_must_name_its_purpose() -> None:
    store = BarStore.memory()
    view = store.view(datetime(2019, 7, 3, tzinfo=UTC))
    for read in (
        lambda: view.hourly(SYMBOL),  # type: ignore[call-arg]
        lambda: view.minutes(
            SYMBOL, datetime(2019, 7, 2, tzinfo=UTC), datetime(2019, 7, 3, tzinfo=UTC)
        ),  # type: ignore[call-arg]
        lambda: view.dividends(SYMBOL, DAY, DAY),  # type: ignore[call-arg]
    ):
        with pytest.raises(TypeError):
            read()


@pytest.mark.spec("I1", "ARCH§6.2")
def test_a_live_ingest_records_when_it_was_known_and_rebuilds_the_hourly_bar() -> None:
    late = datetime(2019, 7, 2, 14, 49, tzinfo=UTC)  # tau was 14:46; the bar arrived 3 minutes late
    clock = SimClock(late)
    store = BarStore.memory(clock=clock)
    minutes = day_minutes()
    first_bar = minutes.select(minutes.t < to_us(datetime(2019, 7, 2, 14, 30, tzinfo=UTC)))
    summary = store.ingest_minutes(SYMBOL, first_bar, known_at=market_db.NOW, source="live")
    assert (summary.minutes.inserted, summary.hourly.inserted) == (60, 1)
    tau = datetime(2019, 7, 2, 14, 46, tzinfo=UTC)
    assert (
        store.view(tau).hourly(SYMBOL, purpose=Purpose.OPERATE) == []
    )  # replay sees what live saw
    bar = store.view(tau, as_of=late + timedelta(seconds=1)).hourly(
        SYMBOL, purpose=Purpose.OPERATE
    )[0]
    assert bar.known_at == late

    # A revised last minute revises the hourly bar too, known at the same moment.
    settle = datetime(2019, 7, 2, 20, 45, tzinfo=UTC)
    clock.advance_to(settle)
    revised = first_bar.with_close(59, float(first_bar.close[59]) + 0.05)
    again = store.ingest_minutes(SYMBOL, revised, known_at=market_db.NOW, source="settle")
    assert (again.minutes.revised, again.hourly.revised) == (1, 1)
    later = store.view(settle + timedelta(seconds=1)).hourly(SYMBOL, purpose=Purpose.OPERATE)[0]
    assert later.close == float(first_bar.close[59]) + 0.05
    assert later.version == 2
    assert later.known_at == settle


@pytest.mark.spec("I1", "ARCH§6.2")
def test_history_is_stamped_on_purpose_and_revisions_must_move_forward() -> None:
    minutes = day_minutes()
    store = BarStore.memory(clock=SimClock(minutes.at(0)))
    store.ingest_minutes(SYMBOL, minutes, known_at=market_db.HISTORY, source="backfill")
    bar = store.view(datetime(2019, 7, 3, tzinfo=UTC)).hourly(SYMBOL, purpose=Purpose.OPERATE)[0]
    assert bar.known_at == bar.available_at
    with pytest.raises(DataError):  # a revision can't claim to be history
        store.ingest_minutes(
            SYMBOL,
            minutes.with_close(5, 1.0 + float(minutes.close[5])),
            known_at=market_db.HISTORY,
            source="x",
        )
    with pytest.raises(DataError):  # nor known before the version it replaces: it's 09:30
        store.ingest_minutes(
            SYMBOL,
            minutes.with_close(5, 1.0 + float(minutes.close[5])),
            known_at=market_db.NOW,
            source="x",
        )


@pytest.mark.spec("AD2", "ARCH§9.1")
def test_two_writers_revising_the_same_minute_both_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "market.db"
    minutes = day_minutes()
    clock = SimClock(datetime(2019, 7, 2, 21, tzinfo=UTC))
    BarStore.open(path).ingest_minutes(
        SYMBOL, minutes, known_at=market_db.HISTORY, source="backfill"
    )
    slow = BarStore._latest_minutes
    first_read = threading.Event()

    def read_slowly(self: BarStore, *args: object) -> object:
        rows = slow(self, *args)
        first_read.set()
        time.sleep(0.2)  # without a write lock around read and write, the second writer reads v1
        return rows

    monkeypatch.setattr(BarStore, "_latest_minutes", read_slowly)
    errors: list[BaseException] = []

    def revise(delta: float) -> None:
        try:
            BarStore.open(path, clock=clock).ingest_minutes(
                SYMBOL,
                minutes.with_close(7, float(minutes.close[7]) + delta),
                known_at=market_db.NOW,
                source="settle",
            )
        except BaseException as exc:
            errors.append(exc)

    # The second writer starts once the first is reading, so it reaches the lock while the
    # first holds it.
    first = threading.Thread(target=revise, args=(0.01,))
    second = threading.Thread(target=revise, args=(0.02,))
    first.start()
    assert first_read.wait(timeout=10)
    second.start()
    for thread in (first, second):
        thread.join()
    assert errors == []
    assert [v.version for v in BarStore.open(path).versions(SYMBOL, minutes.at(7))] == [1, 2, 3]


@pytest.mark.spec("AD2", "ARCH§9.1")
def test_two_processes_migrating_a_new_database_both_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read = store_sqlite._user_version

    def read_slowly(conn: object) -> int:
        version = read(conn)  # type: ignore[arg-type]
        time.sleep(0.2)
        return version

    monkeypatch.setattr(store_sqlite, "_user_version", read_slowly)
    errors: list[BaseException] = []

    def open_store() -> None:
        try:
            BarStore.open(tmp_path / "fresh.db").close()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


@pytest.mark.spec("SPEC§2", "ARCH§6.1")
def test_a_big_move_across_a_missing_minute_is_flagged() -> None:
    m = day_minutes()
    keep = np.ones(len(m), dtype=bool)
    keep[101] = False  # 11:11 is missing
    o, h, lo, c = (x.copy() for x in (m.open, m.high, m.low, m.close))
    for column in (o, h, lo, c):
        column[102:] *= 1.10
    gapped = MinuteFrame(m.t, o, h, lo, c, m.volume).select(keep)
    flags = validate_minutes(TradingCalendar.xnys(), gapped)
    assert [f.t for f in flags] == [m.at(102)]


@pytest.mark.spec("C11", "A2", "ARCH§6.3")
def test_scoring_dev_needs_a_trial_ticket_and_nothing_scores_before_the_data() -> None:
    policy = DataPolicy()
    policy.check(Purpose.SCORE, date(2016, 1, 4), date(2018, 12, 31))  # FIT needs no ticket
    with pytest.raises(PolicyError):
        policy.check(Purpose.SCORE, date(2019, 1, 2), date(2019, 3, 29))
    policy.check(
        Purpose.SCORE,
        date(2019, 1, 2),
        date(2019, 3, 29),
        ticket=Ticket("t1", date(2019, 1, 1), date(2024, 12, 31)),
    )
    with pytest.raises(PolicyError):
        policy.check(
            Purpose.SCORE,
            date(2019, 1, 2),
            date(2019, 3, 29),
            ticket=Ticket("t1", date(2019, 2, 1), date(2024, 12, 31)),
        )
    with pytest.raises(PolicyError):
        policy.check(Purpose.SCORE, date(2015, 12, 1), date(2015, 12, 31))


# Found while fixing the gaps above: behavior the fixes need, written red first.


@pytest.mark.spec("SPEC§2", "ARCH§6.1")
def test_an_ingest_refuses_bad_minutes_and_stores_nothing() -> None:
    m = day_minutes()
    close = m.close.copy()
    close[200] = np.nan
    store = BarStore.memory()
    with pytest.raises(DataError):
        store.ingest_minutes(
            SYMBOL,
            MinuteFrame(m.t, m.open, m.high, m.low, close, m.volume),
            known_at=market_db.HISTORY,
            source="backfill",
        )
    assert store.versions(SYMBOL, m.at(0)) == []


@pytest.mark.spec("SPEC§2", "ARCH§6.1")
def test_an_ingest_keeps_a_big_move_and_records_its_flag() -> None:
    m = day_minutes()
    o, h, lo, c = (x.copy() for x in (m.open, m.high, m.low, m.close))
    for column in (o, h, lo, c):
        column[100:] *= 1.06
    store = BarStore.memory()
    summary = store.ingest_minutes(
        SYMBOL,
        MinuteFrame(m.t, o, h, lo, c, m.volume),
        known_at=market_db.HISTORY,
        source="backfill",
    )
    assert summary.minutes.inserted == 390
    assert [(f.kind, f.t) for f in summary.flags] == [("move_over_5pct", m.at(100))]
    assert store.flags(SYMBOL) == list(summary.flags)


@pytest.mark.spec("I1", "ARCH§6.2")
def test_history_cant_backdate_a_minute_live_trading_missed() -> None:
    minutes = day_minutes()
    first_bar = minutes.select(minutes.t < to_us(datetime(2019, 7, 2, 14, 30, tzinfo=UTC)))
    tau = datetime(2019, 7, 2, 14, 46, tzinfo=UTC)
    fetched = tau + timedelta(seconds=20)
    clock = SimClock(fetched)
    store = BarStore.memory(clock=clock)
    # Live trading fetched the bar at tau without its last minute, so it saw an incomplete bar.
    store.ingest_minutes(
        SYMBOL, first_bar.select(np.arange(60) < 59), known_at=market_db.NOW, source="live"
    )
    with pytest.raises(DataError):  # as history, the minute would complete the bar before tau
        store.ingest_minutes(SYMBOL, first_bar, known_at=market_db.HISTORY, source="backfill")
    assert store.versions(SYMBOL, first_bar.at(59)) == []  # the refused ingest left nothing
    retry = tau + timedelta(minutes=1)
    clock.advance_to(retry)
    summary = store.ingest_minutes(SYMBOL, first_bar, known_at=market_db.NOW, source="live")
    assert (summary.minutes.inserted, summary.hourly.revised) == (1, 1)

    def bar_known_at(as_of: datetime) -> bool:
        view = store.view(tau, as_of=as_of)
        return view.hourly(SYMBOL, purpose=Purpose.OPERATE)[0].complete

    assert not bar_known_at(fetched + timedelta(seconds=1))
    assert bar_known_at(retry + timedelta(seconds=1))


@pytest.mark.spec("AD2", "ARCH§6.2")
def test_a_corporate_action_revision_must_be_known_after_the_version_it_replaces() -> None:
    clock = SimClock(datetime(2019, 6, 19, 20, tzinfo=UTC))
    store = BarStore.memory(clock=clock)
    ex_date, declared = date(2019, 6, 21), date(2019, 6, 19)
    store.put_dividend(
        SYMBOL, ex_date=ex_date, amount=1.4354, declared=declared, known_at=market_db.HISTORY
    )
    # As history it is known at the end of 2019-06-19 in New York, after the clock's reading.
    with pytest.raises(DataError):
        store.put_dividend(
            SYMBOL, ex_date=ex_date, amount=1.44, declared=declared, known_at=market_db.NOW
        )
    corrected = datetime(2019, 6, 20, 20, tzinfo=UTC)
    clock.advance_to(corrected)
    store.put_dividend(
        SYMBOL, ex_date=ex_date, amount=1.44, declared=declared, known_at=market_db.NOW
    )
    view = store.view(corrected + timedelta(seconds=1))
    assert view.dividends(SYMBOL, ex_date, ex_date, purpose=Purpose.OPERATE) == [(ex_date, 1.44)]


@pytest.mark.spec("C11", "A2", "ARCH§6.3")
def test_a_view_scores_dev_only_with_a_ticket_that_covers_the_read() -> None:
    store = store_with_days(date(2019, 1, 2), date(2019, 1, 3))
    tau = datetime(2019, 1, 4, tzinfo=UTC)
    ticket = Ticket("t1", date(2019, 1, 1), date(2024, 12, 31))
    assert len(store.view(tau, ticket=ticket).hourly(SYMBOL, purpose=Purpose.SCORE)) == 7 + 7
    short = Ticket("t2", date(2019, 1, 3), date(2024, 12, 31))
    for view in (store.view(tau), store.view(tau, ticket=short)):
        with pytest.raises(PolicyError):
            view.hourly(SYMBOL, purpose=Purpose.SCORE)
