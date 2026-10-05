"""B2 review: gaps the code review found in the store, the view and the policy (BUILD_LOG, B2)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from quant.core.calendar import TradingCalendar
from quant.core.epoch import to_us
from quant.core.errors import DataError, PolicyError
from quant.data import market_db
from quant.data.frames import MinuteFrame
from quant.data.market_db import BarStore
from quant.data.policy import DataPolicy, Purpose
from quant.data.synthetic import synthetic_minutes
from quant.data.validate import validate_minutes
from quant.store import sqlite as store_sqlite

DAY = date(2019, 7, 2)
SYMBOL = "SPY"


def red(failure: type[BaseException]) -> pytest.MarkDecorator:
    """Red until the review fix lands; `failure` is how each test fails before it (P2)."""
    return pytest.mark.xfail(strict=True, reason="red:B2", raises=failure)


DID_NOT_RAISE = pytest.fail.Exception


def day_minutes() -> MinuteFrame:
    return synthetic_minutes(TradingCalendar.xnys(), DAY, DAY, seed=9)


@red(DID_NOT_RAISE)
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


@red(AttributeError)
@pytest.mark.spec("I1", "ARCH§6.2")
def test_a_live_ingest_records_when_it_was_known_and_rebuilds_the_hourly_bar() -> None:
    store = BarStore.memory()
    minutes = day_minutes()
    first_bar = minutes.select(minutes.t < to_us(datetime(2019, 7, 2, 14, 30, tzinfo=UTC)))
    late = datetime(2019, 7, 2, 14, 49, tzinfo=UTC)  # tau was 14:46; the bar arrived 3 minutes late
    summary = store.ingest_minutes(SYMBOL, first_bar, known_at=late, source="live")
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
    revised = first_bar.with_close(59, float(first_bar.close[59]) + 0.05)
    again = store.ingest_minutes(SYMBOL, revised, known_at=settle, source="settle")
    assert (again.minutes.revised, again.hourly.revised) == (1, 1)
    later = store.view(settle + timedelta(seconds=1)).hourly(SYMBOL, purpose=Purpose.OPERATE)[0]
    assert later.close == float(first_bar.close[59]) + 0.05
    assert later.version == 2
    assert later.known_at == settle


@red(AttributeError)
@pytest.mark.spec("I1", "ARCH§6.2")
def test_history_is_stamped_on_purpose_and_revisions_must_move_forward() -> None:
    store = BarStore.memory()
    minutes = day_minutes()
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
    with pytest.raises(DataError):  # nor known before the version it replaces
        store.ingest_minutes(
            SYMBOL,
            minutes.with_close(5, 1.0 + float(minutes.close[5])),
            known_at=minutes.at(0),
            source="x",
        )


@red(AttributeError)
@pytest.mark.spec("AD2", "ARCH§9.1")
def test_two_writers_revising_the_same_minute_both_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "market.db"
    minutes = day_minutes()
    BarStore.open(path).ingest_minutes(
        SYMBOL, minutes, known_at=market_db.HISTORY, source="backfill"
    )
    slow = BarStore._latest_minutes

    def read_slowly(self: BarStore, *args: object) -> object:
        rows = slow(self, *args)
        time.sleep(0.2)  # without a write lock around read and write, both writers read v1 here
        return rows

    monkeypatch.setattr(BarStore, "_latest_minutes", read_slowly)
    errors: list[BaseException] = []

    def revise(delta: float, known: datetime) -> None:
        try:
            BarStore.open(path).ingest_minutes(
                SYMBOL,
                minutes.with_close(7, float(minutes.close[7]) + delta),
                known_at=known,
                source="settle",
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=revise, args=(0.01, datetime(2019, 7, 2, 21, tzinfo=UTC))),
        threading.Thread(target=revise, args=(0.02, datetime(2019, 7, 2, 22, tzinfo=UTC))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert [v.version for v in BarStore.open(path).versions(SYMBOL, minutes.at(7))] == [1, 2, 3]


@red(AttributeError)
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


@red(AssertionError)
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


@dataclass(frozen=True)
class _Ticket:
    trial_id: str
    first_day: date
    last_day: date


@red(DID_NOT_RAISE)
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
        ticket=_Ticket("t1", date(2019, 1, 1), date(2024, 12, 31)),
    )
    with pytest.raises(PolicyError):
        policy.check(
            Purpose.SCORE,
            date(2019, 1, 2),
            date(2019, 3, 29),
            ticket=_Ticket("t1", date(2019, 2, 1), date(2024, 12, 31)),
        )
    with pytest.raises(PolicyError):
        policy.check(Purpose.SCORE, date(2015, 12, 1), date(2015, 12, 31))
