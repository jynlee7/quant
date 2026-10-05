"""B2: market.db, the first SQLite file, and the migration pattern the others reuse (AD2)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.errors import ConfigError
from quant.data.market_db import HISTORY, MIGRATIONS, NOW, BarStore, WriteSummary
from quant.data.policy import Purpose
from quant.data.synthetic import synthetic_minutes
from quant.store.sqlite import Migration, connect, migrate
from tests.support.bars import with_close


@pytest.mark.spec("AD2", "ARCH§9.1")
def test_migrations_apply_once_and_refuse_a_newer_database(tmp_path: Path) -> None:
    path = tmp_path / "market.db"
    BarStore.open(path).close()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")
    conn.close()
    with pytest.raises(ConfigError):
        BarStore.open(path)


@pytest.mark.spec("AD2")
def test_a_failed_migration_leaves_the_database_as_it_was(tmp_path: Path) -> None:
    conn = connect(tmp_path / "x.db")
    good = Migration(1, ("CREATE TABLE a (x INTEGER)",))
    bad = Migration(2, ("CREATE TABLE b (x INTEGER)", "THIS IS NOT SQL"))
    assert migrate(conn, [good], name="x") == 1
    with pytest.raises(sqlite3.OperationalError):
        migrate(conn, [good, bad], name="x")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"a"}


@pytest.mark.spec("AD2")
def test_migrations_must_be_numbered_from_one_without_gaps(tmp_path: Path) -> None:
    conn = connect(tmp_path / "x.db")
    with pytest.raises(ConfigError):
        migrate(conn, [Migration(2, ("CREATE TABLE a (x INTEGER)",))], name="x")


@pytest.mark.spec("AD2", "ARCH§6.2")
def test_writes_are_idempotent_and_revisions_keep_every_version() -> None:
    cal = TradingCalendar.xnys()
    minutes = synthetic_minutes(cal, date(2019, 7, 2), date(2019, 7, 2), seed=5)
    known = datetime(2019, 7, 2, 22, 0, tzinfo=UTC)
    store = BarStore.memory(clock=SimClock(known))
    first = store.ingest_minutes("SPY", minutes, known_at=HISTORY, source="synthetic")
    assert (first.minutes, first.hourly) == (WriteSummary(390, 0, 0), WriteSummary(7, 0, 0))
    again = store.ingest_minutes("SPY", minutes, known_at=HISTORY, source="synthetic")
    assert (again.minutes, again.hourly) == (WriteSummary(0, 0, 390), WriteSummary(0, 0, 7))

    changed = with_close(minutes, 10, minutes.close[10] + 0.01)
    revision = store.ingest_minutes("SPY", changed, known_at=NOW, source="settle")
    assert revision.minutes == WriteSummary(0, 1, 389)
    versions = store.versions("SPY", minutes.at(10))
    assert [v.version for v in versions] == [1, 2]
    assert [v.source for v in versions] == ["synthetic", "settle"]
    assert versions[1].known_at == known
    assert versions[0].close == minutes.close[10]


@pytest.mark.spec("AD2", "SPEC§2", "ARCH§6.2")
def test_dividends_and_splits_are_point_in_time() -> None:
    announced = datetime(2019, 6, 20, 20, 0, tzinfo=UTC)
    clock = SimClock(announced)
    store = BarStore.memory(clock=clock)
    declared = date(2019, 6, 20)
    store.put_dividend(
        "SPY", ex_date=date(2019, 6, 21), amount=1.4354, declared=declared, known_at=NOW
    )
    clock.advance(timedelta(hours=1))
    store.put_split("SPY", ex_date=date(2030, 1, 2), ratio=2.0, declared=declared, known_at=NOW)
    just_after = announced + timedelta(microseconds=1)
    before = store.view(announced, as_of=announced)
    after = store.view(just_after, as_of=just_after)
    year = (date(2019, 1, 1), date(2019, 12, 31))
    assert before.dividends("SPY", *year, purpose=Purpose.OPERATE) == []
    assert after.dividends("SPY", *year, purpose=Purpose.OPERATE) == [(date(2019, 6, 21), 1.4354)]
    known = clock.now() + timedelta(microseconds=1)
    later = store.view(known, as_of=known)
    assert later.splits("SPY", date(2030, 1, 1), date(2030, 12, 31), purpose=Purpose.OPERATE) == [
        (date(2030, 1, 2), 2.0)
    ]
