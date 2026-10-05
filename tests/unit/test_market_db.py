"""B2: market.db, the first SQLite file, and the migration pattern the others reuse (AD2)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from quant.core.calendar import TradingCalendar
from quant.core.errors import ConfigError
from quant.data.market_db import MIGRATIONS, BarStore
from quant.data.synthetic import synthetic_minutes
from quant.store.sqlite import Migration, connect, migrate


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
    store = BarStore.memory()
    first = store.put_minutes("SPY", minutes, source="synthetic")
    assert (first.inserted, first.revised, first.unchanged) == (390, 0, 0)
    again = store.put_minutes("SPY", minutes, source="synthetic")
    assert (again.inserted, again.revised, again.unchanged) == (0, 0, 390)

    changed = minutes.with_close(10, minutes.close[10] + 0.01)
    known = datetime(2019, 7, 2, 22, 0, tzinfo=UTC)
    revision = store.put_minutes("SPY", changed, source="settle", known_at=known)
    assert (revision.inserted, revision.revised, revision.unchanged) == (0, 1, 389)
    versions = store.versions("SPY", minutes.at(10))
    assert [v.version for v in versions] == [1, 2]
    assert versions[1].known_at == known
    assert versions[0].close == minutes.close[10]


@pytest.mark.spec("AD2", "SPEC§2", "ARCH§6.2")
def test_dividends_and_splits_are_point_in_time() -> None:
    store = BarStore.memory()
    announced = datetime(2019, 6, 20, 20, 0, tzinfo=UTC)
    store.put_dividend("SPY", ex_date=date(2019, 6, 21), amount=1.4354, known_at=announced)
    store.put_split("SPY", ex_date=date(2030, 1, 2), ratio=2.0, known_at=announced)
    before = store.view(announced)
    after = store.view(announced + timedelta(microseconds=1))
    assert before.dividends("SPY", date(2019, 1, 1), date(2019, 12, 31)) == []
    assert after.dividends("SPY", date(2019, 1, 1), date(2019, 12, 31)) == [
        (date(2019, 6, 21), 1.4354)
    ]
    assert after.splits("SPY", date(2030, 1, 1), date(2030, 12, 31)) == [(date(2030, 1, 2), 2.0)]
