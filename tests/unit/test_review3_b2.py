"""B2 third review: gaps the review of the final diff found (BUILD_LOG, B2)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.epoch import to_us
from quant.core.errors import ClockError, PolicyError
from quant.data import market_db
from quant.data.frames import MinuteFrame
from quant.data.market_db import BarStore
from quant.data.policy import DataPolicy, Purpose, Segments
from quant.store.sqlite import connect
from tests.support.bars import SYMBOL, Ticket, store_with_days, synthetic_days

NEW_YORK = ZoneInfo("America/New_York")
TRIAL = Ticket("t1", date(2019, 1, 1), date(2024, 12, 31))


# 1. A view says what it knows: tau in a backtest, the recorded as_of in a replay, NOW live.


@pytest.mark.spec("I1", "ARCH§6.2")
def test_a_view_must_say_what_it_knows() -> None:
    store = store_with_days(date(2019, 7, 2), date(2019, 7, 2))
    tau = datetime(2019, 7, 3, tzinfo=UTC)
    with pytest.raises(TypeError):
        store.view(tau)  # type: ignore[call-arg]
    assert len(store.view(tau, as_of=tau).hourly(SYMBOL, purpose=Purpose.OPERATE)) == 7


# 4. A read is checked over the days it can reach.


@pytest.mark.spec("A2", "C1", "ARCH§6.3")
def test_an_hourly_read_at_midnight_is_checked_over_the_days_before_it() -> None:
    store = store_with_days(date(2024, 12, 30), date(2024, 12, 31))
    midnight = datetime(2025, 1, 1, tzinfo=NEW_YORK)  # the lockbox's first instant
    view = store.view(midnight, as_of=midnight, ticket=TRIAL)
    assert len(view.hourly(SYMBOL, purpose=Purpose.SCORE)) == 7 + 7


# 5. An ingest reads and rebuilds only the sessions it touches.


@pytest.mark.spec("AD2", "ARCH§6.1", "ARCH§9.1")
def test_an_ingest_reads_only_the_sessions_it_touches(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = SimClock(datetime(2019, 7, 8, tzinfo=UTC))
    days = (date(2019, 7, 1), date(2019, 7, 3))
    store = store_with_days(*days, clock=clock)
    read: list[tuple[int, int]] = []
    latest = BarStore._latest_minutes

    def spy(self: BarStore, symbol: str, start: int, end: int) -> object:
        read.append((start, end))
        return latest(self, symbol, start, end)

    monkeypatch.setattr(BarStore, "_latest_minutes", spy)
    # A settle revises the first minute of 07-01 and the last of 07-03, and nothing on 07-02.
    minutes = synthetic_days(*days)
    ends = minutes.select(np.isin(np.arange(len(minutes)), [0, len(minutes) - 1]))
    revised = MinuteFrame(
        ends.t, ends.open, ends.high + 0.05, ends.low, ends.close + 0.05, ends.volume
    )
    store.ingest_minutes(SYMBOL, revised, known_at=market_db.NOW, source="settle")
    between = TradingCalendar.xnys().session(date(2019, 7, 2))
    assert read
    assert all(end <= to_us(between.open) or start >= to_us(between.close) for start, end in read)


# 6. A purpose named by a string is checked as the purpose it names.


@pytest.mark.spec("A2", "C1", "ARCH§6.3")
def test_a_purpose_named_by_a_string_is_checked_the_same() -> None:
    policy = DataPolicy(Segments(lockbox_end=date(2026, 10, 20)))
    with pytest.raises(PolicyError):  # forward data is never scored
        policy.check("score", date(2026, 11, 2), date(2026, 11, 30))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Purpose"):
        policy.check("peek", date(2016, 1, 4), date(2016, 1, 8))  # type: ignore[arg-type]


# 8. A NOW stamp raises the history floor only when it stamps something.


@pytest.mark.spec("I1", "ARCH§6.2")
def test_a_stamp_raises_the_history_floor_only_when_it_is_used() -> None:
    clock = SimClock(datetime(2019, 7, 5, tzinfo=UTC))
    store = BarStore.memory(clock=clock)
    first = synthetic_days(date(2019, 7, 1), date(2019, 7, 1))
    store.ingest_minutes(SYMBOL, first, known_at=market_db.HISTORY, source="backfill")
    store.ingest_minutes(SYMBOL, first, known_at=market_db.NOW, source="refetch")  # no change
    with pytest.raises(ClockError):  # a live view before its decision time is refused
        store.view(datetime(2019, 7, 8, tzinfo=UTC), as_of=market_db.NOW)
    # Neither stamp was used, so history may still load the next day.
    second = synthetic_days(date(2019, 7, 2), date(2019, 7, 2))
    summary = store.ingest_minutes(SYMBOL, second, known_at=market_db.HISTORY, source="backfill")
    assert summary.minutes.inserted == 390


# 9. The switch to WAL is retried on every kind of busy error.


@pytest.mark.spec("AD2", "ARCH§9.1")
def test_connect_retries_a_wal_switch_refused_while_another_connection_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Recovering(sqlite3.Connection):
        """Refuses the first switch to WAL, as SQLite does while another connection recovers
        the log after a crash."""

        refusals = 1

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
            if sql == "PRAGMA journal_mode=WAL" and Recovering.refusals:
                Recovering.refusals -= 1
                busy = sqlite3.OperationalError("database is locked")
                busy.sqlite_errorcode = sqlite3.SQLITE_BUSY_RECOVERY  # an extended busy code
                raise busy
            return super().execute(sql, *args)  # type: ignore[arg-type]

    plain = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: plain(*a, factory=Recovering, **k))
    conn = connect(tmp_path / "x.db")
    assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    conn.close()
