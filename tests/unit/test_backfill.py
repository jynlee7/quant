"""B6: the backfill ingests one session per call, resumes after an interruption to exactly the rows
one straight run stores, and stamps HISTORY only while live trading has never read the store
(ARCHITECTURE §6.1, §6.2; BUILD_LOG B2's open items)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant.cli.main import app
from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.epoch import to_us
from quant.core.errors import DataError
from quant.data.actions import CorporateActions, Dividend, Split
from quant.data.alpaca_data import AlpacaMarketData
from quant.data.backfill import backfill
from quant.data.frames import MinuteFrame
from quant.data.market_db import HISTORY, NOW, BarStore, IngestSummary, Stamp
from quant.data.policy import Purpose
from tests.support.alpaca import (
    BARS,
    CORPORATE_ACTIONS,
    KEYS,
    AlpacaFake,
    FakeSource,
    Reply,
    alpaca_fake,
)
from tests.support.bars import SYMBOL, synthetic_days

OPERATE = Purpose.OPERATE
FIRST, LAST = date(2017, 11, 20), date(2017, 12, 1)  # nine sessions around Thanksgiving
AFTER = datetime(2017, 12, 4, 15, 0, tzinfo=UTC)  # every session in the span is available
XNYS = TradingCalendar.xnys()
SESSIONS = XNYS.sessions(FIRST, LAST)
FEED_DELAY = timedelta(minutes=15)

ACTIONS = CorporateActions(
    dividends=(Dividend(SYMBOL, date(2017, 11, 28), 1.25, date(2017, 11, 22), True),),
    splits=(),
)


@pytest.fixture(scope="module")
def minutes() -> MinuteFrame:
    return synthetic_days(FIRST, LAST, seed=6)


def digest(store: BarStore) -> str:
    """What a view at AFTER reads of everything stored: hourly bars, minutes and actions."""
    view = store.view(AFTER, as_of=AFTER)
    view.hourly(SYMBOL, purpose=OPERATE)
    view.minutes(SYMBOL, datetime(2017, 11, 20, tzinfo=UTC), AFTER, purpose=OPERATE)
    view.dividends(SYMBOL, FIRST, LAST, purpose=OPERATE)
    view.splits(SYMBOL, FIRST, LAST, purpose=OPERATE)
    return view.digest()


@pytest.mark.spec("ARCH§6.1", "ARCH§6.2")
def test_an_interrupted_backfill_resumes_to_the_rows_of_one_straight_run(
    minutes: MinuteFrame,
) -> None:
    clock = SimClock(AFTER)
    straight = BarStore.memory()
    backfill(straight, FakeSource(minutes, ACTIONS), SYMBOL, FIRST, LAST, clock=clock)

    resumed = BarStore.memory()
    failed = FakeSource(minutes, ACTIONS, fail_on=date(2017, 11, 27))
    with pytest.raises(DataError, match="planted"):
        backfill(resumed, failed, SYMBOL, FIRST, LAST, clock=clock)
    assert failed.fetched == SESSIONS[:5]  # four stored, the fifth failed

    again = FakeSource(minutes, ACTIONS)
    result = backfill(resumed, again, SYMBOL, FIRST, LAST, clock=clock)
    assert again.fetched == SESSIONS[4:]  # the stored sessions aren't fetched again
    assert result.fetched == tuple(SESSIONS[4:])
    assert result.skipped == 4
    assert result.last == LAST

    assert digest(resumed) == digest(straight)


@pytest.mark.spec("ARCH§6.2")
def test_each_session_is_ingested_in_one_call(
    minutes: MinuteFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BarStore.memory()
    calls: list[MinuteFrame] = []
    real = store.ingest_minutes

    def spy(symbol: str, frame: MinuteFrame, *, known_at: Stamp, source: str) -> IngestSummary:
        calls.append(frame)
        return real(symbol, frame, known_at=known_at, source=source)

    monkeypatch.setattr(store, "ingest_minutes", spy)
    backfill(store, FakeSource(minutes), SYMBOL, FIRST, LAST, clock=SimClock(AFTER))

    assert len(calls) == len(SESSIONS)
    for day, frame in zip(SESSIONS, calls, strict=True):
        session = XNYS.session(day)
        expected = (to_us(session.close) - to_us(session.open)) // 60_000_000
        assert len(frame) == expected  # the whole session, in one call
        assert int(frame.t[0]) == to_us(session.open)
        assert int(frame.t[-1]) < to_us(session.close)


@pytest.mark.spec("ARCH§6.2", "D2")
def test_history_rows_are_known_when_they_became_available(minutes: MinuteFrame) -> None:
    store = BarStore.memory()
    result = backfill(store, FakeSource(minutes), SYMBOL, FIRST, LAST, clock=SimClock(AFTER))
    assert result.stamp is HISTORY

    for i in (0, 389, len(minutes) - 1):
        (version,) = store.versions(SYMBOL, minutes.at(i))
        assert version.known_at == minutes.at(i) + timedelta(minutes=1) + FEED_DELAY
    bars = store.view(AFTER, as_of=AFTER).hourly(SYMBOL, purpose=OPERATE)
    assert len(bars) == 7 * 8 + 4  # eight full days and the half day
    assert all(bar.known_at == bar.available_at for bar in bars)


@pytest.mark.spec("ARCH§6.2")
def test_once_live_trading_has_read_the_store_late_data_is_known_now(
    minutes: MinuteFrame,
) -> None:
    clock = SimClock(datetime(2017, 11, 22, 3, 0, tzinfo=UTC))
    store = BarStore.memory(clock=clock)
    first = backfill(store, FakeSource(minutes), SYMBOL, FIRST, date(2017, 11, 21), clock=clock)
    assert first.stamp is HISTORY

    clock.advance_to(datetime(2017, 11, 28, 22, 0, tzinfo=UTC))
    live = store.view(clock.now(), as_of=NOW)  # live trading reads the store
    clock.advance(timedelta(minutes=5))
    late = backfill(store, FakeSource(minutes), SYMBOL, FIRST, date(2017, 11, 28), clock=clock)
    assert late.stamp is NOW
    assert late.fetched == tuple(XNYS.sessions(date(2017, 11, 22), date(2017, 11, 28)))

    (version,) = store.versions(SYMBOL, datetime(2017, 11, 22, 14, 30, tzinfo=UTC))
    assert version.known_at > live.as_of
    assert version.known_at <= clock.now()


@pytest.mark.spec("ARCH§6.1", "D2")
def test_by_default_the_backfill_stops_at_the_last_available_session(
    minutes: MinuteFrame,
) -> None:
    # 16:10 ET on 2017-11-22: its last minute is available at 16:15, so the 21st is the last.
    clock = SimClock(datetime(2017, 11, 22, 21, 10, tzinfo=UTC))
    source = FakeSource(minutes)
    result = backfill(BarStore.memory(), source, SYMBOL, FIRST, clock=clock)
    assert result.last == date(2017, 11, 21)
    assert source.fetched == [date(2017, 11, 20), date(2017, 11, 21)]


@pytest.mark.spec("ARCH§6.1")
def test_a_session_the_source_has_none_of_is_reported_empty(minutes: MinuteFrame) -> None:
    missing = XNYS.session(date(2017, 11, 21))
    t = minutes.t
    holed = minutes.select((t < to_us(missing.open)) | (t >= to_us(missing.close)))
    result = backfill(
        BarStore.memory(), FakeSource(holed), SYMBOL, FIRST, LAST, clock=SimClock(AFTER)
    )
    assert result.empty == (date(2017, 11, 21),)
    assert len(result.fetched) == len(SESSIONS)


@pytest.mark.spec("ARCH§6.1")
def test_a_split_in_the_feed_is_surfaced_by_the_backfill(minutes: MinuteFrame) -> None:
    split = CorporateActions(
        (), (Split(SYMBOL, date(2017, 11, 28), 2.0, date(2017, 11, 21), True),)
    )
    result = backfill(
        BarStore.memory(), FakeSource(minutes, split), SYMBOL, FIRST, LAST, clock=SimClock(AFTER)
    )
    assert result.actions.halt is not None
    assert "2017-11-28" in result.actions.halt


@pytest.fixture
def fake() -> Iterator[AlpacaFake]:
    yield from alpaca_fake()


@pytest.mark.spec("ARCH§6.1", "SPEC§2", "D2")
def test_the_alpaca_client_feeds_the_backfill(
    fake: AlpacaFake, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: pytest.fail("slept on the wall clock"))
    fake.serve(
        BARS,
        "bars_2017-11-24_page1.json",
        "bars_2017-11-24_page2.json",
        "bars_2017-11-24_page3.json",
    )
    fake.serve(CORPORATE_ACTIONS, Reply('{"corporate_actions":{},"next_page_token":null}'))
    clock = SimClock(datetime(2018, 6, 1, 15, 0, tzinfo=UTC))
    store = BarStore.memory()
    day = date(2017, 11, 24)
    result = backfill(store, AlpacaMarketData(KEYS, clock=clock), SYMBOL, day, day, clock=clock)

    assert result.fetched == (day,)
    tau = datetime(2017, 11, 27, 15, 0, tzinfo=UTC)
    view = store.view(tau, as_of=tau)
    frame = view.minutes(SYMBOL, datetime(2017, 11, 24, tzinfo=UTC), tau, purpose=OPERATE)
    assert len(frame) == 210
    bars = view.hourly(SYMBOL, purpose=OPERATE)
    assert len(bars) == 4  # a half day: three hourly bars and the 30-minute CLOSE bar
    assert all(bar.complete and bar.known_at == bar.available_at for bar in bars)


@pytest.mark.spec("I6", "ARCH§6.1")
def test_the_backfill_command_refuses_a_missing_env_file(tmp_path: Path) -> None:
    db = tmp_path / "market.db"
    result = CliRunner().invoke(
        app,
        ["data", "backfill", "--env-file", str(tmp_path / ".env"), "--db", str(db)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert ".env" in result.output
    assert not db.exists()
