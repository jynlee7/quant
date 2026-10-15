"""B6a review, round 1: a red test for each confirmed finding (X1-X4 from Codex, R1-R9 from the
code-review skill, and the probe's 403 on a bars request inside the feed delay). X3's tests live
in test_settings.py, beside the tests they guard.
"""

from __future__ import annotations

import math
import typing
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import pytest
from requests.exceptions import ReadTimeout

from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.epoch import to_us
from quant.core.errors import DataError
from quant.data.actions import (
    CorporateActions,
    Dividend,
    Split,
    actions_from_announcements,
    store_actions,
)
from quant.data.alpaca_data import AlpacaMarketData
from quant.data.backfill import backfill
from quant.data.frames import MinuteFrame
from quant.data.market_db import HISTORY, BarStore
from quant.data.policy import Purpose
from quant.data.quality import quality_report
from tests.support.alpaca import (
    ANNOUNCEMENTS,
    BARS,
    CALENDAR,
    CORPORATE_ACTIONS,
    KEYS,
    LATEST_QUOTES,
    LATEST_TRADES,
    AlpacaFake,
    FakeSource,
    Reply,
    alpaca_fake,
    fixture,
    fixture_text,
)
from tests.support.bars import SYMBOL, synthetic_days

OPERATE = Purpose.OPERATE
XNYS = TradingCalendar.xnys()
FIRST, LAST = date(2017, 11, 20), date(2017, 12, 1)  # nine sessions around Thanksgiving
SESSIONS = XNYS.sessions(FIRST, LAST)
AFTER = datetime(2017, 12, 4, 15, 0, tzinfo=UTC)  # every session in the span is available
START = datetime(2018, 6, 1, 15, 0, tzinfo=UTC)
FULL_DAY = date(2016, 3, 1)
HALF_DAY = date(2017, 11, 24)  # 09:30-13:00 ET, 14:30-18:00Z
FEED_DELAY = timedelta(minutes=15)


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


@pytest.fixture(scope="module")
def minutes() -> MinuteFrame:
    return synthetic_days(FIRST, LAST, seed=6)


@pytest.fixture
def clock() -> SimClock:
    return SimClock(START)


@pytest.fixture
def fake(clock: SimClock) -> Iterator[AlpacaFake]:
    yield from alpaca_fake(clock)


@pytest.fixture(autouse=True)
def no_wall_clock_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(seconds: float) -> None:
        raise AssertionError(f"time.sleep({seconds}) on the wall clock")

    monkeypatch.setattr("time.sleep", forbidden)


def client(clock: SimClock, **kwargs: Any) -> AlpacaMarketData:
    return AlpacaMarketData(KEYS, clock=clock, **kwargs)


def at(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def limited(**headers: str) -> Reply:
    return Reply(fixture_text("rate_limited.json"), status=429, headers=headers)


# X1: a resume treated any stored minute as a finished session.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=AssertionError)
@pytest.mark.spec("ARCH§6.1", "ARCH§6.2")
def test_a_partly_stored_session_is_completed_not_skipped(minutes: MinuteFrame) -> None:
    clock = SimClock(AFTER)
    store = BarStore.memory(clock=clock)
    day = date(2017, 11, 21)
    session = XNYS.session(day)
    store.ingest_minutes(
        SYMBOL, minutes.select(minutes.t == to_us(session.open)), known_at=HISTORY, source="seed"
    )
    source = FakeSource(minutes)
    result = backfill(store, source, SYMBOL, FIRST, LAST, clock=clock)

    clock.advance(timedelta(minutes=1))
    view = store.view(clock.now(), as_of=clock.now())
    assert len(view.minutes(SYMBOL, session.open, session.close, purpose=OPERATE)) == 390
    assert day in source.fetched
    # History can't revise the hourly bar the stored minute built, so the session is written
    # NOW, after every other session and the actions went in as history, and reported.
    assert result.revised == (day,)
    other = XNYS.session(date(2017, 11, 22)).open
    (version,) = store.versions(SYMBOL, other)
    assert version.known_at == other + timedelta(minutes=1) + FEED_DELAY


# R9: a resume read every stored session in full to learn it wasn't empty.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=AssertionError)
@pytest.mark.spec("ARCH§6.1")
def test_a_resume_over_finished_sessions_reads_no_stored_minutes(
    minutes: MinuteFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = SimClock(AFTER)
    store = BarStore.memory()
    backfill(store, FakeSource(minutes), SYMBOL, FIRST, LAST, clock=clock)

    def full_read(self: BarStore, *args: object) -> object:
        raise AssertionError("a resume read a stored session in full")

    monkeypatch.setattr(BarStore, "_latest_minutes", full_read)
    again = FakeSource(minutes)
    result = backfill(store, again, SYMBOL, FIRST, LAST, clock=clock)
    assert again.fetched == []
    assert result.skipped == len(SESSIONS)


# X2, R1: an explicit end bypassed the close-plus-feed-delay boundary, and actions were asked
# for up to it.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=AssertionError)
@pytest.mark.spec("ARCH§6.1", "D2")
def test_an_explicit_end_stops_at_the_last_available_session(minutes: MinuteFrame) -> None:
    clock = SimClock(utc(2017, 11, 22, 15, 30))  # 10:30 ET, an hour into the 22nd
    for end in (date(2017, 11, 22), LAST):  # today, and a day in the future
        source = FakeSource(minutes)
        result = backfill(BarStore.memory(), source, SYMBOL, FIRST, end, clock=clock)
        assert source.fetched == [date(2017, 11, 20), date(2017, 11, 21)]
        assert result.last == date(2017, 11, 21)
        assert source.asked_actions == [(FIRST, date(2017, 11, 21))]


# The probe: a bars request whose end is under 15 minutes old gets 403 on the free plan.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=AssertionError)
@pytest.mark.spec("D2", "ARCH§18")
def test_a_bars_request_never_ends_inside_the_feed_delay(fake: AlpacaFake) -> None:
    clock = SimClock(utc(2017, 11, 24, 16, 0))  # 11:00 ET on the half day
    fake.serve(BARS, "bars_2017-11-24_missing_1015.json")
    minutes = client(clock).session_minutes("SPY", HALF_DAY)

    (seen,) = fake.requests_to(BARS)
    assert at(seen.query["end"]) <= clock.now() - FEED_DELAY
    # Only minutes available by now (start + 1 min + feed delay <= now): 14:30-15:44Z, less the
    # missing 15:15.
    assert minutes.at(len(minutes) - 1) == utc(2017, 11, 24, 15, 44)
    assert len(minutes) == 75 - 1


@pytest.mark.xfail(strict=True, reason="red:B6", raises=AssertionError)
@pytest.mark.spec("D2", "ARCH§18")
def test_no_bars_request_before_the_first_minute_is_available(fake: AlpacaFake) -> None:
    clock = SimClock(utc(2017, 11, 24, 14, 40))  # ten minutes after the open
    minutes = client(clock).session_minutes("SPY", HALF_DAY)
    assert fake.requests_to(BARS) == []
    assert len(minutes) == 0


# X4: a fully missing hour, and a missing session's hours, were absent from incomplete_bars.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=AssertionError)
@pytest.mark.spec("ARCH§6.1", "PLAN§3")
def test_every_missing_or_incomplete_hourly_slot_is_listed() -> None:
    first, last = date(2017, 11, 20), date(2017, 11, 22)
    minutes = synthetic_days(first, last, seed=11)
    withheld = XNYS.session(date(2017, 11, 21))
    t = minutes.t
    keep = ((t < to_us(withheld.open)) | (t >= to_us(withheld.close))) & (
        (t < to_us(utc(2017, 11, 22, 15, 30))) | (t >= to_us(utc(2017, 11, 22, 16, 30)))
    )
    store = BarStore.memory()
    store.ingest_minutes(SYMBOL, minutes.select(keep), known_at=HISTORY, source="synthetic")

    report = quality_report(store, SYMBOL, first, last)
    expected = [bar.t_open for bar in withheld.bars] + [utc(2017, 11, 22, 15, 30)]
    assert report.incomplete_bars == tuple(expected)


# R2: no HTTP timeout, so a stalled connection could hang the backfill or the engine.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=AssertionError)
@pytest.mark.spec("ARCH§6.1", "ARCH§18")
def test_every_request_has_a_connect_and_read_timeout(clock: SimClock, fake: AlpacaFake) -> None:
    fake.serve(BARS, "bars_2016-03-01.json")
    fake.serve(CALENDAR, "calendar_2017-11-20_2017-11-28.json")
    fake.serve(LATEST_QUOTES, "latest_quotes_iex.json")
    data = client(clock)
    data.session_minutes("SPY", FULL_DAY)
    data.calendar(date(2017, 11, 20), date(2017, 11, 28))
    data.latest_quote("SPY")

    assert len(fake.seen) == 3
    for seen in fake.seen:
        assert isinstance(seen.timeout, tuple), f"{seen.path}: no timeout"
        connect, read = seen.timeout
        assert 0 < connect < math.inf
        assert 0 < read < math.inf
    fake.serve(LATEST_TRADES, Reply(error=ReadTimeout("read timed out (planted)")))
    with pytest.raises(DataError):
        data.latest_trade("SPY")


# R3: BarStore.latest_minutes read stored prices with no known_at and no DataPolicy purpose.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=AssertionError)
@pytest.mark.spec("ARCH§6.2", "ARCH§6.3")
def test_stored_prices_are_read_only_through_a_view() -> None:
    raw = [
        name
        for name in dir(BarStore)
        if not name.startswith("_")
        and callable(getattr(BarStore, name))
        and typing.get_type_hints(getattr(BarStore, name)).get("return") is MinuteFrame
    ]
    assert raw == []


# R4: a 429 ignored X-RateLimit-Reset, refused an HTTP-date Retry-After, and gave up after
# 5 x 3 s, inside Alpaca's 60 s window.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=AssertionError)
@pytest.mark.spec("ARCH§6.1", "ARCH§18", "K6")
def test_a_429_waits_for_x_ratelimit_reset(clock: SimClock, fake: AlpacaFake) -> None:
    reset = int(START.timestamp()) + 42
    fake.serve(BARS, limited(**{"X-RateLimit-Reset": str(reset)}), "bars_2016-03-01.json")
    client(clock).session_minutes("SPY", FULL_DAY)

    _, second = fake.requests_to(BARS)
    assert second.at is not None
    assert second.at >= datetime.fromtimestamp(reset, UTC)
    assert clock.now() - START < timedelta(seconds=43)


@pytest.mark.xfail(strict=True, reason="red:B6", raises=DataError)
@pytest.mark.spec("ARCH§6.1", "ARCH§18", "K6")
def test_a_429_with_an_http_date_retry_after_waits_until_then(
    clock: SimClock, fake: AlpacaFake
) -> None:
    until = START + timedelta(seconds=20)
    fake.serve(BARS, limited(**{"Retry-After": format_datetime(until, usegmt=True)}))
    fake.serve(BARS, "bars_2016-03-01.json")
    minutes = client(clock).session_minutes("SPY", FULL_DAY)

    _, second = fake.requests_to(BARS)
    assert second.at is not None
    assert second.at >= until
    assert len(minutes) == 390


@pytest.mark.xfail(strict=True, reason="red:B6", raises=DataError)
@pytest.mark.spec("ARCH§6.1", "ARCH§18", "K6")
def test_429s_are_waited_out_for_a_full_rate_window(clock: SimClock, fake: AlpacaFake) -> None:
    # 429s with no wait headers at all, for the whole 60 s window.
    fake.serve(BARS, replace(limited(), until=START + timedelta(seconds=60)))
    fake.serve(BARS, "bars_2016-03-01.json")
    minutes = client(clock).session_minutes("SPY", FULL_DAY)

    assert len(minutes) == 390
    assert len(fake.requests_to(BARS)) <= 10  # backs off, rather than hammering
    assert clock.now() - START < timedelta(seconds=180)


# R8, a guard: alpaca-py's own retry (a fixed 3 s on the wall clock, Retry-After ignored) never
# runs. If the SDK retried, the fake would see a second request it has no reply for, and
# time.sleep is forbidden. It holds today; it keeps the override honest if the SDK changes.
@pytest.mark.spec("ARCH§18", "K6")
def test_the_sdk_makes_one_request_per_call_on_a_429(clock: SimClock, fake: AlpacaFake) -> None:
    fake.serve(BARS, limited(**{"Retry-After": "1"}))
    with pytest.raises(DataError, match="429"):
        client(clock, max_retries=0).session_minutes("SPY", FULL_DAY)
    assert len(fake.requests_to(BARS)) == 1


# R5: refetched actions under HISTORY raised on any correction, so every later run failed.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=DataError)
@pytest.mark.spec("ARCH§6.2")
def test_a_changed_action_under_history_is_reported_and_the_rest_still_store() -> None:
    store = BarStore.memory()
    september = Dividend(SYMBOL, date(2017, 9, 15), 1.23, date(2017, 9, 15), False)
    store_actions(store, SYMBOL, CorporateActions((september,), ()), known_at=HISTORY)
    store_actions(store, SYMBOL, CorporateActions((september,), ()), known_at=HISTORY)  # no-op

    changed = Dividend(SYMBOL, date(2017, 9, 15), 1.25, date(2017, 9, 15), False)
    december = Dividend(SYMBOL, date(2017, 12, 15), 1.35, date(2017, 12, 15), False)
    sync = store_actions(store, SYMBOL, CorporateActions((changed, december), ()), known_at=HISTORY)

    (conflict,) = sync.conflicts
    assert "2017-09-15" in conflict
    tau = utc(2018, 1, 2, 15, 0)
    assert store.view(tau, as_of=tau).dividends(
        SYMBOL, date(2017, 9, 1), date(2017, 12, 31), purpose=OPERATE
    ) == [(date(2017, 9, 15), 1.23), (date(2017, 12, 15), 1.35)]


# R5, proven on real data (VERIFY.md): five SPY announcements are declared after their ex-date.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=AssertionError)
@pytest.mark.spec("ARCH§6.2")
def test_an_announcement_declared_after_its_ex_date_falls_back_to_the_ex_date() -> None:
    (row,) = fixture("announcements_spy_2017-12_planted.json")
    late = {**row, "ex_date": "2020-09-18", "declaration_date": "2020-11-10", "cash": "1.339224"}
    actions = actions_from_announcements([late])
    assert actions.dividends == (
        Dividend(SYMBOL, date(2020, 9, 18), 1.339224, date(2020, 9, 18), False),
    )
    store_actions(BarStore.memory(), SYMBOL, actions, known_at=HISTORY)  # never an abort


# R6: a stock dividend in the announcements was rejected (cash 0) or would be stored as cash.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=DataError)
@pytest.mark.spec("ARCH§6.1", "ARCH§6.2")
def test_a_stock_dividend_is_never_cash_and_never_fails_the_batch() -> None:
    (cash,) = fixture("announcements_spy_2017-12_planted.json")
    stock = {
        **cash,
        "ca_sub_type": "stock",
        "cash": "0",
        "old_rate": "1",
        "new_rate": "1.05",
        "ex_date": "2017-11-28",
        "declaration_date": "2017-11-20",
    }
    actions = actions_from_announcements([cash, stock])
    assert [d.ex_date for d in actions.dividends] == [date(2017, 12, 15)]
    # More shares per share breaks raw prices just as a split does, so it halts like one.
    assert actions.splits == (Split(SYMBOL, date(2017, 11, 28), 1.05, date(2017, 11, 20), True),)


# R7: a page without its `corporate_actions` object raised a bare KeyError.
@pytest.mark.xfail(strict=True, reason="red:B6", raises=KeyError)
@pytest.mark.spec("ARCH§6.1")
def test_a_corporate_actions_page_without_its_object_is_a_data_error(
    clock: SimClock, fake: AlpacaFake
) -> None:
    fake.serve(CORPORATE_ACTIONS, Reply('{"next_page_token":null}'))
    with pytest.raises(DataError):
        client(clock).corporate_actions("SPY", date(2016, 1, 1), date(2018, 12, 31))


@pytest.mark.xfail(strict=True, reason="red:B6", raises=AttributeError)
@pytest.mark.spec("ARCH§6.1")
def test_announcements_that_are_not_a_list_are_a_data_error(
    clock: SimClock, fake: AlpacaFake
) -> None:
    fake.serve(ANNOUNCEMENTS, Reply('{"message":"not a list"}'))
    with pytest.raises(DataError):
        client(clock).announcements("SPY", date(2017, 11, 1), date(2017, 12, 31))
