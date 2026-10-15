"""B6: the Alpaca market-data client against recorded responses, faked at the requests layer.

It asks for raw SIP minutes explicitly (D2, spec §2), follows pages, keeps only regular hours,
leaves a missing minute missing, waits out a 429 on the Clock rather than the wall clock, stays
under the rate limit (ARCHITECTURE §6.1, §18), and reads the latest IEX quote and trade (C8) and
the calendar (ARCHITECTURE §5).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.epoch import MINUTE_US, from_us, to_us
from quant.core.errors import DataError
from quant.data.alpaca_data import AlpacaMarketData
from tests.support.alpaca import (
    BARS,
    CALENDAR,
    DATA_HOST,
    KEYS,
    LATEST_QUOTES,
    LATEST_TRADES,
    TRADING_HOST,
    AlpacaFake,
    Reply,
    alpaca_fake,
    fixture,
    fixture_text,
)

RED = pytest.mark.xfail(strict=True, reason="red:B6", raises=NotImplementedError)

START = datetime(2018, 6, 1, 15, 0, tzinfo=UTC)
HALF_DAY = date(2017, 11, 24)  # 09:30-13:00 ET, 14:30-18:00Z
FULL_DAY = date(2016, 3, 1)  # 09:30-16:00 ET, 14:30-21:00Z


@pytest.fixture
def clock() -> SimClock:
    return SimClock(START)


@pytest.fixture
def fake(clock: SimClock) -> Iterator[AlpacaFake]:
    yield from alpaca_fake(clock)


@pytest.fixture(autouse=True)
def no_wall_clock_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Waiting goes through the Clock (K6). alpaca-py's own retry sleeps 3 s on the wall clock
    and ignores Retry-After; it must never run."""

    def forbidden(seconds: float) -> None:
        raise AssertionError(f"time.sleep({seconds}) on the wall clock")

    monkeypatch.setattr("time.sleep", forbidden)


def client(clock: SimClock, **kwargs: int) -> AlpacaMarketData:
    return AlpacaMarketData(KEYS, clock=clock, **kwargs)


def at(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


@pytest.mark.spec("D2", "SPEC§2", "ARCH§6.1")
@RED
def test_a_session_is_asked_for_as_raw_sip_minutes(clock: SimClock, fake: AlpacaFake) -> None:
    fake.serve(BARS, "bars_2016-03-01.json")
    minutes = client(clock).session_minutes("SPY", FULL_DAY)

    (seen,) = fake.requests_to(BARS)
    assert seen.host == DATA_HOST
    assert seen.query["feed"] == "sip"
    assert seen.query["adjustment"] == "raw"
    assert seen.query["timeframe"] == "1Min"
    assert seen.query["symbols"] == "SPY"
    assert at(seen.query["start"]) == datetime(2016, 3, 1, 14, 30, tzinfo=UTC)
    # The keys travel as headers, never in the URL.
    assert seen.headers["APCA-API-KEY-ID"] == KEYS.key_id
    assert seen.headers["APCA-API-SECRET-KEY"] == KEYS.secret_key
    assert KEYS.secret_key not in seen.url

    # Regular hours only: the 16:00 ET minute Alpaca sends (its end is inclusive) is dropped.
    assert len(minutes) == 390
    assert minutes.at(0) == datetime(2016, 3, 1, 14, 30, tzinfo=UTC)
    assert minutes.at(389) == datetime(2016, 3, 1, 20, 59, tzinfo=UTC)
    # Raw prices pass through exactly, as float64.
    first = fixture("bars_2016-03-01.json")["bars"]["SPY"][0]
    assert minutes.open[0] == first["o"]
    assert minutes.high[0] == first["h"]
    assert minutes.low[0] == first["l"]
    assert minutes.close[0] == first["c"]
    assert minutes.volume[0] == first["v"]
    assert minutes.close.dtype == np.float64


@pytest.mark.spec("D2", "ARCH§6.1")
@RED
def test_pages_are_followed_by_their_token(clock: SimClock, fake: AlpacaFake) -> None:
    pages = [
        "bars_2017-11-24_page1.json",
        "bars_2017-11-24_page2.json",
        "bars_2017-11-24_page3.json",
    ]
    fake.serve(BARS, *pages)
    minutes = client(clock).session_minutes("SPY", HALF_DAY)

    seen = fake.requests_to(BARS)
    assert len(seen) == 3
    assert "page_token" not in seen[0].query
    assert seen[1].query["page_token"] == fixture(pages[0])["next_page_token"]
    assert seen[2].query["page_token"] == fixture(pages[1])["next_page_token"]
    for request in seen:  # every page asks for SIP and raw, not just the first
        assert request.query["feed"] == "sip"
        assert request.query["adjustment"] == "raw"

    # 211 minutes over three pages; the 13:00 ET minute after the half day's close is dropped.
    assert len(minutes) == 210
    assert minutes.at(0) == datetime(2017, 11, 24, 14, 30, tzinfo=UTC)
    assert minutes.at(209) == datetime(2017, 11, 24, 17, 59, tzinfo=UTC)
    assert (np.diff(minutes.t) == MINUTE_US).all()


@pytest.mark.spec("SPEC§2", "ARCH§6.1")
@RED
def test_a_missing_minute_stays_missing(clock: SimClock, fake: AlpacaFake) -> None:
    fake.serve(BARS, "bars_2017-11-24_missing_1015.json")
    minutes = client(clock).session_minutes("SPY", HALF_DAY)

    assert len(minutes) == 209
    times = {from_us(int(t)) for t in minutes.t}
    assert datetime(2017, 11, 24, 15, 15, tzinfo=UTC) not in times
    assert datetime(2017, 11, 24, 15, 14, tzinfo=UTC) in times
    assert datetime(2017, 11, 24, 15, 16, tzinfo=UTC) in times


@pytest.mark.spec("ARCH§6.1")
@RED
def test_an_empty_answer_is_an_empty_session(clock: SimClock, fake: AlpacaFake) -> None:
    fake.serve(BARS, "bars_empty.json")
    minutes = client(clock).session_minutes("SPY", HALF_DAY)
    assert len(minutes) == 0
    assert len(fake.requests_to(BARS)) == 1


@pytest.mark.spec("ARCH§6.1")
@RED
def test_an_empty_page_with_a_token_is_followed(clock: SimClock, fake: AlpacaFake) -> None:
    fake.serve(BARS, "bars_empty_page_with_token.json", "bars_2017-11-24_missing_1015.json")
    minutes = client(clock).session_minutes("SPY", HALF_DAY)

    seen = fake.requests_to(BARS)
    assert len(seen) == 2
    assert (
        seen[1].query["page_token"] == fixture("bars_empty_page_with_token.json")["next_page_token"]
    )
    assert len(minutes) == 209


@pytest.mark.spec("ARCH§6.1", "ARCH§18", "K6")
@RED
def test_a_429_waits_for_its_retry_after_on_the_clock(clock: SimClock, fake: AlpacaFake) -> None:
    limited = Reply(fixture_text("rate_limited.json"), status=429, headers={"Retry-After": "7"})
    fake.serve(BARS, limited, "bars_2016-03-01.json")
    t0 = clock.now()
    minutes = client(clock).session_minutes("SPY", FULL_DAY)

    seen = fake.requests_to(BARS)
    assert len(seen) == 2
    assert seen[0].at is not None
    assert seen[1].at is not None
    assert seen[1].at - seen[0].at >= timedelta(seconds=7)
    assert clock.now() - t0 < timedelta(seconds=8)  # waited for Retry-After, not much longer
    assert len(minutes) == 390


@pytest.mark.spec("ARCH§6.1", "ARCH§18")
@RED
def test_429s_past_the_retry_limit_raise_data_error(clock: SimClock, fake: AlpacaFake) -> None:
    limited = Reply(fixture_text("rate_limited.json"), status=429, headers={"Retry-After": "1"})
    fake.serve(BARS, limited, limited, limited)
    with pytest.raises(DataError, match=r"429|rate"):
        client(clock, max_retries=2).session_minutes("SPY", FULL_DAY)
    assert len(fake.requests_to(BARS)) == 3


@pytest.mark.spec("ARCH§6.1", "ARCH§18")
@RED
def test_requests_stay_under_the_rate_limit(clock: SimClock, fake: AlpacaFake) -> None:
    per_minute = 5
    fake.serve(LATEST_QUOTES, *["latest_quotes_iex.json"] * 11)
    data = client(clock, requests_per_minute=per_minute)
    for _ in range(11):
        data.latest_quote("SPY")

    times = [seen.at for seen in fake.requests_to(LATEST_QUOTES)]
    assert len(times) == 11
    for earlier, later in zip(times, times[per_minute:], strict=False):
        assert earlier is not None
        assert later is not None
        assert later - earlier >= timedelta(minutes=1)  # never 6 requests within a minute


@pytest.mark.spec("C8", "SPEC§2")
@RED
def test_the_latest_quote_is_iex(clock: SimClock, fake: AlpacaFake) -> None:
    fake.serve(LATEST_QUOTES, "latest_quotes_iex.json")
    quote = client(clock).latest_quote("SPY")

    (seen,) = fake.requests_to(LATEST_QUOTES)
    assert seen.host == DATA_HOST
    assert seen.query["feed"] == "iex"
    assert seen.query["symbols"] == "SPY"
    assert quote.bid == 271.49
    assert quote.ask == 271.52
    # Alpaca's nanoseconds, truncated to the microsecond.
    assert quote.at == datetime(2018, 6, 1, 14, 35, 12, 123456, tzinfo=UTC)


@pytest.mark.spec("C8")
@RED
def test_a_quote_with_no_ask_is_refused(clock: SimClock, fake: AlpacaFake) -> None:
    fake.serve(LATEST_QUOTES, "latest_quotes_iex_no_ask.json")
    with pytest.raises(DataError, match="ask"):
        client(clock).latest_quote("SPY")


@pytest.mark.spec("C8")
@RED
def test_the_latest_trade_is_iex(clock: SimClock, fake: AlpacaFake) -> None:
    fake.serve(LATEST_TRADES, "latest_trades_iex.json")
    trade = client(clock).latest_trade("SPY")

    (seen,) = fake.requests_to(LATEST_TRADES)
    assert seen.query["feed"] == "iex"
    assert trade.price == 271.5
    assert trade.size == 100
    assert trade.at == datetime(2018, 6, 1, 14, 35, 11, 987654, tzinfo=UTC)


@pytest.mark.spec("ARCH§5")
@RED
def test_the_calendar_comes_from_the_trading_host_and_agrees_with_xnys(
    clock: SimClock, fake: AlpacaFake
) -> None:
    fake.serve(CALENDAR, "calendar_2017-11-20_2017-11-28.json")
    days = client(clock).calendar(date(2017, 11, 20), date(2017, 11, 28))

    (seen,) = fake.requests_to(CALENDAR)
    assert seen.host == TRADING_HOST
    assert seen.query["start"] == "2017-11-20"
    assert seen.query["end"] == "2017-11-28"
    xnys = TradingCalendar.xnys()
    assert [day.session for day in days] == xnys.sessions(date(2017, 11, 20), date(2017, 11, 28))
    for day in days:
        assert (day.open, day.close) == (
            xnys.session(day.session).open,
            xnys.session(day.session).close,
        )
    half = next(day for day in days if day.session == HALF_DAY)
    assert to_us(half.close) == to_us(datetime(2017, 11, 24, 18, 0, tzinfo=UTC))
