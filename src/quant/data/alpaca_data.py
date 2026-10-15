"""The Alpaca market-data client (spec §2, D2, ARCHITECTURE §6.1 and §18).

It reads, through alpaca-py's data clients and the `requests` session they use:

- historical SIP minute bars, raw: every request names `feed=sip` and `adjustment=raw`, so
  neither the account's plan nor a library default can change the feed or adjust the prices;
- the latest IEX quote and trade, which feed only the order collar and the minute monitor (C8),
  never a feature;
- the calendar, which the engine cross-checks against the exchange calendar (ARCHITECTURE §5);
- corporate actions (`quant.data.actions`).

Bars: `session_minutes` asks for one session, from its open to its close. Alpaca's `end` is
inclusive and the feed has extended hours, so the minute starting at the close comes back too;
only minutes starting in [open, close) are kept. A minute Alpaca doesn't have stays missing:
nothing is filled in, and the quality report counts it. Pages are followed by
`next_page_token` until it is null, including through a page with no bars.

Rate limits: requests are spaced to stay under `requests_per_minute` (200 a minute on the free
plan). A 429 waits for its `Retry-After` and tries again, up to `max_retries` times, then raises
DataError. All waiting goes through the Clock (`Clock.sleep_until`), never `time.sleep`, so a
test on a SimClock waits no wall time; alpaca-py's own retry, which sleeps a fixed 3 s and
ignores Retry-After, must not run.

alpaca-py's trading package belongs to the execution adapter alone (K4), so this module never
imports `alpaca.trading`: the calendar and the announcements come from the trading host by plain
GET.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

from quant.core.calendar import TradingCalendar
from quant.core.clock import Clock
from quant.core.orders import Quote
from quant.core.settings import AlpacaKeys
from quant.data.actions import CorporateActions
from quant.data.frames import MinuteFrame

DATA_URL: Final = "https://data.alpaca.markets"
TRADING_URL: Final = "https://paper-api.alpaca.markets"

FEED_BARS: Final = "sip"
FEED_LIVE: Final = "iex"
ADJUSTMENT: Final = "raw"


@dataclass(frozen=True)
class Trade:
    """The latest trade: its price, its size in shares, and its own time, UTC."""

    price: float
    size: float
    at: datetime


@dataclass(frozen=True)
class CalendarDay:
    """One session as Alpaca's calendar gives it, with its regular-hours open and close in
    UTC."""

    session: date
    open: datetime
    close: datetime


class AlpacaMarketData:
    """Market data for one Alpaca account. Implements the backfill's MarketDataSource."""

    def __init__(
        self,
        keys: AlpacaKeys,
        *,
        clock: Clock,
        calendar: TradingCalendar | None = None,
        requests_per_minute: int = 190,
        max_retries: int = 5,
        data_url: str = DATA_URL,
        trading_url: str = TRADING_URL,
    ) -> None:
        raise NotImplementedError("B6: AlpacaMarketData")

    def session_minutes(self, symbol: str, day: date) -> MinuteFrame:
        """`symbol`'s raw SIP minutes of session `day`, starting in [open, close), in time
        order. Empty when Alpaca has none. Raises CalendarError when `day` isn't a session, and
        DataError when the response is malformed or the retries run out."""
        raise NotImplementedError("B6: AlpacaMarketData.session_minutes")

    def latest_quote(self, symbol: str) -> Quote:
        """The latest IEX quote. Raises DataError when it has no bid or no ask (a zero price, as
        IEX reports outside its hours)."""
        raise NotImplementedError("B6: AlpacaMarketData.latest_quote")

    def latest_trade(self, symbol: str) -> Trade:
        """The latest IEX trade."""
        raise NotImplementedError("B6: AlpacaMarketData.latest_trade")

    def calendar(self, start: date, end: date) -> list[CalendarDay]:
        """Alpaca's sessions from `start` to `end`, inclusive, from the trading host."""
        raise NotImplementedError("B6: AlpacaMarketData.calendar")

    def corporate_actions(self, symbol: str, start: date, end: date) -> CorporateActions:
        """Dividends and splits with ex-dates from `start` to `end`, inclusive, from the
        market-data feed (`actions_from_feed`)."""
        raise NotImplementedError("B6: AlpacaMarketData.corporate_actions")

    def announcements(self, symbol: str, since: date, until: date) -> CorporateActions:
        """Dividend and split announcements from the trading host, with their declaration
        dates (`actions_from_announcements`)."""
        raise NotImplementedError("B6: AlpacaMarketData.announcements")
