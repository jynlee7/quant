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
`next_page_token` until it is null, including through a page with no bars. A request never ends
inside the feed delay: the free plan refuses (403) one whose `end` is less than 15 minutes old
(docs/VERIFY.md), so a session still under way is asked for only up to its last available
minute, and one with none available yet isn't asked for at all.

Every request has a connect and a read timeout, and a timeout is a DataError like any failed
request: nothing waits on the network forever.

Rate limits: requests are spaced to stay under `requests_per_minute` (200 a minute on the free
plan). A 429 waits as long as it says: its `Retry-After` (seconds or an HTTP date), else until
its `X-RateLimit-Reset` (epoch seconds), else a backoff that doubles from 1 s to a full 60 s
window. It tries again up to `max_retries` times, within `max_wait` of waiting in all, which
covers a full window, then raises DataError. All waiting goes through the Clock
(`Clock.sleep_until`), never `time.sleep`, so a test on a SimClock waits no wall time.
alpaca-py's own retry, which sleeps a fixed 3 s and ignores Retry-After, must not run: it is
given no status code to retry on.

alpaca-py's trading package belongs to the execution adapter alone (K4), so this module never
imports `alpaca.trading`: the calendar and the announcements come from the trading host by plain
GET.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Final

import numpy as np
import requests
from alpaca.common.exceptions import APIError
from alpaca.common.rest import RESTClient
from requests.exceptions import RequestException

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.clock import Clock, ensure_utc
from quant.core.epoch import to_us
from quant.core.errors import DataError
from quant.core.orders import Quote
from quant.core.schedule import Timing
from quant.core.settings import AlpacaKeys
from quant.data.actions import (
    CorporateActions,
    Dividend,
    Split,
    actions_from_announcements,
    actions_from_feed,
)
from quant.data.frames import MinuteFrame
from quant.data.validate import check_minutes

DATA_URL: Final = "https://data.alpaca.markets"
TRADING_URL: Final = "https://paper-api.alpaca.markets"

FEED_BARS: Final = "sip"
FEED_LIVE: Final = "iex"
ADJUSTMENT: Final = "raw"

TIMEOUT: Final = (5.0, 30.0)  # seconds: to connect, and between bytes of the answer
RATE_WINDOW: Final = timedelta(seconds=60)  # Alpaca's rate limit counts a minute's requests
MAX_WAIT: Final = timedelta(seconds=120)  # a call's waits for 429s, in all: two full windows

# No HTTP status is -1, so alpaca-py's RESTClient retries nothing itself (its public
# `retry_exception_codes`; `retry_attempts=0` is read as "use the default").
_NO_STATUS: Final = -1
_MINUTE: Final = timedelta(minutes=1)


class _TimeoutSession(requests.Session):
    """A requests session that gives every request a timeout unless it names one."""

    def __init__(self, timeout: tuple[float, float]) -> None:
        super().__init__()
        self._timeout = timeout

    def request(self, method: Any, url: Any, *args: Any, **kwargs: Any) -> requests.Response:
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self._timeout
        return super().request(method, url, *args, **kwargs)


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
        timing: Timing | None = None,
        requests_per_minute: int = 190,
        max_retries: int = 10,
        max_wait: timedelta = MAX_WAIT,
        timeout: tuple[float, float] = TIMEOUT,
        data_url: str = DATA_URL,
        trading_url: str = TRADING_URL,
    ) -> None:
        if requests_per_minute <= 0 or max_retries < 0 or max_wait < timedelta(0):
            raise ValueError("invalid rate or retry limit")
        if len(timeout) != 2 or not all(math.isfinite(t) and t > 0 for t in timeout):
            raise ValueError("timeouts must be finite and positive")
        self._clock = clock
        self._calendar = calendar or TradingCalendar.xnys()
        self._timing = timing or Timing()
        self._spacing = timedelta(microseconds=math.ceil(60_000_000 / requests_per_minute))
        self._next_request = clock.now()
        self._max_retries = max_retries
        self._max_wait = max_wait
        self._trading_url = trading_url
        # Use the SDK transport directly: importing its data package also imports
        # alpaca.trading. It retries no status itself; 429s are waited out here, on the Clock.
        self._client = RESTClient(
            data_url,
            api_key=keys.key_id,
            secret_key=keys.secret_key,
            raw_data=True,
            retry_exception_codes=[_NO_STATUS],
        )
        # The SDK's session sets no timeout and takes none, so it gets one that does. A test
        # checks every request carries the timeout, so this fails loudly if the SDK changes.
        self._client._session = _TimeoutSession(timeout)

    def session_minutes(self, symbol: str, day: date) -> MinuteFrame:
        """`symbol`'s raw SIP minutes of session `day`, starting in [open, close), in time
        order, as far as they are available now: a minute is, once it has closed and the feed
        delay has passed. Empty when Alpaca has none, or none is available yet. Raises
        CalendarError when `day` isn't a session, and DataError when the response is malformed
        or the retries run out."""
        session = self._calendar.session(day)
        # The last minute available now: it starts a minute and the feed delay before now.
        newest = (self._clock.now() - self._timing.feed_delay - _MINUTE).replace(
            second=0, microsecond=0
        )
        end = min(session.close, newest)
        if end < session.open:
            return MinuteFrame.empty()
        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "feed": FEED_BARS,
            "adjustment": ADJUSTMENT,
            "start": session.open.isoformat(),
            "end": end.isoformat(),
            "limit": "10000",
        }
        rows: list[tuple[int, float, float, float, float, float]] = []
        try:
            for page in self._pages("/stocks/bars", params):
                for row in (page.get("bars") or {}).get(symbol, []):
                    at = _timestamp(row["t"])
                    if session.open <= at < session.close and at <= newest:
                        rows.append(
                            (
                                to_us(at),
                                float(row["o"]),
                                float(row["h"]),
                                float(row["l"]),
                                float(row["c"]),
                                float(row["v"]),
                            )
                        )
            if not rows:
                return MinuteFrame.empty()
            rows.sort(key=lambda row: row[0])
            frame = MinuteFrame(
                np.array([r[0] for r in rows], dtype=np.int64),
                *(np.array([r[i] for r in rows], dtype=np.float64) for i in range(1, 6)),
            )
            check_minutes(self._calendar, frame)
            return frame
        except (KeyError, TypeError, ValueError, AttributeError):
            raise DataError("malformed minute bars") from None

    def latest_quote(self, symbol: str) -> Quote:
        """The latest IEX quote. Raises DataError when it has no bid or no ask (a zero price, as
        IEX reports outside its hours)."""
        try:
            row = self._get("/stocks/quotes/latest", {"symbols": symbol, "feed": FEED_LIVE})[
                "quotes"
            ][symbol]
            bid, ask = float(row["bp"]), float(row["ap"])
            if not math.isfinite(bid) or bid <= 0:
                raise DataError("quote has no valid bid")
            if not math.isfinite(ask) or ask <= 0:
                raise DataError("quote has no valid ask")
            return Quote(bid=bid, ask=ask, at=_timestamp(row["t"]))
        except (KeyError, TypeError, ValueError):
            raise DataError("malformed quote") from None

    def latest_trade(self, symbol: str) -> Trade:
        """The latest IEX trade."""
        try:
            row = self._get("/stocks/trades/latest", {"symbols": symbol, "feed": FEED_LIVE})[
                "trades"
            ][symbol]
            price, size = float(row["p"]), float(row["s"])
            if not math.isfinite(price) or price <= 0 or not math.isfinite(size) or size <= 0:
                raise DataError("invalid trade price or size")
            return Trade(price, size, _timestamp(row["t"]))
        except (KeyError, TypeError, ValueError):
            raise DataError("malformed trade") from None

    def calendar(self, start: date, end: date) -> list[CalendarDay]:
        """Alpaca's sessions from `start` to `end`, inclusive, from the trading host."""
        payload = self._get(
            "/calendar", {"start": start.isoformat(), "end": end.isoformat()}, trading=True
        )
        try:
            return [
                CalendarDay(
                    date.fromisoformat(row["date"]),
                    datetime.fromisoformat(f"{row['date']}T{row['open']}")
                    .replace(tzinfo=NEW_YORK)
                    .astimezone(UTC),
                    datetime.fromisoformat(f"{row['date']}T{row['close']}")
                    .replace(tzinfo=NEW_YORK)
                    .astimezone(UTC),
                )
                for row in payload
            ]
        except (KeyError, TypeError, ValueError):
            raise DataError("malformed calendar") from None

    def corporate_actions(self, symbol: str, start: date, end: date) -> CorporateActions:
        """Dividends and splits with ex-dates from `start` to `end`, inclusive, from the
        market-data feed (`actions_from_feed`)."""
        params = {
            "symbols": symbol,
            "types": "cash_dividend,forward_split,reverse_split",
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        dividends: list[Dividend] = []
        splits: list[Split] = []
        for page in self._pages("/corporate-actions", params, version="v1"):
            actions = actions_from_feed(page.get("corporate_actions"))
            dividends.extend(actions.dividends)
            splits.extend(actions.splits)
        return CorporateActions(
            tuple(sorted(dividends, key=lambda d: d.ex_date)),
            tuple(sorted(splits, key=lambda s: s.ex_date)),
        )

    def announcements(self, symbol: str, since: date, until: date) -> CorporateActions:
        """Dividend and split announcements from the trading host, with their declaration
        dates (`actions_from_announcements`)."""
        return actions_from_announcements(
            self._get(
                "/corporate_actions/announcements",
                {
                    "symbol": symbol,
                    "ca_types": "Dividend,Split",
                    "since": since.isoformat(),
                    "until": until.isoformat(),
                },
                trading=True,
            )
        )

    def _get(
        self, path: str, params: dict[str, str], *, trading: bool = False, version: str = "v2"
    ) -> Any:
        waited = timedelta(0)
        for attempt in range(self._max_retries + 1):
            self._clock.sleep_until(max(self._clock.now(), self._next_request))
            self._next_request = self._clock.now() + self._spacing
            try:
                return self._client.get(
                    path,
                    params,
                    api_version=version,
                    base_url=self._trading_url if trading else None,
                )
            except APIError as exc:
                if exc.status_code != 429:
                    raise DataError(f"Alpaca HTTP {exc.status_code}") from None
                if attempt == self._max_retries:
                    raise DataError(f"Alpaca HTTP 429 after {attempt} retries") from None
                wait = self._rate_wait(exc.response, attempt)
                waited += wait
                if waited > self._max_wait:
                    raise DataError(
                        f"Alpaca HTTP 429: waiting {wait} more would pass {self._max_wait}"
                    ) from None
                self._next_request = max(self._next_request, self._clock.now() + wait)
            except (RequestException, ValueError):
                raise DataError("Alpaca request failed or timed out") from None
        raise AssertionError("unreachable retry loop")

    def _rate_wait(self, response: Any, attempt: int) -> timedelta:
        """How long a 429 says to wait: its Retry-After, in seconds or as an HTTP date, else
        until its X-RateLimit-Reset, in epoch seconds, else a backoff doubling from 1 s to a
        full window. Raises DataError on a header it can't read."""
        headers: Mapping[str, str] = {} if response is None else response.headers
        now = self._clock.now()
        retry_after = headers.get("Retry-After")
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except ValueError:
                try:
                    when = parsedate_to_datetime(retry_after)
                except (TypeError, ValueError):
                    raise DataError("invalid Retry-After") from None
                if when.tzinfo is None:
                    raise DataError("invalid Retry-After") from None
                return max(timedelta(0), when - now)
            if not math.isfinite(seconds) or seconds < 0:
                raise DataError("invalid Retry-After")
            return timedelta(seconds=seconds)
        reset = headers.get("X-RateLimit-Reset")
        if reset is not None:
            try:
                epoch = float(reset)
                when = datetime.fromtimestamp(epoch, UTC)
            except (ValueError, OverflowError, OSError):
                raise DataError("invalid X-RateLimit-Reset") from None
            return max(timedelta(0), when - now)
        return min(timedelta(seconds=2**attempt), RATE_WINDOW)

    def _pages(
        self, path: str, params: dict[str, str], *, version: str = "v2"
    ) -> Iterator[dict[str, Any]]:
        params = dict(params)
        tokens: set[str] = set()
        while True:
            page = self._get(path, params, version=version)
            if not isinstance(page, dict):
                raise DataError("malformed page")
            yield page
            token = page.get("next_page_token")
            if token is None:
                break
            if not isinstance(token, str) or not token or token in tokens:
                raise DataError("invalid or repeated page token")
            tokens.add(token)
            params["page_token"] = token


def _timestamp(value: str) -> datetime:
    # fromisoformat truncates sub-microsecond digits rather than rounding them.
    return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
