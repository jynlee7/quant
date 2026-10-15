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

import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final

import numpy as np
from alpaca.common.exceptions import APIError
from alpaca.common.rest import RESTClient
from requests.exceptions import RequestException

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.clock import Clock, ensure_utc
from quant.core.epoch import to_us
from quant.core.errors import DataError
from quant.core.orders import Quote
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
        if requests_per_minute <= 0 or max_retries < 0:
            raise ValueError("invalid rate or retry limit")
        self._clock = clock
        self._calendar = calendar or TradingCalendar.xnys()
        self._spacing = timedelta(microseconds=math.ceil(60_000_000 / requests_per_minute))
        self._next_request = clock.now()
        self._max_retries = max_retries
        self._trading_url = trading_url
        # Use the SDK transport directly: importing its data package also imports
        # alpaca.trading. Disable built-in wall-clock retries for every request.
        self._client = RESTClient(
            data_url,
            api_key=keys.key_id,
            secret_key=keys.secret_key,
            raw_data=True,
            retry_attempts=0,
        )

        # alpaca-py treats retry_attempts=0 as falsy and restores its default.
        self._client._retry = 0

    def session_minutes(self, symbol: str, day: date) -> MinuteFrame:
        """`symbol`'s raw SIP minutes of session `day`, starting in [open, close), in time
        order. Empty when Alpaca has none. Raises CalendarError when `day` isn't a session, and
        DataError when the response is malformed or the retries run out."""
        session = self._calendar.session(day)
        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "feed": FEED_BARS,
            "adjustment": ADJUSTMENT,
            "start": session.open.isoformat(),
            "end": session.close.isoformat(),
            "limit": "10000",
        }
        rows: list[tuple[int, float, float, float, float, float]] = []
        try:
            for page in self._pages("/stocks/bars", params):
                for row in (page.get("bars") or {}).get(symbol, []):
                    at = _timestamp(row["t"])
                    if session.open <= at < session.close:
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
            actions = actions_from_feed(page["corporate_actions"])
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
                if exc.status_code != 429 or attempt == self._max_retries:
                    raise DataError(f"Alpaca HTTP {exc.status_code}") from None
                response = exc._http_error.response
                try:
                    seconds = float(response.headers.get("Retry-After", "3"))
                    if not math.isfinite(seconds) or seconds < 0:
                        raise ValueError
                except (ValueError, TypeError):
                    raise DataError("invalid Retry-After") from None
                self._next_request = max(
                    self._next_request, self._clock.now() + timedelta(seconds=seconds)
                )
            except (RequestException, ValueError):
                raise DataError("Alpaca request failed") from None
        raise AssertionError("unreachable retry loop")

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
