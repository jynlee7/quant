"""Alpaca's HTTP API, faked at the `requests` layer alpaca-py uses, and a fake market-data source
for the backfill (B6).

The fixtures in tests/fixtures/alpaca/ are Alpaca response bodies. Recorded 2026-10-15 with the
paper account's keys, by a throwaway script that saved bodies only (no headers, so no request
ids), and checked to hold no key or account id:

- `bars_2017-11-24_page{1,2,3}.json`: SPY's SIP minutes, raw, for the 2017-11-24 half day
  (09:30-13:00 ET), asked for with `limit=100`, so three pages chained by `next_page_token`.
  The last page ends with the 13:00 ET minute: Alpaca's `end` is inclusive and the feed has
  extended hours.
- `bars_2016-03-01.json`: one full session in one page, 391 minutes with the 16:00 ET one.
- `bars_empty.json`: the answer for a holiday (2017-12-25).
- `corporate_actions_spy_2016_2018.json`: SPY's cash dividends from the v1 feed. It gives no
  declaration dates, and it has no 2018-06 dividend.
- `calendar_2017-11-20_2017-11-28.json`: the trading host's calendar over Thanksgiving 2017.

Hand-built, in the shapes recorded the same day:

- `bars_2017-11-24_missing_1015.json`: the half day in one page, without the 10:15 ET minute.
- `bars_empty_page_with_token.json`: a page with no bars that still names a next page.
- `rate_limited.json`: the body of a 429; tests set its Retry-After header.
- `latest_quotes_iex.json`, `latest_quotes_iex_no_ask.json`, `latest_trades_iex.json`: the
  latest IEX quote and trade, moved to 2018 with made-up prices. The no-ask quote has the zero
  ask IEX reported after hours.
- `corporate_actions_spy_splits_planted.json`: a 2-for-1 and a 1-for-10 split planted on SPY
  in 2017, in the feed's split shape (recorded from AAPL's 2020 split). SPY has never split.
- `announcements_spy_2017-12_planted.json`: SPY's 2017-12-15 dividend in the trading host's
  announcement shape, with a made-up declaration date. The host returned nothing for 2017.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import responses

from quant.core.calendar import TradingCalendar
from quant.core.clock import Clock
from quant.core.epoch import to_us
from quant.core.errors import DataError
from quant.core.settings import AlpacaKeys
from quant.data.actions import CorporateActions
from quant.data.frames import MinuteFrame

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "alpaca"
DATA_HOST = "https://data.alpaca.markets"
TRADING_HOST = "https://paper-api.alpaca.markets"

BARS = "/v2/stocks/bars"
LATEST_QUOTES = "/v2/stocks/quotes/latest"
LATEST_TRADES = "/v2/stocks/trades/latest"
CORPORATE_ACTIONS = "/v1/corporate-actions"
ANNOUNCEMENTS = "/v2/corporate_actions/announcements"
CALENDAR = "/v2/calendar"

# Test keys: long enough to be registered as secrets, and not shaped like real Alpaca keys, so
# only a registered value can redact them.
KEYS = AlpacaKeys(key_id="fake-key-id-b6-0123456789", secret_key="fake-secret-b6-abcdefghijkl")


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text()


@dataclass(frozen=True)
class Seen:
    """One request the fake served: its method, host, path, query and headers."""

    method: str
    host: str
    path: str
    query: dict[str, str]
    headers: Mapping[str, str]
    url: str
    at: datetime | None  # the fake's clock when the request arrived
    timeout: Any  # the timeout requests handed the transport: None means it could wait forever


@dataclass(frozen=True)
class Reply:
    """One answer. With `until`, it is served again for every request that arrives before that
    time on the fake's clock, then the next reply is. With `error`, the transport raises it, as
    requests does on a timeout or a dropped connection."""

    body: str = ""
    status: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    until: datetime | None = None
    error: Exception | None = None


class AlpacaFake:
    """Serves queued replies by path, in order, and records every request.

    A request to a path with nothing queued fails the test, so a client that asks for more than
    it should, or retries when it shouldn't, is caught. Network access stays blocked: responses
    answers before any socket opens.
    """

    def __init__(self, mock: responses.RequestsMock, clock: Clock | None = None) -> None:
        self._clock = clock
        self._queues: dict[str, deque[Reply]] = {}
        self.seen: list[Seen] = []
        pattern = re.compile(rf"^({re.escape(DATA_HOST)}|{re.escape(TRADING_HOST)})/.*")
        for method in (responses.GET, responses.POST, responses.DELETE, responses.PATCH):
            mock.add_callback(method, pattern, callback=self._answer)

    def serve(self, path: str, *replies: Reply | str) -> None:
        """Queue replies for `path`: a fixture file name, or a Reply."""
        queue = self._queues.setdefault(path, deque())
        for reply in replies:
            queue.append(reply if isinstance(reply, Reply) else Reply(fixture_text(reply)))

    def requests_to(self, path: str) -> list[Seen]:
        return [seen for seen in self.seen if seen.path == path]

    def _answer(self, request: Any) -> tuple[int, dict[str, str], str]:
        parts = urlsplit(request.url)
        seen = Seen(
            method=request.method,
            host=f"{parts.scheme}://{parts.netloc}",
            path=parts.path,
            query=dict(parse_qsl(parts.query)),
            headers=dict(request.headers),
            url=request.url,
            at=None if self._clock is None else self._clock.now(),
            timeout=getattr(request, "req_kwargs", {}).get("timeout"),
        )
        self.seen.append(seen)
        queue = self._queues.get(parts.path)
        while queue and queue[0].until is not None:
            if seen.at is None:
                raise AssertionError("a reply served until a time needs the fake's clock")
            if seen.at < queue[0].until:
                break
            queue.popleft()
        if not queue:
            raise AssertionError(f"unexpected request: {request.method} {parts.path}")
        reply = queue[0] if queue[0].until is not None else queue.popleft()
        if reply.error is not None:
            raise reply.error
        headers = {"Content-Type": "application/json", **reply.headers}
        return reply.status, headers, reply.body


def alpaca_fake(clock: Clock | None = None) -> Iterator[AlpacaFake]:
    """Body of a pytest fixture: an AlpacaFake inside a RequestsMock, stamping each request with
    `clock`'s time."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield AlpacaFake(mock, clock)


@dataclass
class FakeSource:
    """A MarketDataSource over given minutes. It records each session it is asked for, and
    `fail_on` makes one session's fetch fail the way a dropped connection would."""

    minutes: MinuteFrame
    actions: CorporateActions = field(default_factory=lambda: CorporateActions((), ()))
    fail_on: date | None = None
    fetched: list[date] = field(default_factory=list)
    asked_actions: list[tuple[date, date]] = field(default_factory=list)
    calendar: TradingCalendar = field(default_factory=TradingCalendar.xnys)

    def session_minutes(self, symbol: str, day: date) -> MinuteFrame:
        self.fetched.append(day)
        if day == self.fail_on:
            raise DataError(f"connection reset fetching {day} (planted)")
        session = self.calendar.session(day)
        t = self.minutes.t
        return self.minutes.select((t >= to_us(session.open)) & (t < to_us(session.close)))

    def corporate_actions(self, symbol: str, start: date, end: date) -> CorporateActions:
        self.asked_actions.append((start, end))
        return CorporateActions(
            tuple(d for d in self.actions.dividends if start <= d.ex_date <= end),
            tuple(s for s in self.actions.splits if start <= s.ex_date <= end),
        )
