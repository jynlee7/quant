"""Alpaca vendor checks for ARCHITECTURE §18 (B6b; a spike, exempt from red/green, PLAN P11).

    .venv/bin/python spikes/alpaca_probe.py           # checks 2-5, and a closed-market snapshot of 1
    .venv/bin/python spikes/alpaca_probe.py --live    # check 1, in regular hours only (~20 min)

1. SIP availability delay (D2). `--live` polls `feed=sip` minute bars for the last 30 minutes
   every 30 s for 20 minutes, with no `end` (the free plan then serves up to its own cut-off),
   and records, for each minute that first appears during the run, the lag between the
   minute's end and the poll that first returned it. Minutes already there at the first poll
   are left out: their lag is only bounded. It refuses outside regular hours and says when to
   rerun. Without `--live`, one request shows the newest SIP minute served now and what an
   explicit `end=now` gets.
2. Historical SIP: whole sessions from 2016 and from last week, `feed=sip`, `adjustment=raw`,
   counted against 390 minutes (210 on a half day), raw and through `AlpacaMarketData`.
3. Rate limits: the `X-RateLimit-*` headers of the requests above (no extra traffic), and the
   request count and time of a 2016 -> today backfill at the client's page size and spacing.
4. Corporate actions: SPY cash dividends from the data feed and the trading host's
   announcements, against SPY's quarterly schedule (ex-date: the third Friday of March, June,
   September and December).
5. Paper dividends: the account's DIV activities, and the types of activity it has at all.

The keys are read only through `quant.core.settings.load_settings`, which registers them as
secrets. Nothing here prints, logs or writes a key, an account number, an id or a request id:
only counts, dates, prices and the rate-limit headers are kept. The results go to
`var/probe/` (gitignored) as JSON; `docs/VERIFY.md` records what they mean.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.clock import WallClock
from quant.core.settings import AlpacaKeys, load_settings
from quant.data.alpaca_data import DATA_URL, TRADING_URL, AlpacaMarketData

SYMBOL = "SPY"
MINUTE = timedelta(minutes=1)
SPACING_S = 0.4  # at most 150 requests a minute from this probe, under the 200 limit
RATE_HEADERS = ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset")
CLIENT_RPM = 190  # AlpacaMarketData's default requests_per_minute
CLIENT_PAGE = 10_000  # AlpacaMarketData's bars page size
BACKFILL_START = date(2016, 1, 1)
LIVE_WINDOW = timedelta(minutes=30)
LIVE_POLL = timedelta(seconds=30)
LIVE_RUN = timedelta(minutes=20)
D2_DELAY = timedelta(minutes=16)

CLOCK = WallClock()


class Http:
    """Plain GETs with the account's keys in headers, spaced, keeping only rate-limit headers."""

    def __init__(self, keys: AlpacaKeys) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {"APCA-API-KEY-ID": keys.key_id, "APCA-API-SECRET-KEY": keys.secret_key}
        )
        self._last = 0.0
        self.rate: dict[str, list[dict[str, str]]] = {}
        self.latency_s: dict[str, list[float]] = {}
        self.calls: Counter[str] = Counter()

    def get(self, base: str, path: str, params: dict[str, str]) -> tuple[int, Any]:
        wait = self._last + SPACING_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        started = time.monotonic()
        response = self._session.get(base + path, params=params, timeout=30)
        self._last = time.monotonic()
        host = base.split("//", 1)[1]
        self.calls[host] += 1
        self.latency_s.setdefault(host, []).append(self._last - started)
        self.rate.setdefault(host, []).append(
            {h: response.headers[h] for h in RATE_HEADERS if h in response.headers}
        )
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code != 200:
            message = body.get("message") if isinstance(body, dict) else None
            return response.status_code, {"message": message}
        return 200, body


def bars(http: Http, params: dict[str, str]) -> tuple[int, list[dict[str, Any]], int, Any]:
    """Every page of SPY minute bars for `params`: (status, rows, pages, error)."""
    params = {"symbols": SYMBOL, "timeframe": "1Min", "limit": str(CLIENT_PAGE), **params}
    rows: list[dict[str, Any]] = []
    pages = 0
    while True:
        status, body = http.get(DATA_URL, "/v2/stocks/bars", params)
        if status != 200:
            return status, rows, pages, body
        pages += 1
        rows.extend((body.get("bars") or {}).get(SYMBOL, []))
        token = body.get("next_page_token")
        if not token:
            return status, rows, pages, None
        params["page_token"] = token


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def ny(value: datetime) -> str:
    return value.astimezone(NEW_YORK).strftime("%Y-%m-%d %H:%M %Z")


# --- 1. SIP delay ------------------------------------------------------------------------


def sip_snapshot(http: Http) -> dict[str, Any]:
    """Outside regular hours: the newest SIP minute served now (extended hours included), and
    what an explicit end=now gets. Not the D2 answer: pre-market minutes are thin."""
    now = CLOCK.now()
    start = (now - timedelta(hours=2)).isoformat()
    status, rows, _, error = bars(http, {"feed": "sip", "adjustment": "raw", "start": start})
    newest = max((utc(r["t"]) for r in rows), default=None)
    explicit, _, _, explicit_error = bars(
        http, {"feed": "sip", "adjustment": "raw", "start": start, "end": now.isoformat()}
    )
    return {
        "asked_at": now.isoformat(),
        "no_end": {
            "status": status,
            "error": error,
            "minutes": len(rows),
            "newest_minute_start": newest.isoformat() if newest else None,
            "newest_minute_end_lag_min": round((now - newest - MINUTE).total_seconds() / 60, 2)
            if newest
            else None,
        },
        "end_now": {"status": explicit, "error": explicit_error},
    }


def live_window(calendar: TradingCalendar, now: datetime) -> tuple[bool, datetime]:
    """Whether a 20-minute run fits now, with 30 minutes of the session behind it; else the
    earliest time a run can start."""
    today = now.astimezone(NEW_YORK).date()
    if calendar.is_session(today):
        session = calendar.session(today)
        first, last = session.open + LIVE_WINDOW, session.close - LIVE_RUN - MINUTE
        if first <= now <= last:
            return True, now
        if now < first:
            return False, first
    return False, calendar.session(calendar.next_session(today)).open + LIVE_WINDOW


def sip_live(http: Http, calendar: TradingCalendar) -> dict[str, Any]:
    now = CLOCK.now()
    ok, when = live_window(calendar, now)
    if not ok:
        latest = calendar.session(when.astimezone(NEW_YORK).date()).close - LIVE_RUN - MINUTE
        print(
            f"--live needs regular trading hours, with 30 minutes of the session behind it and "
            f"20 ahead (now {ny(now)}). Rerun between {ny(when)} and {ny(latest)}."
        )
        raise SystemExit(2)
    session = calendar.session(now.astimezone(NEW_YORK).date())
    first_seen: dict[datetime, datetime] = {}
    values: dict[datetime, tuple[Any, ...]] = {}
    revised: set[datetime] = set()
    present_at_start: set[datetime] = set()
    polls: list[dict[str, Any]] = []
    errors: list[Any] = []
    end_at = now + LIVE_RUN
    poll = 0
    while CLOCK.now() <= end_at:
        asked = CLOCK.now()
        status, rows, _, error = bars(
            http,
            {"feed": "sip", "adjustment": "raw", "start": (asked - LIVE_WINDOW).isoformat()},
        )
        seen = CLOCK.now()
        if status != 200:
            errors.append({"at": asked.isoformat(), "status": status, "error": error})
        newest = None
        for row in rows:
            t = utc(row["t"])
            if not session.open <= t < session.close:
                continue
            newest = t if newest is None or t > newest else newest
            ohlcv = (row["o"], row["h"], row["l"], row["c"], row["v"])
            if t not in first_seen:
                first_seen[t] = seen
                values[t] = ohlcv
                if poll == 0:
                    present_at_start.add(t)
            elif values[t] != ohlcv:
                revised.add(t)
                values[t] = ohlcv
        polls.append(
            {
                "asked": asked.isoformat(),
                "status": status,
                "newest_minute_start": newest.isoformat() if newest else None,
                "newest_end_lag_min": round((seen - newest - MINUTE).total_seconds() / 60, 2)
                if newest
                else None,
            }
        )
        print(f"poll {poll}: {status}, newest minute {newest}", flush=True)
        poll += 1
        CLOCK.sleep_until(asked + LIVE_POLL)
    lags = sorted(
        (seen - t - MINUTE).total_seconds() / 60
        for t, seen in first_seen.items()
        if t not in present_at_start
    )
    worst = max(lags) if lags else None
    # The poll interval bounds the measurement: a lag is at most 0.5 min above the true one.
    holds = None if worst is None else worst <= (D2_DELAY - LIVE_POLL).total_seconds() / 60
    return {
        "session": session.day.isoformat(),
        "started": now.isoformat(),
        "polls": polls,
        "errors": errors,
        "new_minutes": len(lags),
        "lag_min": {
            "min": min(lags) if lags else None,
            "median": statistics.median(lags) if lags else None,
            "max": worst,
        },
        "revised_after_first_seen": sorted(t.isoformat() for t in revised),
        "d2_16min_holds": holds,
    }


# --- 2. Historical sessions -------------------------------------------------------------------


def last_week(today: date) -> list[date]:
    monday = today - timedelta(days=today.weekday() + 7)
    return [monday + timedelta(days=i) for i in range(5)]


def sessions_check(
    http: Http, client: AlpacaMarketData, calendar: TradingCalendar
) -> list[dict[str, Any]]:
    days = [
        date(2016, 1, 4),  # the first session of the backfill
        date(2016, 3, 1),
        date(2016, 11, 25),  # half day, the day after Thanksgiving
        date(2016, 12, 30),
        date(2025, 11, 28),  # a recent half day
        *last_week(CLOCK.now().astimezone(NEW_YORK).date()),
    ]
    out = []
    for day in days:
        if not calendar.is_session(day):
            out.append({"day": day.isoformat(), "session": False})
            continue
        session = calendar.session(day)
        expected = int((session.close - session.open) / MINUTE)
        status, rows, pages, error = bars(
            http,
            {
                "feed": "sip",
                "adjustment": "raw",
                "start": session.open.isoformat(),
                "end": session.close.isoformat(),
            },
        )
        stamps = [utc(r["t"]) for r in rows]
        inside = sorted({t for t in stamps if session.open <= t < session.close})
        grid = {session.open + i * MINUTE for i in range(expected)}
        missing = sorted(grid - set(inside))
        via_client = len(client.session_minutes(SYMBOL, day).t)
        out.append(
            {
                "day": day.isoformat(),
                "half_day": session.is_half_day,
                "status": status,
                "error": error,
                "pages": pages,
                "raw_rows": len(rows),
                "outside_open_close": len(stamps) - len([t for t in stamps if t in grid]),
                "duplicates": len(stamps) - len(set(stamps)),
                "expected": expected,
                "regular_minutes": len(inside),
                "missing": [ny(t) for t in missing],
                "client_minutes": via_client,
                "volume": sum(float(r["v"]) for r in rows if utc(r["t"]) in grid),
                "complete": len(inside) == expected and via_client == expected,
            }
        )
    return out


# --- 3. Rate limits ------------------------------------------------------------------------


def rate_check(http: Http, calendar: TradingCalendar) -> dict[str, Any]:
    today = CLOCK.now().astimezone(NEW_YORK).date()
    sessions = calendar.sessions(BACKFILL_START, today - timedelta(days=1))
    data_host = DATA_URL.split("//", 1)[1]
    latency = statistics.median(http.latency_s.get(data_host, [0.0]))
    per_request_s = max(60 / CLIENT_RPM, latency)
    # One page a session: a full session is 391 raw minutes (the close minute comes back), far
    # under the page size, as the sessions check confirms.
    return {
        "headers": {
            host: {
                "limit": sorted({r.get("X-RateLimit-Limit", "absent") for r in rows}),
                "remaining_seen": [r.get("X-RateLimit-Remaining") for r in rows],
                "reset_present": any("X-RateLimit-Reset" in r for r in rows),
            }
            for host, rows in http.rate.items()
        },
        "calls": dict(http.calls),
        "median_latency_s": {h: round(statistics.median(v), 3) for h, v in http.latency_s.items()},
        "backfill": {
            "sessions": len(sessions),
            "first": sessions[0].isoformat(),
            "last": sessions[-1].isoformat(),
            "requests": len(sessions),
            "approx_minutes_of_bars": len(sessions) * 390,
            "client_rpm": CLIENT_RPM,
            "estimated_minutes": round(len(sessions) * per_request_s / 60, 1),
        },
    }


# --- 4. Dividends -------------------------------------------------------------------------


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def dividends_check(http: Http) -> dict[str, Any]:
    today = CLOCK.now().astimezone(NEW_YORK).date()
    feed: list[dict[str, Any]] = []
    fields: set[str] = set()
    params = {
        "symbols": SYMBOL,
        "types": "cash_dividend",
        "start": BACKFILL_START.isoformat(),
        "end": today.isoformat(),
        "limit": "1000",
    }
    feed_status = 200
    feed_error = None
    while True:
        status, body = http.get(DATA_URL, "/v1/corporate-actions", params)
        if status != 200:
            feed_status, feed_error = status, body
            break
        for row in (body.get("corporate_actions") or {}).get("cash_dividends", []):
            fields.update(row)
            feed.append({k: row.get(k) for k in ("ex_date", "rate", "declaration_date")})
        if not body.get("next_page_token"):
            break
        params["page_token"] = body["next_page_token"]

    announced: list[dict[str, Any]] = []
    ann_errors: list[Any] = []
    since = BACKFILL_START
    while since <= today:
        until = min(since + timedelta(days=89), today)
        status, body = http.get(
            TRADING_URL,
            "/v2/corporate_actions/announcements",
            {
                "ca_types": "Dividend",
                "symbol": SYMBOL,
                "since": since.isoformat(),
                "until": until.isoformat(),
                "date_type": "ex_date",
            },
        )
        if status != 200:
            ann_errors.append({"since": since.isoformat(), "status": status, "error": body})
        else:
            for row in body:
                announced.append({k: row.get(k) for k in ("ex_date", "cash", "declaration_date")})
        since = until + timedelta(days=1)

    def by_quarter(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        quarters: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            ex = date.fromisoformat(row["ex_date"])
            quarters.setdefault(f"{ex.year}-{ex.month:02d}", []).append(row)
        return quarters

    feed_q, ann_q = by_quarter(feed), by_quarter(announced)
    schedule = []
    for year in range(2016, today.year + 1):
        for month in (3, 6, 9, 12):
            expected = third_friday(year, month)
            if expected > today:
                continue
            key = f"{year}-{month:02d}"
            schedule.append(
                {
                    "quarter": key,
                    "third_friday": expected.isoformat(),
                    "feed": feed_q.get(key, []),
                    "announcement": ann_q.get(key, []),
                }
            )
    off_schedule = sorted(
        {r["ex_date"] for r in feed if date.fromisoformat(r["ex_date"]).month not in (3, 6, 9, 12)}
    )
    # A quarter the feed lacks: ask again around its third Friday with no type filter, in case
    # it is filed under another action type.
    missing = [s for s in schedule if not s["feed"]]
    any_type: dict[str, Any] = {}
    for s in missing:
        friday = date.fromisoformat(s["third_friday"])
        status, body = http.get(
            DATA_URL,
            "/v1/corporate-actions",
            {
                "symbols": SYMBOL,
                "start": (friday - timedelta(days=14)).isoformat(),
                "end": (friday + timedelta(days=14)).isoformat(),
                "limit": "1000",
            },
        )
        any_type[s["quarter"]] = (
            {
                kind: [{k: r.get(k) for k in ("ex_date", "rate", "special")} for r in rows]
                for kind, rows in (body.get("corporate_actions") or {}).items()
            }
            if status == 200
            else {"status": status, "error": body}
        )
    declared_after_ex = [
        r for r in announced if r["declaration_date"] and r["declaration_date"] > r["ex_date"]
    ]
    return {
        "feed_missing_any_type": any_type,
        "announcement_declared_after_ex": declared_after_ex,
        "feed_status": feed_status,
        "feed_error": feed_error,
        "feed_fields": sorted(fields),
        "feed_count": len(feed),
        "feed_with_declaration_date": sum(1 for r in feed if r.get("declaration_date")),
        "feed_missing_quarters": [s["quarter"] for s in schedule if not s["feed"]],
        "feed_off_third_friday": [
            r["ex_date"] for s in schedule for r in s["feed"] if r["ex_date"] != s["third_friday"]
        ],
        "feed_off_schedule_months": off_schedule,
        "announcement_errors": ann_errors,
        "announcement_count": len(announced),
        "announcement_quarters": sorted(ann_q),
        "announcement_missing_quarters": [s["quarter"] for s in schedule if not s["announcement"]],
        "schedule": schedule,
    }


# --- 5. Paper dividends ---------------------------------------------------------------------


def paper_dividends_check(http: Http) -> dict[str, Any]:
    status, account = http.get(TRADING_URL, "/v2/account", {})
    created = account.get("created_at") if status == 200 else None
    div_status, divs = http.get(TRADING_URL, "/v2/account/activities/DIV", {"page_size": "100"})
    act_status, acts = http.get(TRADING_URL, "/v2/account/activities", {"page_size": "100"})
    pos_status, positions = http.get(TRADING_URL, "/v2/positions", {})
    return {
        "account_status": account.get("status") if status == 200 else status,
        "account_created": created[:10] if created else None,
        "div_status": div_status,
        "div_entries": len(divs) if div_status == 200 else None,
        "activities_status": act_status,
        "activity_types_last_100": dict(Counter(a.get("activity_type") for a in acts))
        if act_status == 200
        else None,
        "oldest_activity_of_last_100": min(
            (str(a.get("transaction_time") or a.get("date") or "")[:10] for a in acts),
            default=None,
        )
        if act_status == 200
        else None,
        "open_positions": len(positions) if pos_status == 200 else None,
        "docs": "https://docs.alpaca.markets/docs/paper-trading says paper trading does NOT "
        "simulate dividends (page updated 2026-07-07)",
    }


# --- main ----------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--live", action="store_true", help="check 1, during regular hours")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--out", type=Path, default=Path("var/probe"))
    args = parser.parse_args()

    settings = load_settings(args.env)
    http = Http(settings.alpaca)
    calendar = TradingCalendar.xnys()
    started = CLOCK.now()
    results: dict[str, Any] = {
        "run_at": started.isoformat(),
        "mode": "live" if args.live else "default",
    }
    if args.live:
        results["sip_delay"] = sip_live(http, calendar)
    else:
        client = AlpacaMarketData(settings.alpaca, clock=CLOCK, calendar=calendar)
        results["sip_snapshot"] = sip_snapshot(http)
        results["sessions"] = sessions_check(http, client, calendar)
        results["dividends"] = dividends_check(http)
        results["paper_dividends"] = paper_dividends_check(http)
        results["rate_limits"] = rate_check(http, calendar)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"alpaca_probe-{results['mode']}-{started:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nwritten to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
