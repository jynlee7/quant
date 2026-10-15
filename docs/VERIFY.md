# VERIFY: SPY Regime Trader

What the vendor spikes found (PLAN P11). There is one section for each item in ARCHITECTURE §18, then the other checks B6 needed. Each section gives the date checked, the method, the result, and what the build does about it. The probe is `spikes/alpaca_probe.py`. It writes its raw results to `var/probe/` (gitignored), and it never prints, logs or writes a key, an account number, an id or a request id.

Run it with:

- `.venv/bin/python spikes/alpaca_probe.py` for checks 2–5, at any time;
- `.venv/bin/python spikes/alpaca_probe.py --live` for check 1, between 10:00 New York time and 20 minutes before the close. It refuses at other times and says when to rerun.

## 1. Paper accounts and dividends (ARCH §18)

- **Checked:** 2026-10-15, on the trading account's paper keys from `.env`.
- **Method:** `GET /v2/account/activities/DIV`, the last 100 activities of any type, and the open positions. Also Alpaca's paper-trading page.
- **Result:**
  - The account was created on 2026-10-15. It has no activities of any type and no positions, so it can't show a dividend yet.
  - Alpaca's docs say so directly: "Paper trading account **does NOT** simulate dividends", and dividends are listed among what paper trading doesn't account for (https://docs.alpaca.markets/docs/paper-trading, page updated 2026-07-07).
- **The build:** treat paper as not crediting dividends. The parity report adds each dividend the backtest credited, from `market.db`, as an explicit reconciling line (ARCH §18 already says it reconciles them either way). Recheck once the account has held SPY through an ex-date: a DIV entry would contradict the docs.

## 2. A second paper account (AD3, ARCH §18)

- **Checked:** 2026-10-15.
- **Method:** Alpaca's docs and forum. The API can't answer this without opening an account, so the probe doesn't try.
- **Result:** the paper-trading page describes "Open New Paper Account" and "Delete Account" under Account Settings, but states no limit. A forum reply from Alpaca says each login may hold 3 paper accounts and 1 live account (https://forum.alpaca.markets/t/feature-request-more-paper-trading-accounts/18125, not official documentation).
- **The build:** **owner question.** Can your login open a second paper account for `.env.test`? Until one exists, the trading account stands in for the test account (pack status, CP3).

## 3. SIP delay against D2's 16 minutes (ARCH §18)

- **Status:** **pending: run `.venv/bin/python spikes/alpaca_probe.py --live` during regular hours.**
- **Checked so far:** 2026-10-15, 04:09 New York time, market closed.
- **Method (`--live`):** for 20 minutes, every 30 s, ask for `feed=sip`, `adjustment=raw` minute bars from the last 30 minutes, with no `end`. For each regular-hours minute that first appears during the run, record the lag from the minute's end to the poll that first returned it. Minutes already there at the first poll are left out, because their lag is only bounded. The 30 s poll can overstate a lag by up to 30 s, so D2 holds if the worst lag is at most 15.5 minutes. The probe also records any minute whose values change after it first appears.
- **Result so far (closed market):**
  - With an explicit `end` of now, the bars request returns **403 "subscription does not permit querying recent SIP data"**.
  - With `end` omitted, it returns 200, here with no minutes, because nothing traded in the window it would serve.
  - The documented limit is "latest 15 minutes" for historical data on the Basic plan (https://docs.alpaca.markets/docs/about-market-data-api).
- **The build:** D2 stands until the live run. Two points for the live path:
  - a bars request whose `end` is less than 15 minutes old fails with 403, so the engine's request at τ = close + 16 min must end at the bar's close, never at "now";
  - if the live run shows a worst lag over 15.5 minutes, the delay is one config value: propose the new value at CP1, and don't work around it (PLAN B6).

## 4. Rate limits against the backfill (ARCH §18)

- **Checked:** 2026-10-15.
- **Method:** the `X-RateLimit-*` headers on the probe's own requests: 16 to the data host and 48 to the paper trading host, spaced 0.4 s apart. No load test. The backfill estimate uses the client's settings: one request per session (page size 10,000), spaced for 190 requests a minute.
- **Result:**
  - `X-RateLimit-Limit` is **200** on both `data.alpaca.markets` and `paper-api.alpaca.markets`. `X-RateLimit-Reset` is present on both. Remaining never fell below 197, so the window refills continuously.
  - This matches the documented 200 historical calls a minute on the Basic plan. The B6a red agent recorded 1000 a minute for the trading host; today's headers say 200.
  - A full session is 391 raw rows (the close minute comes back too), one page.
  - **The backfill:** 2,711 sessions from 2016-01-04 to 2026-10-14, about 1.06 million minutes, take **2,711 requests and about 14.3 minutes** at 190 a minute. The median latency was 0.16 s, so the spacing sets the pace, not the network.
- **The build:** no change. The client's 190 a minute stays under the limit, and 429 handling stays as a backstop. Rerunning a resumed backfill costs one request per missing session.

## 5. Jev (ARCH §18, v1.6)

- **Status:** not checked here. This is B20's spike, `spikes/jev_probe.py`.

## Historical SIP sessions (B6, SPEC §2)

- **Checked:** 2026-10-15.
- **Method:** each session's bars from its open to its close, `feed=sip`, `adjustment=raw`. Raw rows are counted against the minute grid [open, close), and again through `AlpacaMarketData.session_minutes`.
- **Result:** every session is complete, one page each, with no duplicates and no missing minute:

  | Session | Kind | Minutes |
  |---|---|---|
  | 2016-01-04 | first session of the backfill | 390 |
  | 2016-03-01 | full day | 390 |
  | 2016-11-25 | half day | 210 |
  | 2016-12-30 | full day | 390 |
  | 2025-11-28 | half day | 210 |
  | 2026-10-05 to 2026-10-09 | last week, five full days | 390 each |

  Each request also returns one extra row: the minute that starts at the close, because `end` is inclusive. The client drops it.
- **The build:** no change. The quality report must still explain any gap the full backfill finds.

## Corporate actions: SPY dividends (B6, ARCH §6.1–§6.2)

- **Checked:** 2026-10-15.
- **Method:**
  - The data feed, `GET /v1/corporate-actions`, `types=cash_dividend`, from 2016-01-01 to today.
  - The trading host's announcements, `GET /v2/corporate_actions/announcements`, `ca_types=Dividend`, in 90-day windows by ex-date.
  - Both are checked against SPY's schedule. The ex-date falls on the third Friday of March, June, September and December, or the day before when that Friday is a holiday.
  - For each quarter the feed lacks, the feed was asked again for four weeks around it with no type filter.
- **Result, the feed:**
  - It returns 40 dividends from 2016 to 2026. **None has a declaration date.** Its fields are `cusip, ex_date, foreign, id, payable_date, process_date, rate, record_date, special, symbol`.
  - **Missing quarters: 2016-03 (ex 2016-03-18), 2018-06 (ex 2018-06-15) and 2026-09 (ex 2026-09-18).** Asking with no type filter returns nothing for any of them.
  - That confirms the red agent's 2018-06 finding and adds two more. Within 2016–2025 the feed misses 2 of 40 quarters, and both are in FIT (2016–2018).
  - Every ex-date it does return is on schedule. The one off the third Friday is 2026-06-18, a Thursday, because 2026-06-19 is a holiday.
- **Result, the announcements:**
  - 25 dividends, **2020-09 onward only.** There are none for 2016 to 2020-06.
  - They include 2026-09-18 (1.888834), which the feed lacks. The amounts match the feed's wherever both have a quarter.
  - **Five declaration dates fall after their ex-date:** 2020-09-18 (declared 2020-11-10), 2022-03-18 (2022-03-20), 2022-06-17 (2022-06-20), 2022-09-16 (2022-09-18) and 2022-12-16 (2022-12-18). From 2023 on they are 1–3 days before the ex-date, as expected.
- **The build:**
  - **Declaration dates:** before 2020-09 no source has one, so the ex-date stands in (`declared_in_feed=False`, B6a's call; ARCH §6.2 note due at the B6 merge).
  - **Bad declaration dates:** a declaration date after the ex-date is invalid. `put_dividend` refuses one, so any use of `actions_from_announcements` must treat it as missing (ex-date stands in) rather than fail. B6a's mapping passes it through unchanged. This is for the B6 reviewers.
  - **Missing dividends:** SimBroker credits dividends on the ex-date, (SPEC §2, ARCH §10), and the buy-and-hold baseline includes them (PLAN B12). So 2016-03 and 2018-06 must come from a second source before FIT's report, or the backtest and baseline undercount them. **Owner question:** is a small committed supplement, taken from State Street's SPY distribution history and checked by you, acceptable? Its amounts aren't recorded here, because they haven't been verified.
  - **Recent dividends:** the feed hadn't listed 2026-09-18 four weeks after the ex-date. Live dividend sync must read the announcements, not only the feed.
