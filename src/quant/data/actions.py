"""Corporate actions: dividends and splits, as Alpaca reports them (spec §2, ARCHITECTURE §6.1).

Two Alpaca sources report them, in different shapes:

- the market-data feed, `GET /v1/corporate-actions` (`actions_from_feed`), which reaches back to
  2016 but gives no declaration date: its cash dividends carry only the ex, process, record and
  payable dates;
- the trading host's announcements, `GET /v2/corporate_actions/announcements`
  (`actions_from_announcements`), which give `declaration_date` but, as the probe found on
  2026-10-15, only from 2020-09 on (docs/VERIFY.md).

As history, an action is known at the end of its declaration day in New York (ARCHITECTURE
§6.2). Where the source gives no declaration date, the ex-date stands in for it: the latest day
the store accepts (`BarStore.put_dividend` refuses one declared after its ex-date), so an action
is never known earlier than it could have been. A declaration date after the ex-date can't be
right (five of SPY's announcements have one, VERIFY.md), so it counts as missing too.
`declared_in_feed` says whether the source's date was used.

SPY has never split, and raw prices break across a split, so a split in the feed halts trading
(ARCHITECTURE §6.1). A stock dividend (an announcement's `ca_sub_type` "stock") changes the
share count the same way, so it maps to a split of `new_rate / old_rate`, never to cash.
`store_actions` stores splits like any action and surfaces them in `ActionSync`.

Actions are fetched again on every backfill. One stored before with the same values changes
nothing. One that differs is a revision, which history can't make (ARCHITECTURE §6.2): it is
left as stored and reported in `ActionSync.conflicts`, and the others are still stored.

**The dividend supplement** (B7, the owner's call at CP1): the feed misses SPY's dividends with
ex-dates 2016-03-18, 2018-06-15 and 2026-09-18 (docs/VERIFY.md). They come from State Street's
"ETF Historical Distributions" spreadsheet, as the committed file `config/dividend_supplement.json`
(`load_dividend_supplement`). The source gives no declaration date, so each row is a Dividend
declared on its ex-date with `declared_in_feed=False`, exactly like a feed dividend without one.
The backfill stores the supplement after the feed, under the same stamp
(`quant.data.backfill.backfill`'s `supplement`). A row that differs from a stored dividend is a
conflict, reported in the supplement's `ActionSync.conflicts`, never an overwrite.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

from quant.core.errors import DataError
from quant.data.market_db import BarStore, Stamp

# The committed supplement, relative to the repo root (the CLI's default).
SUPPLEMENT_PATH: Final = Path("config/dividend_supplement.json")


@dataclass(frozen=True)
class Dividend:
    symbol: str
    ex_date: date
    amount: float  # cash per share
    declared: date
    declared_in_feed: bool  # False when the ex-date stands in for a missing declaration date


@dataclass(frozen=True)
class Split:
    symbol: str
    ex_date: date
    ratio: float  # new shares per old share: 2.0 for a 2-for-1 split, 0.1 for 1-for-10 reverse
    declared: date
    declared_in_feed: bool


@dataclass(frozen=True)
class CorporateActions:
    dividends: tuple[Dividend, ...]
    splits: tuple[Split, ...]


@dataclass(frozen=True)
class ActionSync:
    """What `store_actions` stored, and why trading must halt, if it must."""

    dividends: int  # the dividends stored or already stored, conflicts left out
    splits: tuple[Split, ...]
    conflicts: tuple[str, ...] = ()  # actions that couldn't be stored as given, and why

    @property
    def halt(self) -> str | None:
        """Why trading must halt: every split, named by symbol and ex-date. None without one."""
        if not self.splits:
            return None
        return "split: " + ", ".join(f"{s.symbol} {s.ex_date}" for s in self.splits)


def actions_from_feed(payload: Any) -> CorporateActions:
    """Map the market-data feed's `corporate_actions` object (cash_dividends, forward_splits,
    reverse_splits) to dividends and splits, in ex-date order. A dividend's amount is its
    `rate`; a split's ratio is `new_rate / old_rate`. Other action types are ignored. Raises
    DataError when the payload isn't an object, or on a record missing a field it needs."""
    if not isinstance(payload, Mapping):
        raise DataError("malformed corporate actions: not an object")
    dividends: list[Dividend] = []
    splits: list[Split] = []
    try:
        for kind in ("cash_dividends", "forward_splits", "reverse_splits"):
            for row in payload.get(kind, []):
                ex = date.fromisoformat(row["ex_date"])
                given = row.get("declaration_date")
                declared = date.fromisoformat(given) if given else ex
                supplied = declared <= ex and bool(given)
                if not supplied:  # missing, or after the ex-date: the ex-date stands in
                    declared = ex
                symbol = str(row["symbol"])
                if kind == "cash_dividends":
                    dividends.append(
                        Dividend(symbol, ex, _positive(row["rate"]), declared, supplied)
                    )
                else:
                    ratio = _positive(row["new_rate"]) / _positive(row["old_rate"])
                    splits.append(Split(symbol, ex, ratio, declared, supplied))
    except (KeyError, ValueError, TypeError) as exc:
        field = str(exc) if isinstance(exc, KeyError) else "invalid value"
        raise DataError(f"malformed corporate action: {field}") from None
    return CorporateActions(
        tuple(sorted(dividends, key=lambda d: d.ex_date)),
        tuple(sorted(splits, key=lambda s: s.ex_date)),
    )


def actions_from_announcements(payload: Any) -> CorporateActions:
    """Map the trading host's announcements (ca_type dividend or split) to dividends and splits,
    declared on their `declaration_date`. A cash dividend's amount is `cash`, a decimal string;
    a split's ratio, and a stock dividend's, is `new_rate / old_rate`. Raises DataError when the
    payload isn't a list of records, or on a record missing a field it needs."""
    if not isinstance(payload, list):
        raise DataError("malformed announcements: not a list")
    feed: dict[str, list[dict[str, Any]]] = {"cash_dividends": [], "forward_splits": []}
    for row in payload:
        if not isinstance(row, Mapping):
            raise DataError("malformed announcement: not an object")
        kind = str(row.get("ca_type", "")).lower()
        if kind not in ("dividend", "split"):
            continue
        mapped = dict(row)
        if "initiating_symbol" not in row:
            raise DataError("missing initiating_symbol")
        mapped["symbol"] = row["initiating_symbol"]
        if kind == "dividend" and str(row.get("ca_sub_type", "")).lower() == "stock":
            kind = "split"  # more shares per share: raw prices break as at a split
        if kind == "dividend":
            if "cash" not in row:
                raise DataError("missing cash")
            mapped["rate"] = row["cash"]
        feed["cash_dividends" if kind == "dividend" else "forward_splits"].append(mapped)
    return actions_from_feed(feed)


def store_actions(
    store: BarStore, symbol: str, actions: CorporateActions, *, known_at: Stamp
) -> ActionSync:
    """Store `symbol`'s dividends and splits (`BarStore.put_dividend`, `put_split`), stamped as
    `known_at` says. Storing the same actions twice changes nothing. An action the store
    refuses, such as a changed one under HISTORY, is left as stored and reported in
    `ActionSync.conflicts`; the others are still stored. Every split is surfaced, so it
    halts."""
    dividends = [d for d in actions.dividends if d.symbol == symbol]
    splits = tuple(s for s in actions.splits if s.symbol == symbol)
    conflicts: list[str] = []
    stored = 0
    for d in dividends:
        try:
            store.put_dividend(
                symbol, ex_date=d.ex_date, amount=d.amount, declared=d.declared, known_at=known_at
            )
            stored += 1
        except DataError as exc:
            conflicts.append(str(exc))
    for s in splits:
        try:
            store.put_split(
                symbol, ex_date=s.ex_date, ratio=s.ratio, declared=s.declared, known_at=known_at
            )
        except DataError as exc:
            conflicts.append(str(exc))
    return ActionSync(stored, splits, tuple(conflicts))


def _positive(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise DataError("corporate action rate must be finite and positive")
    return number


@dataclass(frozen=True)
class SupplementSource:
    """Where the supplement's rows come from: the publisher's `name`, the `page` that links the
    file, the file's `url` (both https), the day it was `retrieved`, and the `sha256` of the
    retrieved file's bytes (64 lowercase hex characters)."""

    name: str
    page: str
    url: str
    retrieved: date
    sha256: str


@dataclass(frozen=True)
class SupplementRow:
    """One dividend as the source lists it. `amount` is cash per share."""

    symbol: str
    cusip: str
    ex_date: date
    record_date: date
    payable_date: date
    amount: float


@dataclass(frozen=True)
class DividendSupplement:
    """The parsed supplement file: its source, its note, and its rows in the file's order,
    sorted by (ex_date, symbol)."""

    source: SupplementSource
    note: str
    rows: tuple[SupplementRow, ...]

    @property
    def actions(self) -> CorporateActions:
        """The rows as corporate actions: one Dividend per row, in ex-date order, `Dividend(
        symbol, ex_date, amount, declared=ex_date, declared_in_feed=False)` (the ex-date stands
        in for the missing declaration date, ARCHITECTURE §6.2), and no splits."""
        raise NotImplementedError("B7a: DividendSupplement.actions")


def load_dividend_supplement(path: Path) -> DividendSupplement:
    """Parse the supplement file at `path` (`config/dividend_supplement.json`).

    The file is JSON, read with `quant.core.canonical.canonical_loads` (duplicate keys, NaN and
    infinities refused; any whitespace and key order accepted, so it stays reviewable). Its
    top-level object has exactly the keys:

    - "source": an object with exactly "name", "page", "url", "retrieved" (an ISO date) and
      "sha256" (64 lowercase hex characters); "page" and "url" start with "https://";
    - "note": a non-empty string;
    - "dividends": a non-empty list of objects with exactly "symbol", "cusip", "ex_date",
      "record_date", "payable_date" (ISO dates) and "amount" (a JSON number, finite and > 0).
      Within a row ex_date <= record_date <= payable_date. The rows are sorted strictly by
      (ex_date, symbol), so no (symbol, ex_date) appears twice.

    Strings must be non-empty. Raises DataError naming the problem for an unreadable file
    (OSError), text that isn't such JSON, a missing or extra key, a value of the wrong type, a
    bad date or amount, or rows out of order. Nothing else is accepted, so an edit to the file
    either keeps this shape or fails loudly."""
    raise NotImplementedError("B7a: load_dividend_supplement")
