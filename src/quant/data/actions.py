"""Corporate actions: dividends and splits, as Alpaca reports them (spec §2, ARCHITECTURE §6.1).

Two Alpaca sources report them, in different shapes:

- the market-data feed, `GET /v1/corporate-actions` (`actions_from_feed`), which reaches back to
  2016 but gives no declaration date: its cash dividends carry only the ex, process, record and
  payable dates;
- the trading host's announcements, `GET /v2/corporate_actions/announcements`
  (`actions_from_announcements`), which give `declaration_date` but, as recorded on 2026-10-15,
  return nothing for 2017.

As history, an action is known at the end of its declaration day in New York (ARCHITECTURE
§6.2). Where the source gives no declaration date, the ex-date stands in for it: the latest day
the store accepts (`BarStore.put_dividend` refuses one declared after its ex-date), so an action
is never known earlier than it could have been. `declared_in_feed` says which it was.

SPY has never split, and raw prices break across a split, so a split in the feed halts trading
(ARCHITECTURE §6.1). `store_actions` stores it like any action and surfaces it in `ActionSync`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from quant.data.market_db import BarStore, Stamp


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

    dividends: int
    splits: tuple[Split, ...]

    @property
    def halt(self) -> str | None:
        """Why trading must halt: every split, named by symbol and ex-date. None without one."""
        raise NotImplementedError("B6: ActionSync.halt")


def actions_from_feed(payload: Mapping[str, Any]) -> CorporateActions:
    """Map the market-data feed's `corporate_actions` object (cash_dividends, forward_splits,
    reverse_splits) to dividends and splits, in ex-date order. A dividend's amount is its
    `rate`; a split's ratio is `new_rate / old_rate`. Other action types are ignored. Raises
    DataError on a record missing a field it needs."""
    raise NotImplementedError("B6: actions_from_feed")


def actions_from_announcements(payload: Sequence[Mapping[str, Any]]) -> CorporateActions:
    """Map the trading host's announcements (ca_type dividend or split) to dividends and splits,
    declared on their `declaration_date`. A dividend's amount is `cash`, a decimal string; a
    split's ratio is `new_rate / old_rate`. Raises DataError on a record missing a field it
    needs."""
    raise NotImplementedError("B6: actions_from_announcements")


def store_actions(
    store: BarStore, symbol: str, actions: CorporateActions, *, known_at: Stamp
) -> ActionSync:
    """Store `symbol`'s dividends and splits (`BarStore.put_dividend`, `put_split`), stamped as
    `known_at` says. Storing the same actions twice changes nothing."""
    raise NotImplementedError("B6: store_actions")
