"""B6: corporate actions map with their declaration dates, are known as history at the end of that
day (ARCHITECTURE §6.2), and a split is surfaced so trading halts (ARCHITECTURE §6.1).

The market-data feed gives no declaration date (recorded 2026-10-15), so its ex-date stands in:
the latest day the store accepts, which never makes an action known early.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest

from quant.core.clock import SimClock
from quant.core.errors import DataError
from quant.data.actions import (
    CorporateActions,
    Dividend,
    Split,
    actions_from_announcements,
    actions_from_feed,
    store_actions,
)
from quant.data.alpaca_data import AlpacaMarketData
from quant.data.market_db import HISTORY, BarStore
from quant.data.policy import Purpose
from tests.support.alpaca import (
    ANNOUNCEMENTS,
    CORPORATE_ACTIONS,
    DATA_HOST,
    KEYS,
    TRADING_HOST,
    AlpacaFake,
    alpaca_fake,
    fixture,
)
from tests.support.bars import SYMBOL

OPERATE = Purpose.OPERATE


@pytest.fixture
def clock() -> SimClock:
    return SimClock(datetime(2018, 6, 1, 15, 0, tzinfo=UTC))


@pytest.fixture
def fake(clock: SimClock) -> Iterator[AlpacaFake]:
    yield from alpaca_fake(clock)


def feed() -> CorporateActions:
    return actions_from_feed(fixture("corporate_actions_spy_2016_2018.json")["corporate_actions"])


@pytest.mark.spec("ARCH§6.2", "SPEC§2")
def test_feed_dividends_map_with_the_ex_date_standing_in_for_the_declaration() -> None:
    actions = feed()
    assert actions.splits == ()
    assert len(actions.dividends) == 10  # the feed has no 2018-06 dividend
    assert [d.ex_date for d in actions.dividends] == sorted(d.ex_date for d in actions.dividends)
    assert actions.dividends[0] == Dividend(
        symbol="SPY",
        ex_date=date(2016, 6, 17),
        amount=1.078442,
        declared=date(2016, 6, 17),
        declared_in_feed=False,
    )
    assert all(d.declared == d.ex_date and not d.declared_in_feed for d in actions.dividends)


@pytest.mark.spec("ARCH§6.2")
def test_announcements_map_with_their_declaration_dates() -> None:
    actions = actions_from_announcements(fixture("announcements_spy_2017-12_planted.json"))
    assert actions.splits == ()
    assert actions.dividends == (
        Dividend(
            symbol="SPY",
            ex_date=date(2017, 12, 15),
            amount=1.351333,
            declared=date(2017, 12, 13),
            declared_in_feed=True,
        ),
    )


@pytest.mark.spec("ARCH§6.1")
def test_splits_map_to_their_ratios() -> None:
    planted = fixture("corporate_actions_spy_splits_planted.json")["corporate_actions"]
    actions = actions_from_feed(planted)
    assert actions.dividends == ()
    assert actions.splits == (
        Split("SPY", date(2017, 11, 28), 2.0, date(2017, 11, 28), False),
        Split("SPY", date(2017, 12, 12), 0.1, date(2017, 12, 12), False),
    )


@pytest.mark.spec("ARCH§6.1")
def test_a_record_missing_a_field_is_refused() -> None:
    with pytest.raises(DataError, match="ex_date"):
        actions_from_feed({"cash_dividends": [{"symbol": "SPY", "rate": 1.0}]})


@pytest.mark.spec("ARCH§6.1", "SPEC§2")
def test_the_client_asks_the_feed_for_dividends_and_splits(
    clock: SimClock, fake: AlpacaFake
) -> None:
    fake.serve(CORPORATE_ACTIONS, "corporate_actions_spy_2016_2018.json")
    actions = AlpacaMarketData(KEYS, clock=clock).corporate_actions(
        "SPY", date(2016, 1, 1), date(2018, 12, 31)
    )

    (seen,) = fake.requests_to(CORPORATE_ACTIONS)
    assert seen.host == DATA_HOST
    assert seen.query["symbols"] == "SPY"
    assert {"cash_dividend", "forward_split", "reverse_split"} <= set(
        seen.query["types"].split(",")
    )
    assert seen.query["start"] == "2016-01-01"
    assert seen.query["end"] == "2018-12-31"
    assert actions == feed()


@pytest.mark.spec("ARCH§6.2")
def test_the_client_reads_announcements_from_the_trading_host(
    clock: SimClock, fake: AlpacaFake
) -> None:
    fake.serve(ANNOUNCEMENTS, "announcements_spy_2017-12_planted.json")
    actions = AlpacaMarketData(KEYS, clock=clock).announcements(
        "SPY", date(2017, 11, 1), date(2017, 12, 31)
    )

    (seen,) = fake.requests_to(ANNOUNCEMENTS)
    assert seen.host == TRADING_HOST
    assert seen.query["symbol"] == "SPY"
    assert {"Dividend", "Split"} <= set(seen.query["ca_types"].split(","))
    assert seen.query["since"] == "2017-11-01"
    assert seen.query["until"] == "2017-12-31"
    assert actions.dividends[0].declared == date(2017, 12, 13)


@pytest.mark.spec("ARCH§6.2")
def test_a_stored_dividend_is_known_at_the_end_of_its_declaration_day() -> None:
    store = BarStore.memory()
    actions = actions_from_announcements(fixture("announcements_spy_2017-12_planted.json"))
    sync = store_actions(store, SYMBOL, actions, known_at=HISTORY)
    assert sync.dividends == 1
    assert sync.splits == ()
    assert sync.halt is None
    store_actions(store, SYMBOL, actions, known_at=HISTORY)  # storing again changes nothing

    known = datetime(2017, 12, 14, 5, 0, tzinfo=UTC)  # midnight after 2017-12-13 in New York
    span = (date(2017, 12, 1), date(2017, 12, 31))
    before = store.view(known, as_of=known)
    after = store.view(known.replace(minute=1), as_of=known.replace(minute=1))
    assert before.dividends(SYMBOL, *span, purpose=OPERATE) == []
    assert after.dividends(SYMBOL, *span, purpose=OPERATE) == [(date(2017, 12, 15), 1.351333)]


@pytest.mark.spec("ARCH§6.1")
def test_a_split_in_the_feed_is_stored_and_halts_trading() -> None:
    store = BarStore.memory()
    planted = fixture("corporate_actions_spy_splits_planted.json")["corporate_actions"]
    sync = store_actions(store, SYMBOL, actions_from_feed(planted), known_at=HISTORY)

    assert len(sync.splits) == 2
    assert sync.halt is not None
    assert "split" in sync.halt
    assert "2017-11-28" in sync.halt
    assert "2017-12-12" in sync.halt
    tau = datetime(2018, 1, 2, 15, 0, tzinfo=UTC)
    splits = store.view(tau, as_of=tau).splits(
        SYMBOL, date(2017, 11, 1), date(2017, 12, 31), purpose=OPERATE
    )
    assert splits == [(date(2017, 11, 28), 2.0), (date(2017, 12, 12), 0.1)]
