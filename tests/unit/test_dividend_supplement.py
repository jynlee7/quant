"""B7a: the State Street dividend supplement (the owner's call at CP1; ARCHITECTURE §6.2).

The loader test pins the committed file's rows and source, so any edit to the file shows up in
review. The backfill stores the supplement after the feed, under the same stamp; a row that
differs from a stored dividend is a conflict, reported, never an overwrite.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from quant.cli.main import app
from quant.core.clock import SimClock
from quant.core.errors import DataError
from quant.data.actions import (
    SUPPLEMENT_PATH,
    CorporateActions,
    Dividend,
    DividendSupplement,
    SupplementRow,
    SupplementSource,
    load_dividend_supplement,
)
from quant.data.backfill import backfill
from quant.data.market_db import HISTORY, NOW, BarStore
from quant.data.policy import Purpose
from tests.support.alpaca import FakeSource
from tests.support.bars import SYMBOL, synthetic_days

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / SUPPLEMENT_PATH
OPERATE = Purpose.OPERATE

# The supplement row in this span is 2016-03-18; the backfill covers the week.
FIRST, LAST = date(2016, 3, 14), date(2016, 3, 18)
AFTER = datetime(2016, 3, 21, 15, 0, tzinfo=UTC)
EX = date(2016, 3, 18)
AMOUNT = 1.049604
# The end of the ex-date in New York, when the row is known as history: 2016-03-19 00:00 EDT.
KNOWN = datetime(2016, 3, 19, 4, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def minutes() -> Any:
    return synthetic_days(FIRST, LAST, seed=7)


def supplement(*rows: tuple[date, float], symbol: str = SYMBOL) -> DividendSupplement:
    """A supplement holding the given (ex_date, amount) rows."""
    return DividendSupplement(
        source=SupplementSource(
            name="test",
            page="https://example.com/page",
            url="https://example.com/file.xlsx",
            retrieved=date(2026, 10, 15),
            sha256="0" * 64,
        ),
        note="test rows",
        rows=tuple(SupplementRow(symbol, "78462F103", ex, ex, ex, amount) for ex, amount in rows),
    )


def dividends_known(store: BarStore, at: datetime) -> list[tuple[date, float]]:
    view = store.view(at, as_of=at)
    return view.dividends(SYMBOL, date(2016, 1, 1), date(2016, 12, 31), purpose=OPERATE)


# --- The committed file -----------------------------------------------------------------------


@pytest.mark.spec("ARCH§6.2", "SPEC§2")
def test_the_committed_supplement_holds_exactly_state_streets_three_rows() -> None:
    loaded = load_dividend_supplement(COMMITTED)
    assert Path("config/dividend_supplement.json") == SUPPLEMENT_PATH
    assert loaded.source == SupplementSource(
        name="State Street Global Advisors, SPDR ETF Historical Distributions",
        page="https://www.ssga.com/us/en/intermediary/resources/documents/etf-dividend-distributions",
        url="https://www.ssga.com/library-content/products/fund-data/etfs/us/spdr-etf-historical-distributions.xlsx",
        retrieved=date(2026, 10, 15),
        sha256="b162dd17027d1128f2209e312b0f634960daa78319dafe0ab821ab0b7e77ce23",
    )
    assert loaded.note
    assert loaded.rows == (
        SupplementRow(
            "SPY", "78462F103", date(2016, 3, 18), date(2016, 3, 22), date(2016, 4, 29), 1.049604
        ),
        SupplementRow(
            "SPY", "78462F103", date(2018, 6, 15), date(2018, 6, 18), date(2018, 7, 31), 1.245568
        ),
        SupplementRow(
            "SPY", "78462F103", date(2026, 9, 18), date(2026, 9, 18), date(2026, 10, 30), 1.888834
        ),
    )


@pytest.mark.spec("ARCH§6.2")
def test_supplement_rows_are_dividends_with_the_ex_date_standing_in_for_the_declaration() -> None:
    actions = load_dividend_supplement(COMMITTED).actions
    assert actions == CorporateActions(
        dividends=(
            Dividend("SPY", date(2016, 3, 18), 1.049604, date(2016, 3, 18), False),
            Dividend("SPY", date(2018, 6, 15), 1.245568, date(2018, 6, 15), False),
            Dividend("SPY", date(2026, 9, 18), 1.888834, date(2026, 9, 18), False),
        ),
        splits=(),
    )


def _committed_json() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(COMMITTED.read_text())
    return document


def _bad_documents() -> dict[str, Any]:
    """Each a change to the committed document that the loader must refuse."""

    def edit(change: Any) -> Any:
        def build() -> str:
            document = _committed_json()
            change(document)
            return json.dumps(document)

        return build

    def row(key: str, value: Any) -> Any:
        return edit(lambda d: d["dividends"][0].__setitem__(key, value))

    return {
        "missing source": edit(lambda d: d.pop("source")),
        "extra top-level key": edit(lambda d: d.__setitem__("extra", 1)),
        "empty note": edit(lambda d: d.__setitem__("note", "")),
        "no rows": edit(lambda d: d.__setitem__("dividends", [])),
        "bad sha256": edit(lambda d: d["source"].__setitem__("sha256", "B162" + "0" * 60)),
        "http url": edit(lambda d: d["source"].__setitem__("url", "http://www.ssga.com/x.xlsx")),
        "bad retrieved date": edit(lambda d: d["source"].__setitem__("retrieved", "2026-13-01")),
        "missing amount": edit(lambda d: d["dividends"][0].pop("amount")),
        "extra row key": row("declared", "2016-03-01"),
        "amount as a string": row("amount", "1.049604"),
        "zero amount": row("amount", 0.0),
        "negative amount": row("amount", -1.0),
        "bad ex_date": row("ex_date", "2016-02-30"),
        "record before ex": row("record_date", "2016-03-17"),
        "payable before record": row("payable_date", "2016-03-21"),
        "empty symbol": row("symbol", ""),
        "duplicate ex-date": edit(
            lambda d: d["dividends"].insert(1, dict(d["dividends"][0], amount=2.0))
        ),
        "out of order": edit(lambda d: d["dividends"].reverse()),
        "NaN": lambda: COMMITTED.read_text().replace("1.049604", "NaN"),
        "duplicate key": lambda: COMMITTED.read_text().replace(
            '"amount": 1.049604,', '"amount": 1.049604, "amount": 1.049604,'
        ),
        "not JSON": lambda: "{",
    }


@pytest.mark.spec("ARCH§6.2")
@pytest.mark.parametrize("fault", sorted(_bad_documents()))
def test_a_malformed_supplement_is_refused(fault: str, tmp_path: Path) -> None:
    path = tmp_path / "supplement.json"
    path.write_text(_bad_documents()[fault]())
    with pytest.raises(DataError):
        load_dividend_supplement(path)


@pytest.mark.spec("ARCH§6.2")
def test_a_missing_supplement_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        load_dividend_supplement(tmp_path / "absent.json")


# --- The backfill applies it ------------------------------------------------------------------


@pytest.mark.spec("ARCH§6.2", "SPEC§2")
def test_the_backfill_stores_the_supplement_after_the_feed_under_the_same_stamp(
    minutes: Any,
) -> None:
    feed = CorporateActions(
        (Dividend(SYMBOL, date(2016, 3, 16), 0.5, date(2016, 3, 10), True),), ()
    )
    rows = supplement((EX, AMOUNT), (date(2016, 6, 17), 1.05))  # the second is past the span
    store = BarStore.memory()
    result = backfill(
        store,
        FakeSource(minutes, feed),
        SYMBOL,
        FIRST,
        LAST,
        clock=SimClock(AFTER),
        supplement=rows,
    )
    assert result.stamp is HISTORY
    assert result.actions.dividends == 1
    assert result.supplement.dividends == 1  # only the row inside the span
    assert result.supplement.conflicts == ()
    assert dividends_known(store, AFTER) == [(date(2016, 3, 16), 0.5), (EX, AMOUNT)]
    # As history, the row is known at the end of its ex-date in New York, not before.
    assert (EX, AMOUNT) not in dividends_known(store, KNOWN)
    later = datetime(2016, 3, 19, 4, 0, 0, 1, tzinfo=UTC)
    assert (EX, AMOUNT) in dividends_known(store, later)


@pytest.mark.spec("ARCH§6.2")
def test_a_supplement_row_that_differs_from_the_feeds_is_a_conflict_not_an_overwrite(
    minutes: Any,
) -> None:
    feed = CorporateActions((Dividend(SYMBOL, EX, 1.0, EX, False),), ())
    store = BarStore.memory()
    result = backfill(
        store,
        FakeSource(minutes, feed),
        SYMBOL,
        FIRST,
        LAST,
        clock=SimClock(AFTER),
        supplement=supplement((EX, AMOUNT)),
    )
    assert result.actions.dividends == 1
    assert result.supplement.dividends == 0
    assert len(result.supplement.conflicts) == 1
    assert "2016-03-18" in result.supplement.conflicts[0]
    assert dividends_known(store, AFTER) == [(EX, 1.0)]  # the feed's row stands


@pytest.mark.xfail(strict=True, reason="red:B7", raises=AssertionError)
@pytest.mark.spec("ARCH§6.2")
def test_a_supplement_row_that_differs_is_a_conflict_under_now_too(minutes: Any) -> None:
    # B7a review (Codex R1): once the store has a NOW stamp, the backfill stamps NOW, and a NOW
    # write may revise a dividend. The supplement must still never revise the feed's row.
    clock = SimClock(AFTER)
    store = BarStore.memory(clock=clock)
    store.put_dividend(SYMBOL, ex_date=EX, amount=1.0, declared=EX, known_at=NOW)
    feed = CorporateActions((Dividend(SYMBOL, EX, 1.0, EX, False),), ())
    result = backfill(
        store,
        FakeSource(minutes, feed),
        SYMBOL,
        FIRST,
        LAST,
        clock=clock,
        supplement=supplement((EX, AMOUNT)),
    )
    assert result.stamp is NOW
    assert result.supplement.dividends == 0
    assert len(result.supplement.conflicts) == 1
    assert "2016-03-18" in result.supplement.conflicts[0]
    assert dividends_known(store, store.high_water or AFTER) == [(EX, 1.0)]


@pytest.mark.spec("ARCH§6.2")
def test_a_supplement_row_equal_to_the_feeds_changes_nothing(minutes: Any) -> None:
    feed = CorporateActions((Dividend(SYMBOL, EX, AMOUNT, EX, False),), ())
    store = BarStore.memory()
    result = backfill(
        store,
        FakeSource(minutes, feed),
        SYMBOL,
        FIRST,
        LAST,
        clock=SimClock(AFTER),
        supplement=supplement((EX, AMOUNT)),
    )
    assert result.supplement.conflicts == ()
    assert dividends_known(store, AFTER) == [(EX, AMOUNT)]


@pytest.mark.spec("ARCH§6.2")
def test_supplement_rows_of_another_symbol_are_left_out(minutes: Any) -> None:
    store = BarStore.memory()
    result = backfill(
        store,
        FakeSource(minutes),
        SYMBOL,
        FIRST,
        LAST,
        clock=SimClock(AFTER),
        supplement=supplement((EX, AMOUNT), symbol="QQQ"),
    )
    assert result.supplement.dividends == 0
    assert dividends_known(store, AFTER) == []


@pytest.mark.spec("ARCH§6.2", "ARCH§6.1")
def test_the_backfill_command_stores_the_committed_supplement(
    minutes: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant.cli.data as cli_data

    monkeypatch.setattr(cli_data, "load_settings", lambda path: SimpleNamespace(alpaca=None))
    monkeypatch.setattr(cli_data, "AlpacaMarketData", lambda keys, clock: FakeSource(minutes))
    monkeypatch.setattr(cli_data, "WallClock", lambda: SimClock(AFTER))
    db = tmp_path / "market.db"
    result = CliRunner().invoke(
        app,
        [
            "data",
            "backfill",
            "--db",
            str(db),
            "--start",
            "2016-03-14",
            "--end",
            "2016-03-18",
            "--supplement",
            str(COMMITTED),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    store = BarStore.open(db)
    try:
        assert dividends_known(store, AFTER) == [(EX, AMOUNT)]
    finally:
        store.close()


@pytest.mark.spec("ARCH§6.2")
def test_the_backfill_command_refuses_a_malformed_supplement_before_opening_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant.cli.data as cli_data

    monkeypatch.setattr(cli_data, "load_settings", lambda path: SimpleNamespace(alpaca=None))
    bad = tmp_path / "supplement.json"
    bad.write_text("{}")
    db = tmp_path / "market.db"
    result = CliRunner().invoke(
        app,
        ["data", "backfill", "--db", str(db), "--supplement", str(bad)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert "supplement.json" in result.output
    assert not db.exists()
