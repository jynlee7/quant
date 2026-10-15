"""B6: the data-quality report finds every gap on a planted store and says which are unexplained
(PLAN §3 CP1, ARCHITECTURE §6.1). It checks data in every segment, the lockbox included, without
scoring anything (C1)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest

from quant.core.calendar import TradingCalendar
from quant.core.canonical import canonical_dumps, sha256_hex
from quant.core.epoch import to_us
from quant.data.frames import MinuteFrame
from quant.data.market_db import HISTORY, BarStore
from quant.data.quality import GapKind, quality_report
from tests.support.bars import SYMBOL, synthetic_days, with_close

FIRST, LAST = date(2017, 11, 20), date(2017, 11, 28)  # 20, 21, 22, 24 (half), 27, 28
XNYS = TradingCalendar.xnys()


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def without(minutes: MinuteFrame, start: datetime, end: datetime) -> MinuteFrame:
    return minutes.select((minutes.t < to_us(start)) | (minutes.t >= to_us(end)))


def planted_store() -> BarStore:
    """Thanksgiving week 2017 with a session withheld, three minutes and a bar's last minute
    missing, and a 6% one-minute move."""
    minutes = synthetic_days(FIRST, LAST, seed=11)
    withheld = XNYS.session(date(2017, 11, 21))
    minutes = without(minutes, withheld.open, withheld.close)
    minutes = without(minutes, utc(2017, 11, 22, 15, 15), utc(2017, 11, 22, 15, 18))
    minutes = without(minutes, utc(2017, 11, 27, 17, 29), utc(2017, 11, 27, 17, 30))
    jump = int(np.flatnonzero(minutes.t == to_us(utc(2017, 11, 28, 16, 0)))[0])
    minutes = with_close(minutes, jump, float(minutes.close[jump - 1]) * 1.06)
    store = BarStore.memory()
    store.ingest_minutes(SYMBOL, minutes, known_at=HISTORY, source="synthetic")
    return store


@pytest.mark.spec("ARCH§6.1", "SPEC§2", "PLAN§3")
def test_the_report_finds_every_gap_on_a_planted_store() -> None:
    report = quality_report(planted_store(), SYMBOL, FIRST, LAST)

    assert report.sessions == tuple(XNYS.sessions(FIRST, LAST))
    assert report.half_days == (date(2017, 11, 24),)
    assert report.minutes_expected == 5 * 390 + 210
    assert report.minutes_present == 5 * 390 + 210 - 390 - 3 - 1

    gaps = [(g.kind, g.session, g.start, g.minutes, g.key) for g in report.gaps]
    assert gaps == [
        (GapKind.MISSING_SESSION, date(2017, 11, 21), utc(2017, 11, 21, 14, 30), 390, "2017-11-21"),
        (
            GapKind.MISSING_MINUTES,
            date(2017, 11, 22),
            utc(2017, 11, 22, 15, 15),
            3,
            "2017-11-22T15:15:00Z",
        ),
        (
            GapKind.MISSING_MINUTES,
            date(2017, 11, 27),
            utc(2017, 11, 27, 17, 29),
            1,
            "2017-11-27T17:29:00Z",
        ),
    ]
    # 11:30-12:30 ET on the 27th lost its last minute, so decisions treat it as missing.
    assert report.incomplete_bars == (utc(2017, 11, 27, 16, 30),)
    assert utc(2017, 11, 28, 16, 0) in [flag.t for flag in report.flags]

    assert report.unexplained == report.gaps
    assert not report.ok


@pytest.mark.spec("ARCH§6.1", "PLAN§3")
def test_every_gap_must_be_explained() -> None:
    store = planted_store()
    explanations = {
        "2017-11-21": "planted: the session was withheld",
        "2017-11-22T15:15:00Z": "planted: three minutes removed",
    }
    partly = quality_report(store, SYMBOL, FIRST, LAST, explanations=explanations)
    assert [gap.key for gap in partly.unexplained] == ["2017-11-27T17:29:00Z"]
    assert not partly.ok
    assert partly.gaps[0].explanation == "planted: the session was withheld"

    explanations["2017-11-27T17:29:00Z"] = "planted: a bar's last minute removed"
    full = quality_report(store, SYMBOL, FIRST, LAST, explanations=explanations)
    assert full.unexplained == ()
    assert full.ok


@pytest.mark.spec("PLAN§3")
def test_the_report_is_canonical_json() -> None:
    store = planted_store()
    first = quality_report(store, SYMBOL, FIRST, LAST).to_json()
    again = quality_report(store, SYMBOL, FIRST, LAST).to_json()
    assert sha256_hex(first) == sha256_hex(again)
    text = canonical_dumps(first).decode()
    for key in ("2017-11-21", "2017-11-22T15:15:00Z", "2017-11-27T17:29:00Z"):
        assert key in text
    assert GapKind.MISSING_SESSION.value in text


@pytest.mark.spec("ARCH§6.1")
def test_a_clean_store_has_no_gaps() -> None:
    store = BarStore.memory()
    store.ingest_minutes(
        SYMBOL, synthetic_days(FIRST, LAST, seed=11), known_at=HISTORY, source="synthetic"
    )
    report = quality_report(store, SYMBOL, FIRST, LAST)
    assert report.gaps == ()
    assert report.incomplete_bars == ()
    assert report.minutes_present == report.minutes_expected
    assert report.ok


@pytest.mark.spec("C1", "ARCH§6.3")
def test_the_report_checks_lockbox_data_without_a_ticket() -> None:
    first, last = date(2025, 1, 2), date(2025, 1, 3)
    store = BarStore.memory()
    store.ingest_minutes(
        SYMBOL, synthetic_days(first, last, seed=11), known_at=HISTORY, source="synthetic"
    )
    report = quality_report(store, SYMBOL, first, last)
    assert report.sessions == (first, last)
    assert report.ok
