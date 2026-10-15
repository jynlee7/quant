"""B6: move flags read point in time, through the view (I1, ARCHITECTURE §6.1, §6.2).

`BarStore.flags(as_of=...)` knows when a flag was raised but not whether its minute is visible
yet, so a decision can't use it: a flag known before τ on a minute available after τ would leak
the minute. `PointInTimeView.flags` applies the view's own rules. The truncation harness's
actions layer reads it (B5 open item).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.epoch import to_us
from quant.core.errors import PolicyError
from quant.data import market_db
from quant.data.frames import MinuteFrame
from quant.data.market_db import BarStore
from quant.data.policy import Purpose
from quant.data.synthetic import synthetic_minutes
from quant.data.validate import MOVE_KIND, DataFlag
from tests.support.bars import SYMBOL

RED = pytest.mark.xfail(strict=True, reason="red:B6", raises=NotImplementedError)

DAY = date(2016, 3, 1)  # inside FIT; New York is UTC-5
OPEN = datetime(2016, 3, 1, 14, 30, tzinfo=UTC)
CLOSE = datetime(2016, 3, 1, 21, 0, tzinfo=UTC)
AVAILABLE = timedelta(minutes=16)  # a minute is available at its close plus the feed delay
US = timedelta(microseconds=1)


def jumped(at: int) -> MinuteFrame:
    """The day's minutes with the price level 6% higher from minute `at` on: a move flag on
    minute `at`."""
    m = synthetic_minutes(TradingCalendar.xnys(), DAY, DAY, seed=9)
    o, h, lo, c = (x.copy() for x in (m.open, m.high, m.low, m.close))
    for column in (o, h, lo, c):
        column[at:] *= 1.06
    return MinuteFrame(m.t, o, h, lo, c, m.volume)


def flagged_store(clock: SimClock | None = None) -> tuple[BarStore, MinuteFrame]:
    store = BarStore.memory(clock=clock)
    minutes = jumped(60)
    store.ingest_minutes(SYMBOL, minutes, known_at=market_db.HISTORY, source="backfill")
    return store, minutes


@RED
@pytest.mark.spec("I1", "ARCH§6.1", "ARCH§6.2")
def test_a_flag_shows_once_its_minute_is_available() -> None:
    store, m = flagged_store()
    t = m.at(60)
    (flag,) = store.flags(SYMBOL)
    assert flag.t == t
    ready = t + AVAILABLE
    early = store.view(ready, as_of=ready)
    assert early.flags(SYMBOL, OPEN, CLOSE, purpose=Purpose.OPERATE) == []
    later = store.view(ready + US, as_of=ready + US)
    assert later.flags(SYMBOL, OPEN, CLOSE, purpose=Purpose.OPERATE) == [flag]


@RED
@pytest.mark.spec("I1", "ARCH§6.2")
def test_a_flag_known_early_on_a_minute_the_view_hides_stays_hidden() -> None:
    store, m = flagged_store()
    # Flags known at the open on a minute not yet available and on a minute never stored:
    # the store's flags show them, the view's must not.
    hidden, absent = m.at(200), CLOSE + timedelta(hours=1)
    for t in (hidden, absent):
        store._conn.execute(
            "INSERT INTO data_flags VALUES (?, ?, ?, 1, 1, 'planted', ?)",
            (SYMBOL, to_us(t), MOVE_KIND, to_us(OPEN)),
        )
    tau = hidden  # minute 200 starts at τ: not visible
    assert {hidden, absent} <= {f.t for f in store.flags(SYMBOL, as_of=tau)}
    view = store.view(tau, as_of=tau)
    assert view.flags(SYMBOL, OPEN, CLOSE + timedelta(hours=2), purpose=Purpose.OPERATE) == [
        DataFlag(m.at(60), MOVE_KIND, store.flags(SYMBOL)[0].detail)
    ]


@RED
@pytest.mark.spec("I1", "ARCH§6.1", "ARCH§6.2")
def test_the_latest_version_known_before_as_of_counts() -> None:
    # A settle lifts minute 59 to the new level, so the jump moves into 59, and the flag on 60
    # is cleared by a new version: before the settle the view sees 60, after it 59.
    settled_at = CLOSE + timedelta(hours=1)
    clock = SimClock(settled_at)
    store, m = flagged_store(clock)
    one = m.select(m.t == to_us(m.at(59)))
    settle = MinuteFrame(
        one.t, one.open * 1.06, one.high * 1.06, one.low * 1.06, one.close * 1.06, one.volume
    )
    store.ingest_minutes(SYMBOL, settle, known_at=market_db.NOW, source="settle")
    before = store.view(settled_at, as_of=settled_at)
    assert [f.t for f in before.flags(SYMBOL, OPEN, CLOSE, purpose=Purpose.OPERATE)] == [m.at(60)]
    after = store.view(settled_at + US, as_of=settled_at + US)
    assert [f.t for f in after.flags(SYMBOL, OPEN, CLOSE, purpose=Purpose.OPERATE)] == [m.at(59)]


@RED
@pytest.mark.spec("I1", "ARCH§6.2")
def test_the_span_is_half_open() -> None:
    store, m = flagged_store()
    t = m.at(60)
    view = store.view(CLOSE + timedelta(hours=1), as_of=CLOSE + timedelta(hours=1))
    assert [f.t for f in view.flags(SYMBOL, t, t + US, purpose=Purpose.OPERATE)] == [t]
    assert view.flags(SYMBOL, t - timedelta(minutes=1), t, purpose=Purpose.OPERATE) == []
    assert view.flags(SYMBOL, t + US, CLOSE, purpose=Purpose.OPERATE) == []


@RED
@pytest.mark.spec("I1", "A2", "ARCH§6.2", "ARCH§6.3")
def test_a_flags_read_names_its_purpose_and_goes_into_the_digest() -> None:
    store, _ = flagged_store()
    tau = CLOSE + timedelta(hours=1)
    read, again, unread = (store.view(tau, as_of=tau) for _ in range(3))
    assert read.flags(SYMBOL, OPEN, CLOSE, purpose=Purpose.OPERATE)
    again.flags(SYMBOL, OPEN, CLOSE, purpose=Purpose.OPERATE)
    assert read.digest() == again.digest() != unread.digest()
    # Scoring DEV needs a trial ticket (C11): the flags read is checked like any other.
    dev = datetime(2019, 3, 1, 21, tzinfo=UTC)
    with pytest.raises(PolicyError):
        store.view(dev, as_of=dev).flags(
            SYMBOL, dev - timedelta(hours=7), dev, purpose=Purpose.SCORE
        )
