"""B2: data segments and purpose-tagged reads (A2, C1, ARCHITECTURE §6.3).

Every read names its purpose. Operating the engine (features, the filter, refits) may read any
market data, the lockbox included (C1). Scoring a strategy may read only FIT and DEV, and nothing
may score or report on the lockbox before the go-live test (A2).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from quant.core.errors import LockboxError, PolicyError
from quant.data.policy import DataPolicy, Purpose, Segment, Segments
from tests.support.bars import SYMBOL, store_with_days


@pytest.mark.spec("A2", "C1", "ARCH§6.3")
def test_days_fall_in_the_segments_the_spec_names() -> None:
    open_lockbox = Segments()
    assert open_lockbox.segment_of(date(2016, 1, 4)) is Segment.FIT
    assert open_lockbox.segment_of(date(2018, 12, 31)) is Segment.FIT
    assert open_lockbox.segment_of(date(2019, 1, 2)) is Segment.DEV
    assert open_lockbox.segment_of(date(2024, 12, 31)) is Segment.DEV
    assert open_lockbox.segment_of(date(2025, 1, 2)) is Segment.LOCKBOX
    assert open_lockbox.segment_of(date(2026, 10, 5)) is Segment.LOCKBOX  # not sealed yet
    sealed = Segments(lockbox_end=date(2026, 10, 20))
    assert sealed.segment_of(date(2026, 10, 20)) is Segment.LOCKBOX
    assert sealed.segment_of(date(2026, 10, 21)) is Segment.FORWARD


@pytest.mark.spec("A2", "C1", "ARCH§6.3")
def test_the_policy_refuses_to_score_or_report_on_the_lockbox() -> None:
    policy = DataPolicy(Segments(lockbox_end=date(2026, 10, 20)))
    policy.check(Purpose.OPERATE, date(2016, 1, 4), date(2026, 11, 30))  # C1: market data only
    policy.check(Purpose.SCORE, date(2016, 1, 4), date(2024, 12, 31))
    policy.check(Purpose.REPORT, date(2026, 10, 21), date(2026, 11, 30))
    for purpose in (Purpose.SCORE, Purpose.REPORT):
        with pytest.raises(LockboxError):
            policy.check(purpose, date(2024, 12, 1), date(2025, 1, 2))
        with pytest.raises(LockboxError):
            policy.check(purpose, date(2025, 6, 1), date(2025, 6, 30))
    with pytest.raises(PolicyError):
        policy.check(Purpose.SCORE, date(2026, 10, 21), date(2026, 11, 30))


@pytest.mark.spec("A2", "C1", "ARCH§6.3")
def test_the_view_enforces_the_policy_on_every_read() -> None:
    store = store_with_days(date(2024, 12, 30), date(2025, 1, 3))
    tau = datetime(2025, 1, 4, tzinfo=UTC)
    view = store.view(tau)
    assert len(view.hourly(SYMBOL, purpose=Purpose.OPERATE)) > 0
    with pytest.raises(LockboxError):
        view.hourly(SYMBOL, purpose=Purpose.SCORE)
    with pytest.raises(LockboxError):
        view.minutes(SYMBOL, datetime(2025, 1, 2, 14, tzinfo=UTC), tau, purpose=Purpose.SCORE)
    dev_only = store.view(datetime(2024, 12, 31, 23, tzinfo=UTC))
    assert len(dev_only.hourly(SYMBOL, purpose=Purpose.SCORE)) == 7 + 7
