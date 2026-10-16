"""B7's "done when" (PLAN §4 B7): monthly refits through 2018 on real FIT data, starting from a
model selected on 2016–2017, with the calibration each pass builds (spec §5, §9; ARCHITECTURE
§7.3, §7.4, §10's walk-forward; C13; T-NL2). Full tier only.

The first model is selected on 2016–2017 with B4's selection (`select_first_model`, r1's grid),
not given the CP1 model's K. Then the evening job runs at 17:30 New York on every session of
2018, exactly as live trading and the walk-forward run it, on r1's training, drift and
calibration configs. The store is an in-memory copy of var/market.db with every row known after
FIT's end deleted, so nothing here can read DEV (C11); it skips when var/market.db is absent.
Nothing computes strategy P&L.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.calibration import load_calibration, r1_calibration_config, save_calibration
from quant.core.clock import SimClock
from quant.core.epoch import to_us
from quant.data.market_db import BarStore
from quant.data.policy import DEV_START, FIT_START
from quant.features.menu import r1_features
from quant.hmm.drift import r1_drift_thresholds
from quant.hmm.labels import r1_thresholds
from quant.hmm.model import load_model
from quant.hmm.train import r1_train_config
from quant.store.ops import OpsStore, RefitOutcomeKind
from tests.support.truncation import MARKET_DB, known_at_tables

SYMBOL = "SPY"
FIRST_LAST = date(2017, 12, 29)  # the last session of 2017
DEV_MIDNIGHT = datetime.combine(DEV_START, time(), tzinfo=NEW_YORK).astimezone(UTC)
EVENING = time(17, 30)
PROVENANCE = {"source": "B7b full tier: monthly refits through 2018"}


def _fit_only_store() -> BarStore:
    """var/market.db copied into memory with every row known at or after DEV's first New York
    midnight deleted from every table with a known_at column: DEV stays dark (C11)."""
    source = sqlite3.connect(f"{MARKET_DB.resolve().as_uri()}?mode=ro", uri=True)
    conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        source.backup(conn)
    finally:
        source.close()
    for table in known_at_tables(conn):
        conn.execute(
            f"DELETE FROM {table} WHERE known_at >= ?",  # noqa: S608 — names from sqlite_master
            (to_us(DEV_MIDNIGHT),),
        )
    return BarStore(conn)


@pytest.mark.full
@pytest.mark.spec("SPEC§5", "SPEC§9", "C13", "T-NL2", "ARCH§7.3", "ARCH§7.4", "C7", "C11")
def test_monthly_refits_through_2018_on_real_fit_data(tmp_path: Path) -> None:
    if not MARKET_DB.is_file():
        pytest.skip(f"no market.db at {MARKET_DB}: fill it with `quant data backfill`")
    from quant.jobs.calibrate import calibrate
    from quant.jobs.first_model import select_first_model
    from quant.jobs.refit import refit

    store = _fit_only_store()
    cal = TradingCalendar.xnys()
    features = r1_features()
    calibration_config = r1_calibration_config()
    with capture_logs():
        first = select_first_model(
            store,
            symbol=SYMBOL,
            first=FIT_START,
            last=FIRST_LAST,
            features=features,
            config=r1_train_config(),
            thresholds=r1_thresholds(),
            provenance=PROVENANCE,
            artifacts=tmp_path,
            clock=SimClock(DEV_MIDNIGHT),
        )
    start = first.result.model
    assert start.fit_end == datetime(2017, 12, 30, 5, tzinfo=UTC)  # midnight after 12-29
    assert start.training.info["selection"] is not None  # B4's grid chose K, not CP1

    clock = SimClock(start.fit_end)
    ops = OpsStore.open(tmp_path / "ops.db", clock=clock)
    ops.activate("model", start.artifact_hash(), reason="first model")
    initial = calibrate(
        store,
        ops,
        as_of=start.fit_end,
        symbol=SYMBOL,
        features=features,
        artifacts=tmp_path,
        config=calibration_config,
    )
    assert initial.document.bars == 0
    save_calibration(initial, tmp_path / "calibrations")
    ops.activate("calibration", initial.artifact_hash(), reason="first model")

    models = {start.artifact_hash(): start}
    outcomes = []
    for day in cal.sessions(date(2018, 1, 2), date(2018, 12, 31)):
        at = datetime.combine(day, EVENING, tzinfo=NEW_YORK).astimezone(UTC)
        clock.advance_to(at)
        current = models[ops.current("model", at).artifact]
        with capture_logs():
            outcome = refit(
                store,
                current,
                at=at,
                symbol=SYMBOL,
                features=features,
                ops=ops,
                artifacts=tmp_path,
                config=r1_train_config(),
                thresholds=r1_thresholds(),
                drift=r1_drift_thresholds(),
                provenance=PROVENANCE,
                calibration=calibration_config,
            )
        if outcome is not None:
            outcomes.append(outcome)
            if outcome.model is not None:
                models[outcome.model.artifact_hash()] = outcome.model

    # Monthly: every 30 New York days or more apart, all inside FIT.
    events = ops.refit_events()
    assert len(events) == len(outcomes) >= 10
    cutoffs = [e.cutoff for e in events]
    assert all(c < DEV_MIDNIGHT for c in cutoffs)
    days = [c.astimezone(NEW_YORK).date() for c in cutoffs]
    assert all((b - a).days >= 30 for a, b in pairwise(days))
    # K and the covariance type never change, and every window expands from FIT's start.
    for outcome in outcomes:
        if outcome.model is not None:
            assert outcome.model.params.n_states == start.params.n_states
            assert outcome.model.params.covariance_type == start.params.covariance_type
            assert outcome.model.fit_start == start.fit_start
            assert outcome.model.fit_end == outcome.cutoff
            assert load_model(outcome.directory).artifact_hash() == outcome.model.artifact_hash()

    # Each pass built a calibration as of its cutoff, activated with its model at the next
    # open; nothing else moved the calibration pointer.
    activated = [o for o in outcomes if o.kind is RefitOutcomeKind.ACTIVATED]
    calibrations = ops.pointers("calibration")
    assert len(calibrations) == 1 + len(activated)
    for outcome, pointer in zip(activated, calibrations[1:], strict=True):
        assert outcome.calibration is not None
        assert outcome.calibration.as_of == outcome.cutoff
        assert pointer.artifact == outcome.calibration.artifact_hash()
        assert pointer.effective_at == outcome.event.effective_at > outcome.cutoff
        document = outcome.calibration.document
        assert document.scored == max(0, document.bars - 35)
        assert document.sufficient == (document.scored >= 1000)
        assert load_calibration(outcome.calibration_directory).artifact_hash() == (
            outcome.calibration.artifact_hash()
        )
    for outcome in outcomes:
        if outcome.kind is not RefitOutcomeKind.ACTIVATED:
            assert outcome.calibration is None

    # T-NL2 at every open of 2018 and DEV's first: the governing model's data cutoff and the
    # governing calibration's as-of time both precede it.
    opens = [cal.session(day).open for day in cal.sessions(date(2018, 1, 2), date(2019, 1, 2))]
    for instant in opens:
        model = models[ops.current("model", instant).artifact]
        assert model.fit_end < instant
        pointer = ops.current("calibration", instant)
        directory = tmp_path / "calibrations" / pointer.artifact
        load_calibration(directory).require_before(instant)

    # A nightly report calibration as of 2018's last evening scores the year's governed bars:
    # enough for a verdict (C13).
    year = calibrate(
        store,
        ops,
        as_of=datetime(2018, 12, 31, 17, tzinfo=NEW_YORK).astimezone(UTC),
        symbol=SYMBOL,
        features=features,
        artifacts=tmp_path,
        config=calibration_config,
    )
    assert year.document.sufficient
    assert year.document.bars > 1500
    assert year.document.models[0] == start.artifact_hash()
    ops.close()
    store.close()
    # For the BUILD_LOG: what the year's refits and calibration found.
    print(
        {
            "K": start.params.n_states,
            "covariance": start.params.covariance_type,
            "refits": [(e.cutoff.date().isoformat(), e.outcome.value) for e in events],
            "labels": {
                v.label.value: (v.passed, v.raw.ece, v.raw.skill) for v in year.document.labels
            },
            "outcome": (year.document.outcome.covered, year.document.outcome.n),
        }
    )
    assert timedelta(0) < opens[-1] - DEV_MIDNIGHT  # DEV's first open, read only as a time
