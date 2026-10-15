"""The monthly refit (spec §5, ARCHITECTURE §7.1, §7.3; C5, C7). Built in B7.

An evening job. The live scheduler (B13) and the backtest (B11) call the same function at the
same simulated times, each with the refit's instant `at` read from its Clock (K6).

1. **Due** (`refit_due`): every REFIT_DAYS calendar days, counted from the *active* model's
   last data day, the New York date of the last instant of its fit window, `(fit_end - 1 µs)`
   in New York. The refit is due when the cutoff's New York date is at least REFIT_DAYS days
   later. So a skipped evening refits on the next one, and DST can't move the cadence. A job
   that isn't due records nothing (`refit` returns None).
2. **The cutoff** (`refit_cutoff`): the latest CUTOFF_ET (17:00 New York) at or before `at`:
   the bars available at 17:00 (ARCHITECTURE §7.3).
3. **The rows** (`refit_data`): an expanding window [current.fit_start, cutoff), read only
   through the point-in-time view at τ = as_of = cutoff, with purpose OPERATE, as
   `quant.jobs.first_model.training_data` reads FIT: the feature stream starts at FIT's first
   New York midnight and the window is cut with `FeatureMatrix.window`; returns and sessions
   follow the same rule. Nothing known at or after the cutoff reaches the refit. Nothing here
   refuses DEV: the caller picks the instants (B7's full tier stops at 2018-12-31; C11).
4. **The fit** (`run_refit`): `train_candidate` with the current model's K and covariance type
   (fixed: re-selection is a gated change, never a refit) and `previous=current`, so one
   restart starts from the current model (`quant.hmm.train`, "The seeded restart").
5. **Sanity reject** (spec §4, ARCHITECTURE §7.3): a fit whose SanityReport fails, or that
   raises ModelError (a state with no filtered mass, every restart failing), is REJECTED: not
   saved, the current model stays, the pointer and the freeze don't change, and a structlog
   warning REJECT_EVENT is the alert.
6. **Matching and drift**: `match_states(current, new)`, then the stored model is
   `inherit_labels(new, match, previous=current hash)`, then `refit_drift(current, stored,
   match, thresholds)`.
7. **Store**: the stored model is saved (`save_model(model, artifacts / "models")`) whether or
   not it passes. A pass is ACTIVATED: the model pointer moves at the next open after the
   recording (C7). A failed drift check is FROZEN: stored, not activated, entries frozen (C5).
   Either way `OpsStore.record_refit` writes the event and its consequences in one transaction,
   and a structlog event names the outcome (info ACTIVATE_EVENT, warning FREEZE_EVENT).

The calibration artifact a pass also builds (ARCHITECTURE §7.3 step 5) is B7b's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import structlog

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.clock import ensure_utc
from quant.core.epoch import to_us
from quant.core.errors import ModelError
from quant.data.market_db import BarStore
from quant.data.policy import FIT_START, DataPolicy, Purpose
from quant.features.menu import FeatureConfig
from quant.features.stream import compute_features
from quant.hmm.drift import DriftReport, DriftThresholds, refit_drift
from quant.hmm.labels import LabelThresholds
from quant.hmm.match import StateMatch, inherit_labels, match_states
from quant.hmm.model import HmmModel, save_model
from quant.hmm.select import Candidate, SanityReport
from quant.hmm.train import TrainConfig, TrainingData, train_candidate
from quant.store.ops import OpsStore, RefitEvent, RefitOutcomeKind

# Spec §5: a refit every 30 calendar days.
REFIT_DAYS: Final = 30
# ARCHITECTURE §7.3: the refit reads the bars available at 17:00 New York time.
CUTOFF_ET: Final = time(17, 0)
# The structlog events that report each outcome; the rejection and the freeze are alerts.
ACTIVATE_EVENT: Final = "refit activated"
FREEZE_EVENT: Final = "refit drift: entries frozen"
REJECT_EVENT: Final = "refit rejected"


def refit_cutoff(at: datetime) -> datetime:
    """The refit's data cutoff, in UTC: CUTOFF_ET on `at`'s New York date when `at` is at or
    after it, else CUTOFF_ET on the day before (module docstring, step 2). ClockError for a
    naive `at`."""
    local = ensure_utc(at).astimezone(NEW_YORK)
    cutoff = datetime.combine(local.date(), CUTOFF_ET, tzinfo=NEW_YORK)
    if local < cutoff:
        cutoff -= timedelta(days=1)
    return cutoff.astimezone(UTC)


def refit_due(current: HmmModel, at: datetime) -> bool:
    """Whether a refit is due at `at` under the active model `current` (module docstring, step
    1): `refit_cutoff(at)`'s New York date minus `current`'s last data day (the New York date
    of `current.fit_end - 1 µs`) is at least REFIT_DAYS days. ClockError for a naive `at`."""
    last = (current.fit_end - timedelta(microseconds=1)).astimezone(NEW_YORK).date()
    return (refit_cutoff(at).astimezone(NEW_YORK).date() - last).days >= REFIT_DAYS


def refit_data(
    store: BarStore,
    symbol: str,
    *,
    start: datetime,
    cutoff: datetime,
    features: FeatureConfig,
    policy: DataPolicy | None = None,
) -> TrainingData:
    """The refit's rows (module docstring, step 3): the window [start, cutoff), read through
    `store.view(cutoff, as_of=cutoff, policy=policy).hourly(symbol, purpose=Purpose.OPERATE,
    since=<FIT_START's New York midnight>)`, with `compute_features(bars, features).window(
    start, cutoff)`, and `returns`, `sessions` and `feature_config_hash` exactly as
    `quant.jobs.first_model.training_data` builds them (complete bars only; each row's return
    against the complete bar before it). The result's fit_start is `start` and its fit_end
    `cutoff`.

    Equal for any two stores that agree on what was known before the cutoff: a bar, revision or
    session known at or after it never changes the rows. Raises ModelError (from TrainingData)
    when fewer than two warm rows remain, ClockError for a naive time, and ModelError when
    start >= cutoff."""
    start, cutoff = ensure_utc(start), ensure_utc(cutoff)
    if start >= cutoff:
        raise ModelError("refit window must have positive length")
    origin = datetime.combine(FIT_START, time(), tzinfo=NEW_YORK).astimezone(UTC)
    bars = store.view(cutoff, as_of=cutoff, policy=policy).hourly(
        symbol, purpose=Purpose.OPERATE, since=origin
    )
    matrix = compute_features(bars, features).window(start, cutoff)
    by_open = {
        to_us(bar.t_open): (math.log(bar.close / previous.close), bar.session.toordinal())
        for previous, bar in pairwise(bar for bar in bars if bar.complete)
    }
    return TrainingData(
        matrix,
        np.array([by_open[int(t)][0] for t in matrix.t_open], dtype=np.float64),
        np.array([by_open[int(t)][1] for t in matrix.t_open], dtype=np.int64),
        start,
        cutoff,
        features.config_hash(),
    )


@dataclass(frozen=True, eq=False)
class RefitOutcome:
    """What one refit did (module docstring).

    - `kind`: ACTIVATED, FROZEN or REJECTED;
    - `at`, `cutoff`: the job's instant and its data cutoff (the data's fit_end);
    - `previous`: the current model's artifact hash;
    - `model`: the stored model (inherited labels), None when REJECTED;
    - `directory`: its artifact directory, None when REJECTED;
    - `match`, `drift`: the matching and the drift report, None when REJECTED;
    - `sanity`: the new fit's SanityReport, None when training raised ModelError;
    - `reasons`: the drift reasons (FROZEN), the sanity reasons or the ModelError's message
      (REJECTED), () when ACTIVATED;
    - `event`: the RefitEvent recorded in ops.db (its `effective_at` is the activation's).
    """

    kind: RefitOutcomeKind
    at: datetime
    cutoff: datetime
    previous: str
    model: HmmModel | None
    directory: Path | None
    match: StateMatch | None
    drift: DriftReport | None
    sanity: SanityReport | None
    reasons: tuple[str, ...]
    event: RefitEvent


def run_refit(
    data: TrainingData,
    current: HmmModel,
    *,
    at: datetime,
    ops: OpsStore,
    artifacts: Path,
    config: TrainConfig,
    thresholds: LabelThresholds,
    drift: DriftThresholds,
    provenance: Mapping[str, Any],
    calendar: TradingCalendar | None = None,
    workers: int | None = None,
) -> RefitOutcome:
    """Refit `current` on `data` and record the outcome (module docstring, steps 4–7). `data`
    is the refit's window, normally `refit_data(...)` at `refit_cutoff(at)`; the event's
    `cutoff` is `data.fit_end`. `config` gives the restarts, seeds, EM settings, bootstrap
    draws and validation block (its `candidates` are ignored); `calendar` (XNYS when None)
    gives the next open. The ops store should share the caller's Clock, so the recording's
    stamp is `at` or later.

    Raises ModelError, before training and writing nothing, when: the window doesn't expand
    `current`'s (`data.fit_start != current.fit_start`, or `data.fit_end <= current.fit_end`);
    `data.fit_end > at` (rows from the future); the feature columns or feature-config hash
    aren't `current`'s; or ops.db's model pointer at `at` doesn't name `current`
    (`ops.current("model", at)`), so a refit is always of the active model. ClockError for a
    naive `at`. Training's own ModelError is a REJECTED outcome, not an exception.

    Training goes through this module's `train_candidate` name (`train_candidate(data,
    Candidate(K, covariance type of current), thresholds=..., config=..., provenance=...,
    workers=..., previous=current)`), which the tests replace to plant a sanity reject."""
    at = ensure_utc(at)
    previous = current.artifact_hash()
    pointer = ops.current("model", at)
    if (
        data.fit_start != current.fit_start
        or data.fit_end <= current.fit_end
        or data.fit_end > at
        or data.matrix.columns != current.scaler.columns
        or data.feature_config_hash != current.feature_config_hash
        or pointer is None
        or pointer.artifact != previous
    ):
        raise ModelError("refit requires an expanding past window of the active model")
    sanity = None
    model = None
    directory = None
    match = None
    report = None
    try:
        result = train_candidate(
            data,
            Candidate(
                n_states=current.params.n_states, covariance_type=current.params.covariance_type
            ),
            thresholds=thresholds,
            config=config,
            provenance=provenance,
            workers=workers,
            previous=current,
        )
        sanity = result.sanity
        reasons = sanity.reasons
    except ModelError as exc:
        reasons = (str(exc),)
    if reasons:
        kind = RefitOutcomeKind.REJECTED
    else:
        match = match_states(current, result.model)
        model = inherit_labels(result.model, match, previous=previous)
        report = refit_drift(current, model, match, drift)
        reasons = report.reasons
        kind = RefitOutcomeKind.ACTIVATED if report.passed else RefitOutcomeKind.FROZEN
        directory = save_model(model, artifacts / "models")
    event = ops.record_refit(
        at=at,
        cutoff=data.fit_end,
        outcome=kind,
        artifact=None if model is None else model.artifact_hash(),
        previous=previous,
        reasons=reasons,
        calendar=calendar,
    )
    logger = structlog.get_logger()
    if kind is RefitOutcomeKind.ACTIVATED:
        logger.info(ACTIVATE_EVENT, artifact=event.artifact, effective_at=event.effective_at)
    else:
        logger.warning(
            REJECT_EVENT if kind is RefitOutcomeKind.REJECTED else FREEZE_EVENT,
            previous=previous,
            reasons=reasons,
            artifact=event.artifact,
        )
    return RefitOutcome(
        kind, at, data.fit_end, previous, model, directory, match, report, sanity, reasons, event
    )


def refit(
    store: BarStore,
    current: HmmModel,
    *,
    at: datetime,
    symbol: str,
    features: FeatureConfig,
    ops: OpsStore,
    artifacts: Path,
    config: TrainConfig,
    thresholds: LabelThresholds,
    drift: DriftThresholds,
    provenance: Mapping[str, Any],
    calendar: TradingCalendar | None = None,
    workers: int | None = None,
    policy: DataPolicy | None = None,
) -> RefitOutcome | None:
    """The evening job: None, reading and writing nothing, when `refit_due(current, at)` is
    false; otherwise `run_refit(refit_data(store, symbol, start=current.fit_start,
    cutoff=refit_cutoff(at), features=features, policy=policy), current, at=at, ...)` with the
    other arguments passed through."""
    if not refit_due(current, at):
        return None
    data = refit_data(
        store,
        symbol,
        start=current.fit_start,
        cutoff=refit_cutoff(at),
        features=features,
        policy=policy,
    )
    return run_refit(
        data,
        current,
        at=at,
        ops=ops,
        artifacts=artifacts,
        config=config,
        thresholds=thresholds,
        drift=drift,
        provenance=provenance,
        calendar=calendar,
        workers=workers,
    )
