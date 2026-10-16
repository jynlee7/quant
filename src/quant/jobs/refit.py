"""The monthly refit (spec §5, ARCHITECTURE §7.1, §7.3; C5, C7). Built in B7.

An evening job. The live scheduler (B13) and the backtest (B11) call the same function at the
same simulated times, each with the refit's instant `at` read from its Clock (K6).

1. **Due** (`refit_due`): every REFIT_DAYS New York calendar days, counted from the
   later of the active model's last data day (`fit_end - 1 µs`) and the latest refit event's
   cutoff day, whether ACTIVATED, FROZEN or REJECTED. A skipped evening refits on the next
   one; DST can't move the cadence. A job that isn't due records nothing (returns None).
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

8. **The calibration** (ARCHITECTURE §7.3 step 5, §7.4; B7b): a pass builds a calibration as
   of the refit's cutoff and moves the calibration pointer with the model's. `refit` hands
   `run_refit` a builder (`quant.jobs.calibrate.calibrate` over the job's store, ops store,
   symbol, features, artifacts and policy, with `calibration`'s config, r1's when None).
   `run_refit` calls it only when the outcome is ACTIVATED, after training and before recording:
   the builder's calibration must be as of `data.fit_end` (ModelError otherwise, recording
   nothing); it is saved under `artifacts / "calibrations"`
   (`quant.core.calibration.save_calibration`), and `OpsStore.record_refit(...,
   calibration=<its hash>)` appends both pointers in the refit's one transaction, effective at
   the same next open. Its as-of time (the cutoff) is before that open, so before every decision
   that uses it (T-NL2). A FROZEN or REJECTED refit builds nothing and moves no calibration
   pointer. The calibration scores only the models that governed before the cutoff: the new
   model's pointer isn't effective yet. A builder that raises leaves no event: the job failed,
   and is due again the next evening.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final

import structlog

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.calibration import Calibration, CalibrationConfig, r1_calibration_config
from quant.core.clock import ensure_utc
from quant.core.errors import ModelError
from quant.data.market_db import BarStore
from quant.data.policy import DataPolicy
from quant.features.menu import FeatureConfig
from quant.hmm.drift import DriftReport, DriftThresholds, refit_drift
from quant.hmm.labels import LabelThresholds
from quant.hmm.match import StateMatch, inherit_labels, match_states
from quant.hmm.model import HmmModel, save_model
from quant.hmm.select import Candidate, SanityReport
from quant.hmm.train import TrainConfig, TrainingData, train_candidate
from quant.jobs._rows import training_rows
from quant.jobs.calibrate import calibrate
from quant.store.ops import OpsStore, RefitEvent, RefitOutcomeKind

# Spec §5: a refit every 30 calendar days.
REFIT_DAYS: Final = 30
# ARCHITECTURE §7.3: the refit reads the bars available at 17:00 New York time.
CUTOFF_ET: Final = time(17, 0)
# The structlog events that report each outcome; the rejection and the freeze are alerts.
ACTIVATE_EVENT: Final = "refit activated"
FREEZE_EVENT: Final = "refit drift: entries frozen"
REJECT_EVENT: Final = "refit rejected"
# Builds a pass's calibration as of the given cutoff (module docstring, step 8).
CalibrationBuilder = Callable[[datetime], Calibration]


def refit_cutoff(at: datetime) -> datetime:
    """The refit's data cutoff, in UTC: CUTOFF_ET on `at`'s New York date when `at` is at or
    after it, else CUTOFF_ET on the day before (module docstring, step 2). ClockError for a
    naive `at`."""
    local = ensure_utc(at).astimezone(NEW_YORK)
    cutoff = datetime.combine(local.date(), CUTOFF_ET, tzinfo=NEW_YORK)
    if local < cutoff:
        cutoff -= timedelta(days=1)
    return cutoff.astimezone(UTC)


def refit_due(current: HmmModel, at: datetime, *, last_refit: datetime | None = None) -> bool:
    """Whether a refit is due at `at` (module docstring, step 1): `refit_cutoff(at)`'s New
    York date minus the later of `current`'s last data day (the New York date of
    `current.fit_end - 1 µs`) and `last_refit`'s New York date (the latest refit event's
    cutoff, whatever its outcome) is at least REFIT_DAYS days. ClockError for a naive time."""
    last = (current.fit_end - timedelta(microseconds=1)).astimezone(NEW_YORK).date()
    if last_refit is not None:
        last = max(last, ensure_utc(last_refit).astimezone(NEW_YORK).date())
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
    start >= cutoff. Raises DataError naming any sessions missing hourly bars in the view."""
    start, cutoff = ensure_utc(start), ensure_utc(cutoff)
    if start >= cutoff:
        raise ModelError("refit window must have positive length")
    return training_rows(
        store,
        symbol,
        start=start,
        end=cutoff,
        first=start.astimezone(NEW_YORK).date(),
        last=cutoff.astimezone(NEW_YORK).date(),
        features=features,
        policy=policy,
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
    - `event`: the RefitEvent recorded in ops.db (its `effective_at` is the activation's);
    - `calibration`, `calibration_directory`: the calibration an ACTIVATED refit built and
      activated, and its artifact directory (step 8); None otherwise, and None when
      `run_refit` was given no builder.
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
    calibration: Calibration | None = None
    calibration_directory: Path | None = None


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
    calibration: CalibrationBuilder | None = None,
) -> RefitOutcome:
    """Refit `current` on `data` and record the outcome (module docstring, steps 4–8). `data`
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
    naive `at`. ModelError from training, matching or drift is a REJECTED outcome.

    Training goes through this module's `train_candidate` name (`train_candidate(data,
    Candidate(K, covariance type of current), thresholds=..., config=..., provenance=...,
    workers=..., previous=current)`), which the tests replace to plant a sanity reject.

    `calibration` (step 8): None builds no calibration (an ACTIVATED refit moves the model
    pointer alone, as B7a did); a builder is called as `calibration(data.fit_end)` only for an
    ACTIVATED outcome, before anything is recorded. ModelError, recording nothing, when its
    calibration's `as_of` isn't `data.fit_end`; anything the builder raises propagates and
    records nothing (the model artifact may already be saved: it is content-addressed and
    unreferenced)."""
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
        if reasons:
            kind = RefitOutcomeKind.REJECTED
        else:
            match = match_states(current, result.model)
            model = inherit_labels(result.model, match, previous=previous)
            report = refit_drift(current, model, match, drift)
            reasons = report.reasons
            kind = RefitOutcomeKind.ACTIVATED if report.passed else RefitOutcomeKind.FROZEN
    except ModelError as exc:
        reasons = (str(exc),)
        kind = RefitOutcomeKind.REJECTED
        model = None
        match = None
        report = None
    if model is not None:
        directory = save_model(model, artifacts / "models")
    if kind is RefitOutcomeKind.ACTIVATED and calibration is not None:
        raise NotImplementedError("B7b: run_refit builds and activates the pass's calibration")
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
    calibration: CalibrationConfig | None = None,
) -> RefitOutcome | None:
    """The evening job: None, reading and writing nothing, when `refit_due(current, at)` is
    false; otherwise `run_refit(refit_data(store, symbol, start=current.fit_start,
    cutoff=refit_cutoff(at), features=features, policy=policy), current, at=at, ...,
    calibration=<builder>)` with the other arguments passed through, the builder being
    `lambda as_of: quant.jobs.calibrate.calibrate(store, ops, as_of=as_of, symbol=symbol,
    features=features, artifacts=artifacts, config=calibration or r1_calibration_config(),
    policy=policy)` (module docstring, step 8): every refit job builds its pass's calibration.
    The calibration loads the models that governed from `artifacts / "models"`, so the active
    model must be saved there (the first model's job and every refit save theirs)."""
    events = ops.refit_events()
    if not refit_due(current, at, last_refit=events[-1].cutoff if events else None):
        return None
    data = refit_data(
        store,
        symbol,
        start=current.fit_start,
        cutoff=refit_cutoff(at),
        features=features,
        policy=policy,
    )
    rules = calibration or r1_calibration_config()

    def build(as_of: datetime) -> Calibration:
        return calibrate(
            store,
            ops,
            as_of=as_of,
            symbol=symbol,
            features=features,
            artifacts=artifacts,
            config=rules,
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
        calibration=build,
    )
