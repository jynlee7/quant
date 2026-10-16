"""B7b: the calibration job and the calibration a refit builds (spec §9, A4; ARCHITECTURE §7.3
step 5, §7.4; C13; T-NL2).

Pinned here: each model's forecasts are its own filter's (warmed from its fit window) and its
targets the smoothed labels over its rows up to the end of its span, never later ones; the F1
interval is the mixture under q_t = p_t A in raw units; the spans are exactly what ops.db's
model pointer governed before the as-of time; the job reads through the view at the as-of time,
so stores that agree on what was known by then give the same series; a passing refit builds a
calibration as of its cutoff and moves its pointer with the model's, in one transaction,
effective at the next open (so the as-of time precedes every use); a frozen or rejected refit
builds none.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm
from structlog.testing import capture_logs

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.calibration import CalibrationSource, r1_calibration_config
from quant.core.clock import SimClock
from quant.core.epoch import to_us
from quant.core.errors import ClockError, DataError, ModelError
from quant.core.regime import Label
from quant.eval.smoother import smooth
from quant.features.menu import r1_features
from quant.features.stream import FeatureMatrix
from quant.hmm.filter import forward_pass
from quant.hmm.labels import label_probs, r1_thresholds, top_label
from quant.hmm.model import HmmModel, ScalerParams, save_model
from tests.support.hmm import FEATURE_HASH, label_table, params_of, sample, three_full
from tests.support.hmm import training_record as hmm_training_record
from tests.support.truncation import (
    ACTIVATION,
    garbage_store,
    lifecycle_spec,
    truncated_store,
    world,
)

XNYS = TradingCalendar.xnys()
US = timedelta(microseconds=1)
LABELS = tuple(Label)
CONFIG = r1_calibration_config()
PROVENANCE = {"commit": "0" * 40, "lockfile_sha256": "1" * 64}
# The synthetic world's refit evening: due 30 days after the fixture's data (2016-10-31).
REFIT_AT = datetime(2016, 11, 30, 17, 30, tzinfo=NEW_YORK).astimezone(UTC)
CUTOFF = datetime(2016, 11, 30, 17, 0, tzinfo=NEW_YORK).astimezone(UTC)
NEXT_OPEN = datetime(2016, 12, 1, 9, 30, tzinfo=NEW_YORK).astimezone(UTC)


def ny(*args: int) -> datetime:
    return datetime(*args, tzinfo=NEW_YORK).astimezone(UTC)


# --- a known HMM in raw units ---------------------------------------------------------------

SCALER = ScalerParams(("ret", "x"), np.array([0.1, -0.2]), np.array([2.0, 0.5]))
ROW_START = datetime(2016, 3, 1, 15, tzinfo=UTC)
N_ROWS, N_FIT = 600, 400
# A span of rows [400, 464): with seed 5's rows, smoothing over the later rows changes two of its
# targets (rows 400–599 change targets at 19 span ends; this one changes two).
SHORT_SPAN = 464


def _known() -> tuple[HmmModel, FeatureMatrix]:
    """three_full in standardized units under SCALER (raw = z · std + mean), labeled CHOP,
    CALM_UP, STRESS; 600 hourly rows from ROW_START, the first 400 its fit window."""
    truth = three_full()
    _, z = sample(truth, N_ROWS, seed=5)
    t_open = np.array(
        [to_us(ROW_START + timedelta(hours=i)) for i in range(N_ROWS)], dtype=np.int64
    )
    available = t_open + 3_600_000_000
    matrix = FeatureMatrix(SCALER.columns, t_open, available, z * SCALER.std + SCALER.mean)
    model = HmmModel(
        params=params_of(truth),
        scaler=SCALER,
        fit_start=ROW_START,
        fit_end=ROW_START + timedelta(hours=N_FIT + 1),  # after row 399 became available
        feature_config_hash=FEATURE_HASH,
        labels=label_table([Label.CHOP, Label.CALM_UP, Label.STRESS]),
        training=hmm_training_record(),
    )
    return model, matrix


def _span_end(rows: int) -> datetime:
    """The end of a span holding rows [N_FIT, rows): just after row rows - 1 became available."""
    return ROW_START + timedelta(hours=rows, minutes=30)


def _upto(matrix: FeatureMatrix, end: datetime) -> FeatureMatrix:
    return matrix.window(ROW_START, end)


@pytest.mark.spec("SPEC§9", "A4", "I2")
def test_a_models_forecasts_are_its_filters_and_its_targets_the_smoothed_labels() -> None:
    from quant.eval.calibration import f1_interval, model_series

    model, matrix = _known()
    start, end = model.fit_end, _span_end(500)
    rows = _upto(matrix, end)
    series = model_series(model, rows, start=start, end=end, level=0.9)
    assert series.models == (model.artifact_hash(),)
    assert len(series) == 100  # rows 400..499
    assert np.array_equal(series.available_at, rows.available_at[N_FIT:])

    trace = forward_pass(model.params, model.scaler, rows.values, columns=rows.columns)
    smoothed = smooth(model.params, model.scaler, rows.values, columns=rows.columns)
    for i, t in enumerate(range(N_FIT, 500)):
        filtered = label_probs(trace.state_probs[t], model.labels)
        expected = [min(1.0, max(0.0, filtered.get(label, 0.0))) for label in LABELS]
        assert series.forecast[i].tolist() == expected
        smoothed_labels = label_probs(smoothed[t], model.labels)
        assert series.target[i] == LABELS.index(top_label(smoothed_labels))
        assert series.smoothed[i].tolist() == [
            min(1.0, max(0.0, smoothed_labels.get(label, 0.0))) for label in LABELS
        ]
        q = trace.state_probs[t] @ model.params.transmat
        assert (series.low[i], series.high[i]) == f1_interval(model, q, level=0.9)
        assert series.realized[i] == rows.values[t, 0]


@pytest.mark.spec("SPEC§9", "A4", "I1")
def test_the_smoother_sees_nothing_past_the_span() -> None:
    from quant.eval.calibration import model_series

    model, matrix = _known()
    start, cut = model.fit_end, SHORT_SPAN
    short = model_series(
        model, _upto(matrix, _span_end(cut)), start=start, end=_span_end(cut), level=0.9
    )
    long = model_series(
        model, _upto(matrix, _span_end(N_ROWS)), start=start, end=_span_end(N_ROWS), level=0.9
    )
    shared = cut - N_FIT
    # The forecasts of the shared bars agree (the filter only looks back)...
    assert np.array_equal(short.forecast, long.forecast[:shared])
    # ...but the targets were smoothed without the later rows: two of them differ here.
    smoothed_short = smooth(model.params, model.scaler, matrix.values[:cut], columns=matrix.columns)
    expected = [
        LABELS.index(top_label(label_probs(smoothed_short[t], model.labels)))
        for t in range(N_FIT, cut)
    ]
    assert short.target.tolist() == expected
    assert int((short.target != long.target[:shared]).sum()) == 2
    # A row at or after the span's end is refused, not ignored.
    with pytest.raises(ModelError):
        model_series(
            model, _upto(matrix, _span_end(cut + 6)), start=start, end=_span_end(cut), level=0.9
        )


@pytest.mark.spec("SPEC§9", "A4", "T-NL2")
@pytest.mark.parametrize(
    "fault", ["in-sample span", "empty span", "columns", "before fit_start", "naive"]
)
def test_model_series_refuses_what_isnt_an_out_of_sample_span(fault: str) -> None:
    from quant.eval.calibration import model_series

    model, matrix = _known()
    start, end = model.fit_end, _span_end(500)
    rows = _upto(matrix, end)
    if fault == "in-sample span":
        start = model.fit_end - timedelta(hours=5)
    elif fault == "empty span":
        end = start
    elif fault == "columns":
        rows = FeatureMatrix(("ret", "y"), rows.t_open, rows.available_at, rows.values)
    elif fault == "before fit_start":
        model = HmmModel(
            params=model.params,
            scaler=model.scaler,
            fit_start=ROW_START + timedelta(hours=10),
            fit_end=model.fit_end,
            feature_config_hash=model.feature_config_hash,
            labels=model.labels,
            training=model.training,
        )
    else:
        with pytest.raises(ClockError):
            model_series(model, rows, start=start.replace(tzinfo=None), end=end, level=0.9)
        return
    with pytest.raises(ModelError):
        model_series(model, rows, start=start, end=end, level=0.9)


def _ref_interval(
    means: np.ndarray, sds: np.ndarray, q: np.ndarray, level: float
) -> tuple[float, float]:
    def cdf(x: float) -> float:
        return float(sum(qk * norm.cdf(x, m, s) for qk, m, s in zip(q, means, sds, strict=True)))

    lo, hi = float(means.min() - 20 * sds.max()), float(means.max() + 20 * sds.max())
    return (
        brentq(lambda x: cdf(x) - (1 - level) / 2, lo, hi, xtol=1e-14),
        brentq(lambda x: cdf(x) - (1 + level) / 2, lo, hi, xtol=1e-14),
    )


@pytest.mark.spec("SPEC§9", "A4", "C13")
@pytest.mark.parametrize("q", [[0.2, 0.5, 0.3], [0.0, 1.0, 0.0], [0.98, 0.01, 0.01]])
def test_the_f1_interval_is_the_central_mixture_interval_in_raw_units(q: list[float]) -> None:
    from quant.eval.calibration import f1_interval

    model, _ = _known()
    probs = np.array(q)
    means = model.params.means[:, 0] * SCALER.std[0] + SCALER.mean[0]
    sds = np.sqrt(model.params.covars[:, 0, 0]) * SCALER.std[0]
    low, high = f1_interval(model, probs, level=0.9)
    ref_low, ref_high = _ref_interval(means, sds, probs, 0.9)
    assert math.isclose(low, ref_low, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(high, ref_high, rel_tol=0, abs_tol=1e-9)
    if max(q) == 1.0:
        k = q.index(1.0)
        z = norm.ppf(0.95)
        assert math.isclose(low, means[k] - z * sds[k], abs_tol=1e-9)
        assert math.isclose(high, means[k] + z * sds[k], abs_tol=1e-9)


@pytest.mark.spec("SPEC§9", "C13")
@pytest.mark.parametrize("fault", ["no ret column", "q sum", "q shape", "level"])
def test_the_f1_interval_refuses_what_it_cant_compute(fault: str) -> None:
    from quant.eval.calibration import f1_interval

    model, _ = _known()
    q, level = np.array([0.2, 0.5, 0.3]), 0.9
    if fault == "no ret column":
        model = HmmModel(
            params=model.params,
            scaler=ScalerParams(("a", "b"), SCALER.mean, SCALER.std),
            fit_start=model.fit_start,
            fit_end=model.fit_end,
            feature_config_hash=model.feature_config_hash,
            labels=model.labels,
            training=model.training,
        )
    elif fault == "q sum":
        q = np.array([0.2, 0.5, 0.4])
    elif fault == "q shape":
        q = np.array([0.5, 0.5])
    else:
        level = 1.0
    with pytest.raises(ModelError):
        f1_interval(model, q, level=level)


# --- the spans ------------------------------------------------------------------------------


def _ops(at: datetime) -> tuple[Any, SimClock]:
    from quant.store.ops import OpsStore

    clock = SimClock(at)
    return OpsStore.memory(clock=clock), clock


@pytest.mark.spec("SPEC§9", "ARCH§7.4", "ARCH§9.2", "C7")
def test_the_spans_are_what_the_model_pointer_governed_before_the_as_of_time() -> None:
    from quant.jobs.calibrate import ModelSpan, model_spans

    a, b, c = "a1" * 32, "b2" * 32, "c3" * 32
    ops, clock = _ops(ny(2017, 3, 1, 18))
    ops.activate("model", a, reason="first model")  # governs from 03-02 09:30
    clock.advance_to(ny(2017, 3, 6, 18))
    ops.activate("model", b, reason="refit")  # governs from 03-07 09:30
    clock.advance_to(ny(2017, 3, 6, 19))
    ops.activate("model", c, reason="refit again")  # supersedes b at the same open

    assert model_spans(ops, as_of=ny(2017, 3, 2, 9, 30)) == ()
    assert model_spans(ops, as_of=ny(2017, 3, 3, 12)) == (
        ModelSpan(a, ny(2017, 3, 2, 9, 30), ny(2017, 3, 3, 12)),
    )
    assert model_spans(ops, as_of=ny(2017, 3, 8, 17)) == (
        ModelSpan(a, ny(2017, 3, 2, 9, 30), ny(2017, 3, 7, 9, 30)),
        ModelSpan(c, ny(2017, 3, 7, 9, 30), ny(2017, 3, 8, 17)),
    )
    for t in (ny(2017, 3, 2, 9, 30), ny(2017, 3, 7, 9, 29), ny(2017, 3, 7, 9, 30)):
        governing = ops.current("model", t).artifact
        (span,) = [s for s in model_spans(ops, as_of=ny(2017, 3, 8, 17)) if s.start <= t < s.end]
        assert span.artifact == governing, t
    with pytest.raises(ClockError):
        model_spans(ops, as_of=ny(2017, 3, 8, 17).replace(tzinfo=None))


# --- the job on the synthetic world ---------------------------------------------------------


def _world_ops(
    artifacts: Path, model: HmmModel, *, at: datetime = ACTIVATION
) -> tuple[Any, SimClock]:
    """An ops store whose model pointer names `model` from the first open after `at`, with the
    model saved under `artifacts`."""
    save_model(model, artifacts / "models")
    ops, clock = _ops(at)
    ops.activate("model", model.artifact_hash(), reason="first model")
    return ops, clock


@pytest.mark.spec("SPEC§9", "ARCH§7.4", "I1", "T-NL1")
def test_the_job_reads_through_the_view_at_the_as_of_time_and_nothing_later(tmp_path: Path) -> None:
    from quant.jobs.calibrate import calibration_series

    w = world()
    model = lifecycle_spec(w).start
    ops, _ = _world_ops(tmp_path, model)

    def read(store: Any) -> Any:
        return calibration_series(
            store,
            ops,
            as_of=CUTOFF,
            symbol="SPY",
            features=r1_features(),
            artifacts=tmp_path,
            level=CONFIG.outcome_level,
        )

    series = read(w.store)
    assert series.models == (model.artifact_hash(),)
    first_open = ny(2016, 11, 1, 9, 30)
    assert int(series.available_at[0]) >= to_us(first_open)
    assert int(series.available_at[-1]) < to_us(CUTOFF)
    assert 120 <= len(series) <= 150  # November 2016's bars
    assert len(set(series.target.tolist())) == 2  # the start model's two labels both occur
    assert read(truncated_store(w, CUTOFF)).data_hash() == series.data_hash()
    garbage, _ = garbage_store(w, CUTOFF, seed=11)
    assert read(garbage).data_hash() == series.data_hash()


@pytest.mark.spec("SPEC§9", "ARCH§7.4", "C13")
def test_with_no_governed_bars_the_series_is_empty(tmp_path: Path) -> None:
    from quant.jobs.calibrate import calibrate

    w = world()
    ops, _ = _world_ops(tmp_path, w.model)
    calibration = calibrate(
        w.store,
        ops,
        as_of=ACTIVATION,  # before the first model's first open
        symbol="SPY",
        features=r1_features(),
        artifacts=tmp_path,
        config=CONFIG,
    )
    assert calibration.document.bars == 0
    assert calibration.document.models == ()
    assert all(calibration.cap_fraction(label) == 0.25 for label in Label)


@pytest.mark.spec("SPEC§9", "ARCH§7.4", "T-NL2")
@pytest.mark.parametrize("fault", ["missing artifact", "feature config", "in-sample span"])
def test_the_job_refuses_a_history_it_cant_score(fault: str, tmp_path: Path) -> None:
    from quant.features.menu import FeatureConfig, Ret, Rvol
    from quant.jobs.calibrate import calibration_series

    w = world()
    model = w.model
    features = r1_features()
    if fault == "missing artifact":
        ops, _ = _ops(ACTIVATION)
        ops.activate("model", model.artifact_hash(), reason="never saved")
    elif fault == "feature config":
        ops, _ = _world_ops(tmp_path, model)
        features = FeatureConfig(ret=Ret(), rvol=Rvol())
    else:
        # A pointer governing from 2016-10-31's open, before the model's data ended.
        ops, _ = _world_ops(tmp_path, model, at=ny(2016, 10, 30, 18))
    with pytest.raises(ModelError):
        calibration_series(
            w.store,
            ops,
            as_of=CUTOFF,
            symbol="SPY",
            features=features,
            artifacts=tmp_path,
            level=0.9,
        )


@pytest.mark.spec("SPEC§9", "ARCH§7.4")
def test_calibrate_scores_the_series_and_writes_nothing(tmp_path: Path) -> None:
    from quant.eval.calibration import build_calibration
    from quant.jobs.calibrate import calibrate, calibration_series

    w = world()
    model = lifecycle_spec(w).start
    ops, _ = _world_ops(tmp_path, model)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    pointers = ops.pointers("calibration")
    config = lifecycle_spec(w).calibration
    calibration = calibrate(
        w.store,
        ops,
        as_of=CUTOFF,
        symbol="SPY",
        features=r1_features(),
        artifacts=tmp_path,
        config=config,
        source=CalibrationSource.BACKTEST,
    )
    series = calibration_series(
        w.store,
        ops,
        as_of=CUTOFF,
        symbol="SPY",
        features=r1_features(),
        artifacts=tmp_path,
        level=config.outcome_level,
    )
    expected = build_calibration(
        series, as_of=CUTOFF, source=CalibrationSource.BACKTEST, config=config
    )
    assert calibration.artifact_hash() == expected.artifact_hash()
    assert calibration.document.sufficient  # the harness's rules: 60 scored bars suffice
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert ops.pointers("calibration") == pointers


# --- the calibration a refit builds ---------------------------------------------------------


def _refit(w: Any, current: HmmModel, ops: Any, artifacts: Path, **kwargs: Any) -> Any:
    from quant.hmm.drift import r1_drift_thresholds
    from quant.jobs.refit import refit

    spec = lifecycle_spec(w)
    with capture_logs():
        return refit(
            w.store,
            current,
            at=REFIT_AT,
            symbol="SPY",
            features=r1_features(),
            ops=ops,
            artifacts=artifacts,
            config=spec.train,
            thresholds=r1_thresholds(),
            drift=r1_drift_thresholds(),
            provenance=PROVENANCE,
            workers=1,
            calibration=kwargs.pop("calibration", spec.calibration),
            **kwargs,
        )


@pytest.mark.spec("ARCH§7.3", "ARCH§7.4", "T-NL2", "C7", "SPEC§9")
def test_a_passing_refit_builds_a_calibration_as_of_its_cutoff_and_activates_it_with_the_model(
    tmp_path: Path,
) -> None:
    from quant.core.calibration import load_calibration
    from quant.store.ops import RefitOutcomeKind

    w = world()
    start = lifecycle_spec(w).start
    ops, clock = _world_ops(tmp_path, start)
    clock.advance_to(REFIT_AT)
    outcome = _refit(w, start, ops, tmp_path)
    assert outcome.kind is RefitOutcomeKind.ACTIVATED, outcome.reasons

    calibration = outcome.calibration
    assert calibration is not None
    assert calibration.as_of == outcome.cutoff == CUTOFF
    assert calibration.document.models == (start.artifact_hash(),)  # not the new model
    assert calibration.document.source is CalibrationSource.BACKTEST
    digest = calibration.artifact_hash()
    assert outcome.calibration_directory == tmp_path / "calibrations" / digest
    assert load_calibration(outcome.calibration_directory).artifact_hash() == digest

    # Both pointers move together, at the next open, in one refit.
    assert outcome.event.effective_at == NEXT_OPEN
    (pointer,) = ops.pointers("calibration")
    (_, model_pointer) = ops.pointers("model")
    assert (pointer.artifact, pointer.effective_at) == (digest, NEXT_OPEN)
    assert (pointer.recorded_at, pointer.reason) == (
        model_pointer.recorded_at,
        model_pointer.reason,
    )
    assert ops.current("calibration", NEXT_OPEN - US) is None
    assert ops.current("calibration", NEXT_OPEN).artifact == digest
    # T-NL2: its as-of time precedes every decision that can use it.
    assert calibration.as_of < pointer.effective_at
    calibration.require_before(pointer.effective_at)


@pytest.mark.spec("ARCH§7.3", "ARCH§7.4", "C5")
@pytest.mark.parametrize("how", ["frozen", "rejected"])
def test_a_frozen_or_rejected_refit_builds_no_calibration(
    how: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant.jobs.calibrate as calibrate_job
    import quant.jobs.refit as refit_job
    from quant.store.ops import RefitOutcomeKind

    w = world()
    current = w.model  # its refit drifts: derived CHOP, inherited CALM_UP
    if how == "rejected":

        def reject(*args: Any, **kwargs: Any) -> Any:
            raise ModelError("no filtered mass")

        monkeypatch.setattr(refit_job, "train_candidate", reject)
    built = []
    real = calibrate_job.calibrate
    monkeypatch.setattr(
        refit_job, "calibrate", lambda *a, **k: built.append(k["as_of"]) or real(*a, **k)
    )
    ops, clock = _world_ops(tmp_path, current)
    clock.advance_to(REFIT_AT)
    outcome = _refit(w, current, ops, tmp_path)
    assert outcome.kind is (
        RefitOutcomeKind.FROZEN if how == "frozen" else RefitOutcomeKind.REJECTED
    )
    assert outcome.calibration is None
    assert outcome.calibration_directory is None
    assert built == []
    assert ops.pointers("calibration") == ()
    assert not (tmp_path / "calibrations").exists()


def _passing_data(w: Any) -> Any:
    from quant.jobs.refit import refit_data

    return refit_data(
        w.store,
        "SPY",
        start=lifecycle_spec(w).start.fit_start,
        cutoff=CUTOFF,
        features=r1_features(),
    )


def _run(w: Any, ops: Any, artifacts: Path, builder: Any) -> Any:
    from quant.hmm.drift import r1_drift_thresholds
    from quant.jobs.refit import run_refit

    spec = lifecycle_spec(w)
    with capture_logs():
        return run_refit(
            _passing_data(w),
            spec.start,
            at=REFIT_AT,
            ops=ops,
            artifacts=artifacts,
            config=spec.train,
            thresholds=r1_thresholds(),
            drift=r1_drift_thresholds(),
            provenance=PROVENANCE,
            workers=1,
            calibration=builder,
        )


@pytest.mark.spec("ARCH§7.3", "ARCH§7.4", "T-NL2")
@pytest.mark.parametrize("offset", [timedelta(hours=-1), US], ids=["earlier", "later"])
def test_a_calibration_not_as_of_the_cutoff_is_refused_and_nothing_recorded(
    offset: timedelta, tmp_path: Path
) -> None:
    from quant.eval.calibration import ForecastSeries, build_calibration

    w = world()
    ops, clock = _world_ops(tmp_path, lifecycle_spec(w).start)
    clock.advance_to(REFIT_AT)
    asked = []

    def builder(as_of: datetime) -> Any:
        asked.append(as_of)
        return build_calibration(
            ForecastSeries.empty(),
            as_of=as_of + offset,
            source=CalibrationSource.BACKTEST,
            config=CONFIG,
        )

    with pytest.raises(ModelError):
        _run(w, ops, tmp_path, builder)
    assert asked == [CUTOFF]
    assert ops.refit_events() == ()
    assert len(ops.pointers("model")) == 1
    assert ops.pointers("calibration") == ()


@pytest.mark.spec("ARCH§7.3", "ARCH§7.4")
def test_a_calibration_that_cant_be_built_leaves_no_refit_recorded(tmp_path: Path) -> None:
    w = world()
    ops, clock = _world_ops(tmp_path, lifecycle_spec(w).start)
    clock.advance_to(REFIT_AT)

    def builder(as_of: datetime) -> Any:
        raise DataError("the store lacks a session")

    with pytest.raises(DataError):
        _run(w, ops, tmp_path, builder)
    assert ops.refit_events() == ()
    assert len(ops.pointers("model")) == 1
    assert ops.pointers("calibration") == ()


# --- ops.db ---------------------------------------------------------------------------------


@pytest.mark.spec("ARCH§9.2", "ARCH§7.3", "C7")
def test_record_refit_appends_the_calibration_pointer_in_the_same_transaction() -> None:
    from quant.store.ops import RefitOutcomeKind

    ops, _ = _ops(ny(2017, 6, 30, 17, 30))
    previous, new, calibration = "a1" * 32, "b2" * 32, "c3" * 32
    event = ops.record_refit(
        at=ny(2017, 6, 30, 17, 30),
        cutoff=ny(2017, 6, 30, 17),
        outcome=RefitOutcomeKind.ACTIVATED,
        artifact=new,
        previous=previous,
        reasons=(),
        calibration=calibration,
    )
    (model,) = ops.pointers("model")
    (pointer,) = ops.pointers("calibration")
    assert pointer.seq == model.seq + 1
    assert pointer.artifact == calibration
    assert pointer.effective_at == model.effective_at == event.effective_at == ny(2017, 7, 3, 9, 30)
    assert (pointer.recorded_at, pointer.reason) == (model.recorded_at, model.reason)


@pytest.mark.spec("ARCH§9.2", "ARCH§7.3")
@pytest.mark.parametrize("outcome", ["frozen", "rejected", "bad hash"])
def test_record_refit_refuses_a_calibration_it_cant_activate(outcome: str) -> None:
    from quant.store.ops import RefitOutcomeKind

    ops, _ = _ops(ny(2017, 6, 30, 17, 30))
    kind = {
        "frozen": RefitOutcomeKind.FROZEN,
        "rejected": RefitOutcomeKind.REJECTED,
        "bad hash": RefitOutcomeKind.ACTIVATED,
    }[outcome]
    with pytest.raises(DataError):
        ops.record_refit(
            at=ny(2017, 6, 30, 17, 30),
            cutoff=ny(2017, 6, 30, 17),
            outcome=kind,
            artifact=None if kind is RefitOutcomeKind.REJECTED else "b2" * 32,
            previous="a1" * 32,
            reasons=() if kind is RefitOutcomeKind.ACTIVATED else ("drift",),
            calibration="not a hash" if outcome == "bad hash" else "c3" * 32,
        )
    assert ops.refit_events() == ()
    assert ops.pointers("calibration") == ()
    assert ops.pointers("model") == ()
    assert ops.freeze_events() == ()
