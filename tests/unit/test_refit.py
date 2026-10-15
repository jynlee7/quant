"""B7a: the monthly refit (spec §5, ARCHITECTURE §7.1, §7.3; C5, C7).

The fast tier fits K=2 diag with two restarts on synthetic bars from a known HMM (laid on a
1990s calendar, so nobody mistakes them for market data), and the store-backed tests use the
truncation harness's synthetic FIT world. Real-data refits through 2018 are B7b's full tier.

Pinned here: one restart of a refit really starts from the previous model's parameters,
converted between the two scalers, and meta.json names it; the cadence and the 17:00 cutoff;
the rows come through the view at the cutoff and nothing later reaches them; a pass activates
at the next open and never sooner; a failed drift check stores the model, doesn't activate it
and freezes entries; a sanity reject keeps the current model and alerts.
"""

from __future__ import annotations

import functools
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from structlog.testing import capture_logs

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.clock import SimClock
from quant.core.errors import ClockError, ModelError
from quant.core.regime import Label
from quant.features.menu import r1_features
from quant.features.scaler import Scaler
from quant.features.stream import FeatureMatrix
from quant.hmm.labels import r1_thresholds
from quant.hmm.model import GaussianParams, HmmModel, ScalerParams, load_model
from quant.hmm.select import Candidate, SanityReport
from quant.hmm.train import TrainConfig, TrainingData, TrainResult, r1_train_config
from tests.support.hmm import Truth, model_of, three_full
from tests.support.train import training_data
from tests.support.truncation import garbage_store, truncated_store, world

RED = pytest.mark.xfail(strict=True, reason="red:B7", raises=NotImplementedError)
XNYS = TradingCalendar.xnys()
US = timedelta(microseconds=1)
PROVENANCE = {"commit": "0" * 40, "lockfile_sha256": "1" * 64}
CANDIDATE = Candidate(n_states=2, covariance_type="diag")
SESSION = 7  # bars per synthetic session (tests.support.train)


def ny(*args: int) -> datetime:
    """A New York wall time, in UTC."""
    return datetime(*args, tzinfo=NEW_YORK).astimezone(UTC)


def _f(values: Any) -> Any:
    return np.array(values, dtype=np.float64)


def apart() -> Truth:
    """Two well-separated states over two features, diag: a refit on more of its bars finds
    the same states, so it passes every drift check."""
    return Truth(
        "diag",
        _f([0.5, 0.5]),
        _f([[0.95, 0.05], [0.08, 0.92]]),
        _f([[-1.0, 0.0], [1.5, 1.0]]),
        np.stack([np.diag(_f([0.5, 0.8])), np.diag(_f([1.2, 0.6]))]),
    )


def tiny_config() -> TrainConfig:
    """r1's config with two restarts, 50 bootstrap draws and a one-month validation block."""
    return r1_train_config().model_copy(
        update={
            "n_restarts": 2,
            "candidates": (CANDIDATE,),
            "bootstrap_draws": 50,
            "validation_months": 1,
        }
    )


@functools.cache
def _bars() -> TrainingData:
    return training_data(apart(), 300 * SESSION, 11)


def cut(
    data: TrainingData, sessions: int, *, skip: int = 0, feature_hash: str | None = None
) -> TrainingData:
    """The rows of sessions [skip, sessions) of `data`, as a window from the first kept
    session's midnight to the midnight after the last (UTC, as tests.support.train lays it)."""
    rows = slice(skip * SESSION, sessions * SESSION)
    matrix = data.matrix
    first = date.fromordinal(int(data.sessions[rows][0]))
    last = date.fromordinal(int(data.sessions[rows][-1])) + timedelta(days=1)
    return TrainingData(
        FeatureMatrix(
            matrix.columns,
            matrix.t_open[rows].copy(),
            matrix.available_at[rows].copy(),
            matrix.values[rows].copy(),
        ),
        data.returns[rows].copy(),
        data.sessions[rows].copy(),
        datetime(first.year, first.month, first.day, tzinfo=UTC),
        datetime(last.year, last.month, last.day, tzinfo=UTC),
        feature_hash or data.feature_config_hash,
    )


def _train(data: TrainingData, **kwargs: Any) -> TrainResult:
    from quant.hmm.train import train_candidate

    return train_candidate(
        data,
        CANDIDATE,
        thresholds=r1_thresholds(),
        config=tiny_config(),
        provenance=PROVENANCE,
        workers=1,
        **kwargs,
    )


@functools.cache
def first_model() -> HmmModel:
    """The 'current' model: fit on the first 240 sessions."""
    return _train(cut(_bars(), 240)).model


def drifted(model: HmmModel) -> HmmModel:
    """`model` with a transition matrix far from the generator's, so a refit drifts."""
    p = model.params
    return HmmModel(
        params=GaussianParams(
            p.covariance_type, p.startprob, _f([[0.6, 0.4], [0.4, 0.6]]), p.means, p.covars
        ),
        scaler=model.scaler,
        fit_start=model.fit_start,
        fit_end=model.fit_end,
        feature_config_hash=model.feature_config_hash,
        labels=model.labels,
        training=model.training,
    )


# --- The seeded restart (quant.hmm.train) ------------------------------------------------------


@RED
@pytest.mark.spec("ARCH§7.1", "SPEC§5")
def test_seed_params_carry_the_previous_model_through_raw_units() -> None:
    from quant.hmm.match import raw_gaussians
    from quant.hmm.train import seed_params

    previous = first_model()
    other = ScalerParams(previous.scaler.columns, _f([0.3, -0.2]), _f([1.7, 0.6]))
    seeded = seed_params(previous, other)
    assert seeded.covariance_type == "diag"
    np.testing.assert_array_equal(seeded.startprob, previous.params.startprob)
    np.testing.assert_array_equal(seeded.transmat, previous.params.transmat)
    old_means, old_covars = raw_gaussians(previous.params, previous.scaler)
    new_means, new_covars = raw_gaussians(seeded, other)
    np.testing.assert_allclose(new_means, old_means, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new_covars, old_covars, rtol=1e-12, atol=1e-15)
    for covariance in seeded.covars:  # a diag model stays diagonal
        assert np.array_equal(covariance, np.diag(np.diag(covariance)))
    with pytest.raises(ModelError):
        seed_params(previous, ScalerParams(("a", "b"), other.mean, other.std))


@RED
@pytest.mark.spec("ARCH§7.1", "SPEC§5", "I2")
def test_one_restart_of_a_refit_starts_from_the_previous_models_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant.hmm.train as train
    from quant.hmm.train import SEEDED_RESTART, seed_params

    fits: list[dict[str, Any]] = []

    class Recording(train.GaussianHMM):  # type: ignore[misc, name-defined]
        def fit(self, X: Any, lengths: Any = None) -> Any:  # noqa: N803 (hmmlearn's name)
            def grab(name: str) -> Any:
                value = getattr(self, name, None)
                return None if value is None else np.array(value, copy=True)

            fits.append(
                {
                    "init_params": self.init_params,
                    "params": self.params,
                    "random_state": self.random_state,
                    "rows": len(X),
                    **{n: grab(n) for n in ("startprob_", "transmat_", "means_", "_covars_")},
                }
            )
            return super().fit(X, lengths)

    monkeypatch.setattr(train, "GaussianHMM", Recording)
    previous = first_model()
    data = cut(_bars(), 270)
    result = _train(data, previous=previous)

    config = tiny_config()
    assert len(fits) == 2 * config.n_restarts  # the full-window fit, then the validation fit
    full, validation = fits[: config.n_restarts], fits[config.n_restarts :]
    seeded = [i for i, fit in enumerate(full) if fit["init_params"] == ""]
    assert seeded == [SEEDED_RESTART] == [0]
    assert all(fit["init_params"] == "stmc" for fit in validation)  # the validation fit isn't
    assert all(fit["params"] == "stmc" for fit in fits)
    assert full[0]["random_state"] == config.base_seed + SEEDED_RESTART
    assert full[0]["rows"] == len(data.matrix)

    scaler = ScalerParams.from_scaler(
        Scaler.fit(data.matrix, start=data.fit_start, end=data.fit_end)
    )
    expected = seed_params(previous, scaler)
    start = full[0]
    np.testing.assert_array_equal(start["startprob_"], expected.startprob)
    np.testing.assert_array_equal(start["transmat_"], expected.transmat)
    np.testing.assert_array_equal(start["means_"], expected.means)
    covars = start["_covars_"]  # hmmlearn keeps a diag model's (K, d) diagonals
    assert covars.shape == (2, 2)
    np.testing.assert_array_equal(covars, np.diagonal(expected.covars, axis1=1, axis2=2))

    record = result.model.training
    assert dict(record.info["restarts"]["seeded"]) == {
        "index": SEEDED_RESTART,
        "previous": previous.artifact_hash(),
    }
    assert record.seeds[0] == config.base_seed + SEEDED_RESTART
    # A fit without a previous model records none.
    assert _train(data).model.training.info["restarts"]["seeded"] is None


@RED
@pytest.mark.spec("ARCH§7.1", "SPEC§5")
@pytest.mark.parametrize("fault", ["states", "covariance type", "columns", "feature hash"])
def test_a_seed_that_doesnt_fit_the_refit_is_refused(fault: str) -> None:
    previous = first_model()
    if fault == "states":
        bad = model_of(three_full())  # K = 3
    elif fault == "covariance type":
        p = previous.params
        bad = HmmModel(
            params=GaussianParams("full", p.startprob, p.transmat, p.means, p.covars),
            scaler=previous.scaler,
            fit_start=previous.fit_start,
            fit_end=previous.fit_end,
            feature_config_hash=previous.feature_config_hash,
            labels=previous.labels,
            training=previous.training,
        )
    elif fault == "columns":
        bad = HmmModel(
            params=previous.params,
            scaler=ScalerParams(("a", "b"), previous.scaler.mean, previous.scaler.std),
            fit_start=previous.fit_start,
            fit_end=previous.fit_end,
            feature_config_hash=previous.feature_config_hash,
            labels=previous.labels,
            training=previous.training,
        )
    else:
        bad = HmmModel(
            params=previous.params,
            scaler=previous.scaler,
            fit_start=previous.fit_start,
            fit_end=previous.fit_end,
            feature_config_hash="cd" * 32,
            labels=previous.labels,
            training=previous.training,
        )
    with pytest.raises(ModelError):
        _train(cut(_bars(), 270), previous=bad)


# --- Cadence and cutoff -----------------------------------------------------------------------


@RED
@pytest.mark.spec("SPEC§5", "ARCH§7.3")
@pytest.mark.parametrize(
    ("at", "cutoff"),
    [
        (ny(2019, 3, 31, 18, 10), ny(2019, 3, 31, 17, 0)),
        (ny(2019, 3, 31, 17, 0), ny(2019, 3, 31, 17, 0)),
        (ny(2019, 3, 31, 16, 59), ny(2019, 3, 30, 17, 0)),
        (ny(2019, 1, 2, 3, 0), ny(2019, 1, 1, 17, 0)),
    ],
)
def test_the_cutoff_is_the_latest_five_pm_in_new_york(at: datetime, cutoff: datetime) -> None:
    from quant.jobs.refit import CUTOFF_ET, refit_cutoff

    assert CUTOFF_ET.hour == 17
    assert refit_cutoff(at) == cutoff
    with pytest.raises(ClockError):
        refit_cutoff(datetime(2019, 3, 31, 18))  # noqa: DTZ001 (naive on purpose)


@RED
@pytest.mark.spec("SPEC§5", "ARCH§7.3")
@pytest.mark.parametrize(
    ("fit_end", "at", "due"),
    [
        # A first model: its window ends at New York midnight 2019-01-01, last data day 12-31.
        (ny(2019, 1, 1, 0, 0), ny(2019, 1, 29, 17, 30), False),
        (ny(2019, 1, 1, 0, 0), ny(2019, 1, 30, 17, 30), True),
        (ny(2019, 1, 1, 0, 0), ny(2019, 2, 4, 17, 30), True),  # skipped evenings: the next one
        # A refit's window ends at its 17:00 cutoff; DST starts 2019-03-10 in between.
        (ny(2019, 3, 1, 17, 0), ny(2019, 3, 30, 17, 30), False),
        (ny(2019, 3, 1, 17, 0), ny(2019, 3, 31, 17, 30), True),
        (ny(2019, 3, 1, 17, 0), ny(2019, 3, 31, 16, 0), False),  # cutoff still 03-30
    ],
)
def test_a_refit_is_due_thirty_calendar_days_after_the_active_models_data(
    fit_end: datetime, at: datetime, due: bool
) -> None:
    from quant.jobs.refit import REFIT_DAYS, refit_due

    assert REFIT_DAYS == 30
    assert refit_due(model_of(three_full(), fit_end=fit_end), at) is due


# --- The rows -----------------------------------------------------------------------------


@RED
@pytest.mark.spec("SPEC§5", "ARCH§7.3", "ARCH§6.2", "I1")
def test_the_refit_reads_through_the_view_at_its_cutoff_and_nothing_later() -> None:
    from quant.jobs.refit import refit_cutoff, refit_data

    w = world()
    cutoff = refit_cutoff(ny(2016, 12, 1, 17, 30))
    current = w.model

    def rows(store: Any) -> TrainingData:
        return refit_data(
            store, "SPY", start=current.fit_start, cutoff=cutoff, features=r1_features()
        )

    data = rows(w.store)
    assert (data.fit_start, data.fit_end) == (current.fit_start, cutoff)
    assert data.feature_config_hash == r1_features().config_hash()
    assert int(data.matrix.available_at.max()) < int(cutoff.timestamp() * 1_000_000)
    # The last session before the cutoff is in: its bars were available by 16:15.
    assert date.fromordinal(int(data.sessions[-1])) == date(2016, 12, 1)
    assert rows(truncated_store(w, cutoff)).data_hash() == data.data_hash()
    garbage, _ = garbage_store(w, cutoff, seed=5)
    assert rows(garbage).data_hash() == data.data_hash()


# --- The outcomes ---------------------------------------------------------------------------


def _ops(at: datetime, model: HmmModel) -> tuple[Any, SimClock]:
    """An ops store whose model pointer names `model` from before `at`."""
    from quant.store.ops import OpsStore

    clock = SimClock(at - timedelta(days=7))
    ops = OpsStore.memory(clock=clock)
    ops.activate("model", model.artifact_hash(), reason="first model")
    clock.advance_to(at)
    return ops, clock


def _run(data: TrainingData, current: HmmModel, at: datetime, ops: Any, artifacts: Path) -> Any:
    from quant.hmm.drift import r1_drift_thresholds
    from quant.jobs.refit import run_refit

    return run_refit(
        data,
        current,
        at=at,
        ops=ops,
        artifacts=artifacts,
        config=tiny_config(),
        thresholds=r1_thresholds(),
        drift=r1_drift_thresholds(),
        provenance=PROVENANCE,
        workers=1,
    )


@RED
@pytest.mark.spec("SPEC§5", "C7", "C5", "ARCH§7.3", "ARCH§9.2")
@pytest.mark.parametrize(
    ("at", "opens"),
    [
        (ny(2017, 6, 30, 17, 30), ny(2017, 7, 3, 9, 30)),  # Friday: over the weekend
        (ny(2017, 7, 3, 17, 30), ny(2017, 7, 5, 9, 30)),  # half day: over the Fourth of July
    ],
    ids=["weekend", "holiday"],
)
def test_a_passing_refit_governs_from_the_next_open_and_ends_a_freeze(
    at: datetime, opens: datetime, tmp_path: Path
) -> None:
    from quant.jobs.refit import ACTIVATE_EVENT
    from quant.store.ops import FreezeCause, RefitOutcomeKind

    current = first_model()
    ops, _ = _ops(at, current)
    ops.record_freeze(FreezeCause.LIVE_DRIFT, "35-bar mean below the baseline")
    data = cut(_bars(), 270)
    with capture_logs() as logs:
        outcome = _run(data, current, at, ops, tmp_path)

    assert outcome.kind is RefitOutcomeKind.ACTIVATED, outcome.reasons
    assert (outcome.at, outcome.cutoff, outcome.previous) == (
        at,
        data.fit_end,
        current.artifact_hash(),
    )
    assert outcome.reasons == ()
    model = outcome.model
    assert (model.params.n_states, model.params.covariance_type) == (2, "diag")  # K fixed
    assert (model.fit_start, model.fit_end) == (data.fit_start, data.fit_end)
    assert model.labels == outcome.match.inherited
    assert model.training.info["match"]["previous"] == current.artifact_hash()
    assert dict(model.training.info["restarts"]["seeded"])["previous"] == current.artifact_hash()
    assert outcome.directory == tmp_path / "models" / model.artifact_hash()
    assert load_model(outcome.directory).artifact_hash() == model.artifact_hash()
    assert outcome.drift.passed
    assert outcome.sanity.passed

    # The refit doesn't govern that evening or any bar before the next open.
    assert outcome.event.effective_at == opens
    for t in (at, at + timedelta(hours=3), opens - US):
        assert ops.current("model", t).artifact == current.artifact_hash(), t
    assert ops.current("model", opens).artifact == model.artifact_hash()
    # The passing refit ends the freeze exactly when its model starts governing.
    assert ops.frozen(opens - US)
    assert not ops.frozen(opens)
    assert any(entry["event"] == ACTIVATE_EVENT for entry in logs)


@RED
@pytest.mark.spec("SPEC§5", "C5", "D6", "ARCH§7.3")
def test_a_refit_that_fails_a_drift_check_is_stored_not_activated_and_freezes_entries(
    tmp_path: Path,
) -> None:
    from quant.jobs.refit import FREEZE_EVENT
    from quant.store.ops import RefitOutcomeKind

    current = drifted(first_model())
    at = ny(2017, 6, 30, 17, 30)
    ops, _ = _ops(at, current)
    with capture_logs() as logs:
        outcome = _run(cut(_bars(), 270), current, at, ops, tmp_path)

    assert outcome.kind is RefitOutcomeKind.FROZEN
    assert "transition" in {f.rule for f in outcome.drift.findings}
    assert outcome.reasons == outcome.drift.reasons
    # Stored: the artifact is saved, with its inherited labels.
    assert load_model(outcome.directory).artifact_hash() == outcome.model.artifact_hash()
    # Not activated: the current model keeps running, now and after the next open.
    assert [p.artifact for p in ops.pointers("model")] == [current.artifact_hash()]
    assert ops.current("model", ny(2017, 7, 5, 12, 0)).artifact == current.artifact_hash()
    assert outcome.event.effective_at is None
    # Entries freeze from the recording on.
    assert not ops.frozen(at - US)
    assert ops.frozen(at)
    assert ops.frozen(ny(2017, 7, 3, 9, 30))
    (event,) = ops.refit_events()
    assert (event.outcome, event.artifact, event.reasons) == (
        RefitOutcomeKind.FROZEN,
        outcome.model.artifact_hash(),
        outcome.reasons,
    )
    assert any(entry["event"] == FREEZE_EVENT and entry["log_level"] == "warning" for entry in logs)


def _sanity_failure(current: HmmModel) -> TrainResult:
    report = SanityReport(
        _f([0.99, 0.01]), _f([20.0, 10.0]), _f([1.0, 1.0]), ("state 1: occupancy 0.01 < 0.02",)
    )
    return TrainResult(current, report, None)


@RED
@pytest.mark.spec("SPEC§4", "SPEC§5", "ARCH§7.3")
@pytest.mark.parametrize("how", ["sanity report", "no filtered mass"])
def test_a_sanity_reject_keeps_the_current_model_and_alerts(
    how: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant.jobs.refit as job
    from quant.store.ops import RefitOutcomeKind

    current = first_model()

    def fake(*args: Any, **kwargs: Any) -> TrainResult:
        assert kwargs["previous"] is current
        if how == "no filtered mass":
            raise ModelError("sanity reject: state 1: occupancy 0.0 < 0.02")
        return _sanity_failure(current)

    monkeypatch.setattr(job, "train_candidate", fake)
    at = ny(2017, 6, 30, 17, 30)
    ops, _ = _ops(at, current)
    with capture_logs() as logs:
        outcome = _run(cut(_bars(), 270), current, at, ops, tmp_path)

    assert outcome.kind is RefitOutcomeKind.REJECTED
    assert (outcome.model, outcome.directory, outcome.match, outcome.drift) == (
        None,
        None,
        None,
        None,
    )
    assert any("occupancy" in reason for reason in outcome.reasons)
    assert (outcome.sanity is None) is (how == "no filtered mass")
    assert not (tmp_path / "models").exists() or not any((tmp_path / "models").iterdir())
    assert [p.artifact for p in ops.pointers("model")] == [current.artifact_hash()]
    assert not ops.frozen(ny(2017, 7, 3, 12, 0))
    (event,) = ops.refit_events()
    assert (event.outcome, event.artifact) == (RefitOutcomeKind.REJECTED, None)
    from quant.jobs.refit import REJECT_EVENT

    assert any(entry["event"] == REJECT_EVENT and entry["log_level"] == "warning" for entry in logs)


@RED
@pytest.mark.spec("SPEC§5", "ARCH§7.3", "T-NL2")
@pytest.mark.parametrize(
    "fault", ["start", "not expanding", "future rows", "feature hash", "not the active model"]
)
def test_a_refit_of_the_wrong_window_or_model_is_refused_before_training(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant.jobs.refit as job

    monkeypatch.setattr(job, "train_candidate", lambda *a, **k: pytest.fail("trained"))
    current = first_model()
    at = ny(2017, 6, 30, 17, 30)
    data = cut(_bars(), 270)
    active, run_at = current, at
    if fault == "start":
        data = cut(_bars(), 270, skip=1)
    elif fault == "not expanding":
        data = cut(_bars(), 240)
    elif fault == "future rows":
        run_at = data.fit_end - US  # the job runs before its rows were available
    elif fault == "feature hash":
        data = cut(_bars(), 270, feature_hash="cd" * 32)
    else:
        active = drifted(current)
    ops, _ = _ops(at, active)
    with pytest.raises(ModelError):
        _run(data, current, run_at, ops, tmp_path)
    assert ops.refit_events() == ()
    assert not (tmp_path / "models").exists()


@RED
@pytest.mark.spec("SPEC§5", "ARCH§7.3")
def test_the_evening_job_refits_only_when_due(tmp_path: Path) -> None:
    from quant.hmm.drift import r1_drift_thresholds
    from quant.jobs.refit import refit

    w = world()
    current = w.model  # its data ends 2016-10-31

    def job(at: datetime, ops: Any) -> Any:
        return refit(
            w.store,
            current,
            at=at,
            symbol="SPY",
            features=r1_features(),
            ops=ops,
            artifacts=tmp_path,
            config=tiny_config(),
            thresholds=r1_thresholds(),
            drift=r1_drift_thresholds(),
            provenance=PROVENANCE,
            workers=1,
        )

    early = ny(2016, 11, 29, 17, 30)
    ops, _ = _ops(early, current)
    assert job(early, ops) is None
    assert ops.refit_events() == ()

    due = ny(2016, 11, 30, 17, 30)
    ops, _ = _ops(due, current)
    outcome = job(due, ops)
    assert outcome is not None
    assert outcome.cutoff == ny(2016, 11, 30, 17, 0)
    assert outcome.event.cutoff == outcome.cutoff
    if outcome.model is not None:
        assert (outcome.model.fit_start, outcome.model.fit_end) == (
            current.fit_start,
            outcome.cutoff,
        )
    assert len(ops.refit_events()) == 1


@RED
@pytest.mark.spec("SPEC§5")
def test_the_stored_refit_keeps_its_rule_derived_labels_beside_the_inherited_ones(
    tmp_path: Path,
) -> None:
    current = first_model()
    at = ny(2017, 6, 30, 17, 30)
    ops, _ = _ops(at, current)
    outcome = _run(cut(_bars(), 270), current, at, ops, tmp_path)
    info = outcome.model.training.info
    derived = [Label(state["label"]) for state in info["states"]]
    assert [s.label for s in outcome.match.derived.states] == derived
    assert list(info["match"]["derived"]) == [label.value for label in derived]
    assert [s.label for s in outcome.model.labels.states] == [
        current.labels.label_of(o) for o in outcome.match.old_of_new
    ]
