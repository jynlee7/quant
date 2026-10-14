"""B4: training, restarts, selection on known HMMs, the report and determinism (spec §4, I2).

The fast tier fits small grids with two restarts; the full tier runs r1's 50 restarts over the
8-candidate grid on 20,000 bars and reports how long a whole selection takes (PLAN B4).
"""

from __future__ import annotations

import functools
import math
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from structlog.testing import capture_logs

from quant.core.errors import ClockError, ModelError
from tests.support.hmm import FEATURE_HASH, Truth, identity_columns, identity_scaler, three_full
from tests.support.train import (
    R1_BASE_SEED,
    calendar,
    match_states,
    r1_candidates,
    ref_bic,
    ref_n_params,
    three_diag,
    training_data,
)

TRUTHS = {"three_full": three_full, "three_diag": three_diag}
PROVENANCE = {"code_commit": "0" * 40, "lockfile_sha256": "1" * 64}
FORBIDDEN = ("score", "score_samples", "predict", "predict_proba", "decode", "sample")


def _candidate(k: int, covariance_type: str) -> Any:
    from quant.hmm.select import Candidate

    return Candidate(n_states=k, covariance_type=covariance_type)  # type: ignore[arg-type]


def _config(grid: list[tuple[int, str]], restarts: int = 2, **changes: Any) -> Any:
    """r1's config with a smaller grid and fewer restarts, for the fast tier."""
    from quant.hmm.train import r1_train_config

    base = r1_train_config()
    values = dict(base) | {
        "n_restarts": restarts,
        "candidates": tuple(_candidate(k, c) for k, c in grid),
    }
    return type(base)(**(values | changes))


def _thresholds() -> Any:
    from quant.hmm.labels import r1_thresholds

    return r1_thresholds()


# --- Config and training data ---------------------------------------------------------------


@pytest.mark.spec("SPEC§4", "PLAN§5")
def test_r1_trains_fifty_restarts_over_the_eight_candidates() -> None:
    from quant.hmm.train import r1_train_config

    config = r1_train_config()
    assert config.base_seed == R1_BASE_SEED == 20160104
    assert config.n_restarts == 50
    assert [(c.n_states, c.covariance_type) for c in config.candidates] == r1_candidates()
    assert (config.bootstrap_draws, config.validation_months) == (1000, 12)
    assert (config.n_iter, config.tol, config.min_covar) == (1000, 1e-2, 1e-3)
    assert r1_train_config() is not config


@pytest.mark.spec("SPEC§4")
@pytest.mark.parametrize(
    "change",
    [
        {"n_restarts": 0},
        {"base_seed": -1},
        {"candidates": ()},
        {"candidates": "duplicate"},
        {"candidates": "one state"},
        {"n_iter": 0},
        {"tol": 0.0},
        {"min_covar": 0.0},
        {"bootstrap_draws": 1},
        {"validation_months": 0},
    ],
)
def test_a_config_out_of_range_is_refused(change: dict[str, Any]) -> None:
    import pydantic

    from quant.hmm.train import TrainConfig, r1_train_config

    values = dict(r1_train_config())
    if change.get("candidates") == "duplicate":
        change = {"candidates": (_candidate(3, "full"), _candidate(3, "full"))}
    elif change.get("candidates") == "one state":
        change = {"candidates": (_candidate(1, "diag"), _candidate(2, "diag"))}
    with pytest.raises(pydantic.ValidationError):
        TrainConfig(**(values | change))


@pytest.mark.spec("SPEC§4")
def test_training_data_hash_names_its_contents() -> None:
    data = training_data(three_full(), 700, 1)
    same = training_data(three_full(), 700, 1)
    assert data.data_hash() == same.data_hash()
    assert len(data.data_hash()) == 64
    assert training_data(three_full(), 700, 2).data_hash() != data.data_hash()
    other = training_data(three_full(), 700, 1, feature_hash="cd" * 32)
    assert other.data_hash() != data.data_hash()
    with pytest.raises(ValueError, match="read-only"):
        data.returns[0] = 1.0


def _data_parts(n: int = 70) -> dict[str, Any]:
    from quant.features.stream import FeatureMatrix

    t_open, available, sessions, start, end = calendar(n)
    values = np.random.default_rng(1).normal(size=(n, 2))
    return {
        "matrix": FeatureMatrix(identity_columns(2), t_open, available, values),
        "returns": values[:, 0].copy(),
        "sessions": sessions,
        "fit_start": start,
        "fit_end": end,
        "feature_config_hash": FEATURE_HASH,
    }


@pytest.mark.spec("SPEC§4")
@pytest.mark.parametrize(
    "fault",
    [
        "short returns",
        "nan return",
        "decreasing sessions",
        "int32 sessions",
        "row after end",
        "row before start",
        "empty window",
        "bad hash",
    ],
)
def test_training_data_is_checked(fault: str) -> None:
    from quant.hmm.train import TrainingData

    parts = _data_parts()
    if fault == "short returns":
        parts["returns"] = parts["returns"][:-1]
    elif fault == "nan return":
        parts["returns"][3] = math.nan
    elif fault == "decreasing sessions":
        parts["sessions"] = parts["sessions"][::-1].copy()
    elif fault == "int32 sessions":
        parts["sessions"] = parts["sessions"].astype(np.int32)
    elif fault == "row after end":
        parts["fit_end"] = datetime(1990, 1, 3, tzinfo=UTC)
    elif fault == "row before start":
        parts["fit_start"] = datetime(1990, 1, 5, tzinfo=UTC)
    elif fault == "empty window":
        parts["fit_end"] = parts["fit_start"]
    else:
        parts["feature_config_hash"] = "AB" * 32
    with pytest.raises(ModelError):
        TrainingData(**parts)


@pytest.mark.spec("SPEC§4")
def test_training_data_needs_aware_times() -> None:
    from quant.hmm.train import TrainingData

    parts = _data_parts()
    parts["fit_start"] = parts["fit_start"].replace(tzinfo=None)
    with pytest.raises(ClockError):
        TrainingData(**parts)


# --- One candidate: restarts, the winner, our log-likelihood --------------------------------


def _fit(truth: Truth, n: int, restarts: int, *, seed: int = 1, workers: int | None = 1) -> Any:
    from quant.hmm.train import fit_candidate
    from tests.support.hmm import sample

    _, x = sample(truth, n, seed)
    config = _config([(truth.k, truth.covariance_type)], restarts)
    return x, fit_candidate(
        x,
        identity_scaler(truth.d),
        _candidate(truth.k, truth.covariance_type),
        columns=identity_columns(truth.d),
        config=config,
        workers=workers,
    )


@pytest.mark.spec("SPEC§4", "ARCH§7.1")
def test_restarts_use_base_seed_plus_i_and_the_best_wins() -> None:
    from quant.hmm.filter import forward_pass

    x, fit = _fit(three_full(), 1500, 4)
    assert fit.seeds == tuple(R1_BASE_SEED + i for i in range(4))
    assert fit.failed == ()
    assert len(fit.logliks) == len(fit.iterations) == 4
    assert fit.choice.best == int(np.argmax(fit.logliks))
    # The log-likelihood is ours: Σ l_t of our forward pass over the raw rows.
    trace = forward_pass(fit.params, identity_scaler(2), x, columns=identity_columns(2))
    assert fit.loglik == float(np.sum(trace.loglik))
    assert fit.trace.loglik.tobytes() == trace.loglik.tobytes()
    assert fit.trace.state_probs.tobytes() == trace.state_probs.tobytes()


@pytest.mark.spec("SPEC§4")
def test_fitted_covariances_are_exactly_symmetric_and_diag_stays_diagonal() -> None:
    _, full = _fit(three_full(), 1500, 1)
    for covariance in full.params.covars:
        assert np.array_equal(covariance, covariance.T)
    _, diag = _fit(three_diag(), 1500, 1)
    assert diag.params.covariance_type == "diag"
    for covariance in diag.params.covars:
        assert np.array_equal(covariance, np.diag(np.diag(covariance)))


@pytest.mark.spec("SPEC§4")
def test_one_restart_is_an_unstable_optimum_and_warns() -> None:
    with capture_logs() as logs:
        _, fit = _fit(three_full(), 1500, 1)
    assert (fit.choice.near_best, fit.choice.stable) == (1, False)
    assert [e["event"] for e in logs if e["log_level"] == "warning"] == ["hmm optimum unstable"]


@pytest.mark.spec("I2")
def test_training_calls_only_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """I2 at run time: every other GaussianHMM method raises, and fit runs once a restart."""
    from hmmlearn.hmm import GaussianHMM

    fit = GaussianHMM.fit
    calls: list[int] = []

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return fit(self, *args, **kwargs)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("training may call only GaussianHMM.fit (I2)")

    monkeypatch.setattr(GaussianHMM, "fit", counted)
    for name in FORBIDDEN:
        monkeypatch.setattr(GaussianHMM, name, forbidden)
    _, result = _fit(three_full(), 1000, 3, workers=1)
    assert len(calls) == 3
    assert result.failed == ()


@pytest.mark.spec("SPEC§4")
@pytest.mark.parametrize("truth_name", ["three_full", "three_diag"])
def test_training_recovers_a_known_hmm_up_to_relabeling(truth_name: str) -> None:
    truth = TRUTHS[truth_name]()
    _, fit = _fit(truth, 20_000, 2, workers=None)
    params = fit.params
    perm = list(match_states(truth.means, params.means))
    np.testing.assert_allclose(params.means[perm], truth.means, atol=0.1)
    np.testing.assert_allclose(params.covars[perm], truth.covars, atol=0.2)
    np.testing.assert_allclose(params.transmat[np.ix_(perm, perm)], truth.transmat, atol=0.03)


@pytest.mark.spec("SPEC§4")
def test_fit_candidate_refuses_other_columns_and_bad_workers() -> None:
    from quant.hmm.train import fit_candidate

    x = np.random.default_rng(1).normal(size=(200, 2))
    args = (x, identity_scaler(2), _candidate(2, "diag"))
    with pytest.raises(ModelError):
        fit_candidate(*args, columns=("a", "b"), config=_config([(2, "diag")]), workers=1)
    with pytest.raises(ModelError):
        fit_candidate(*args, columns=identity_columns(2), config=_config([(2, "diag")]), workers=0)


# --- Selection on known HMMs -----------------------------------------------------------------

FAST_GRID = [(k, c) for k in (2, 3, 4) for c in ("diag", "full")]


@functools.cache
def _selected(truth_name: str, n: int = 6000) -> tuple[Any, Any, Any]:
    """(data, config, result) of a fast selection, computed once per truth."""
    from quant.hmm.train import select_and_train

    data = training_data(TRUTHS[truth_name](), n, 1)
    config = _config(FAST_GRID, 2)
    result = select_and_train(data, thresholds=_thresholds(), config=config, provenance=PROVENANCE)
    return data, config, result


@pytest.mark.spec("SPEC§4")
@pytest.mark.parametrize("truth_name", ["three_full", "three_diag"])
def test_selection_picks_three_states_and_the_generators_covariance_type(truth_name: str) -> None:
    truth = TRUTHS[truth_name]()
    _, _, result = _selected(truth_name)
    chosen = result.selection.chosen
    assert (chosen.n_states, chosen.covariance_type) == (3, truth.covariance_type)
    assert result.model.params.n_states == 3
    assert result.model.params.covariance_type == truth.covariance_type
    assert result.sanity.passed


@pytest.mark.spec("SPEC§4", "ARCH§7.1")
def test_the_chosen_model_is_its_full_window_refit() -> None:
    from quant.hmm.train import train_candidate

    data, config, result = _selected("three_full")
    refit = train_candidate(
        data,
        result.selection.chosen,
        thresholds=_thresholds(),
        config=config,
        provenance=PROVENANCE,
    )
    for name, array in result.model.arrays().items():
        assert refit.model.arrays()[name].tobytes() == array.tobytes(), name
    assert refit.model.training.restart_logliks == result.model.training.restart_logliks
    assert refit.selection is None
    assert refit.model.training.info["selection"] is None
    # The scaler is fit on the whole window.
    np.testing.assert_array_equal(result.model.scaler.mean, data.matrix.values.mean(axis=0))
    np.testing.assert_array_equal(result.model.scaler.std, data.matrix.values.std(axis=0))
    assert (result.model.fit_start, result.model.fit_end) == (data.fit_start, data.fit_end)


@pytest.mark.spec("SPEC§4", "ARCH§7.2")
def test_the_training_record_and_report_are_complete() -> None:
    from quant.core.epoch import to_us
    from quant.hmm.filter import forward_pass
    from quant.hmm.labels import BARS_PER_YEAR
    from quant.hmm.select import validation_start

    data, config, result = _selected("three_full")
    model = result.model
    record = model.training
    columns = data.matrix.columns
    assert record.base_seed == R1_BASE_SEED
    assert record.seeds == (R1_BASE_SEED, R1_BASE_SEED + 1)
    assert record.best == int(np.argmax(record.restart_logliks))
    trace = forward_pass(model.params, model.scaler, data.matrix.values, columns=columns)
    loglik = float(np.sum(trace.loglik))
    assert record.restart_logliks[record.best] == loglik

    info = record.info
    assert set(info) == {
        "data_hash", "config", "candidate", "n_rows", "loglik", "bic", "restarts", "sanity",
        "states", "selection", "provenance", "versions", "blas",
    }  # fmt: skip
    assert info["data_hash"] == data.data_hash()
    assert dict(info["provenance"]) == PROVENANCE
    assert info["config"]["n_restarts"] == config.n_restarts
    assert info["versions"]["hmmlearn"] == "0.3.3"
    assert set(info["versions"]) == {"python", "numpy", "scipy", "scikit-learn", "hmmlearn"}
    assert set(info["blas"]) == {"name", "version"}
    n, k = len(data.matrix), model.params.n_states
    p = ref_n_params(k, 2, model.params.covariance_type)
    assert (info["n_rows"], info["loglik"]) == (n, loglik)
    assert info["bic"] == ref_bic(loglik, p, n)
    assert dict(info["candidate"]) == {
        "n_states": k, "covariance_type": model.params.covariance_type, "n_params": p,
    }  # fmt: skip
    assert info["restarts"]["near_best"] >= 1
    assert info["restarts"]["failed"] == ()
    assert info["sanity"]["passed"] is True
    assert info["sanity"]["reasons"] == ()

    # Spec §4's per-state report.
    occupancy = trace.state_probs.mean(axis=0)
    sigma_all = float(data.returns.std())
    assert len(info["states"]) == k
    for state, row in enumerate(info["states"]):
        a = float(model.params.transmat[state, state])
        assert row["state"] == state
        assert row["label"] == model.labels.label_of(state).value
        assert (row["v"], row["d"]) == (model.labels.states[state].v, model.labels.states[state].d)
        assert row["duration_bars"] == pytest.approx(1 / (1 - a), rel=1e-12)
        assert row["duration_sessions"] == pytest.approx(row["duration_bars"] / 7, rel=1e-12)
        assert row["occupancy"] == pytest.approx(occupancy[state], rel=1e-12)
        # The annualized return and volatility are the labels' mu_k and sigma_k.
        assert row["volatility"] == pytest.approx(row["v"] * sigma_all * math.sqrt(BARS_PER_YEAR))
        assert row["mean_return"] / row["volatility"] == pytest.approx(row["d"], rel=1e-9)

    # The BIC and validation table, in grid order.
    selection = info["selection"]
    assert selection["rule"] in ("close_on_both", "simpler_winner")
    assert dict(selection["chosen"]) == {"n_states": 3, "covariance_type": "full"}
    table = selection["table"]
    assert [(r["n_states"], r["covariance_type"]) for r in table] == FAST_GRID
    for row, score in zip(table, result.selection.scores, strict=True):
        assert row["n_params"] == ref_n_params(row["n_states"], 2, row["covariance_type"])
        assert row["bic"] == ref_bic(row["loglik"], row["n_params"], n)
        assert (row["validation_mean"], row["validation_se"]) == (
            score.validation.mean,
            score.validation.se,
        )
        assert row["validation_se"] > 0
    chosen_row = table[FAST_GRID.index((3, "full"))]
    assert (chosen_row["loglik"], chosen_row["bic"]) == (info["loglik"], info["bic"])
    validation = selection["validation"]
    start = validation_start(data.fit_end, 12)
    assert datetime.fromisoformat(validation["start"]) == start
    block = data.matrix.available_at >= to_us(start)
    assert (
        validation["rows"] == int(block.sum()) == len(result.selection.scores[0].validation.loglik)
    )
    assert validation["sessions"] == len(np.unique(data.sessions[block]))
    assert (validation["draws"], validation["seed"]) == (1000, R1_BASE_SEED)
    distribution = selection["distribution"]
    assert set(distribution["quantiles"]) == {"0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99"}
    assert distribution["mean"] == chosen_row["validation_mean"]


# --- Determinism ------------------------------------------------------------------------------


def _small_run(workers: int | None, *, base_seed: int = R1_BASE_SEED) -> Any:
    from quant.hmm.train import select_and_train

    data = training_data(three_full(), 3000, 4)
    config = _config([(2, "diag"), (3, "full")], 2, base_seed=base_seed)
    return select_and_train(
        data, thresholds=_thresholds(), config=config, provenance=PROVENANCE, workers=workers
    ).model


@pytest.mark.spec("SPEC§4", "ARCH§14")
def test_the_same_seeds_give_the_same_artifact_hash_across_runs_and_pool_or_serial() -> None:
    serial = _small_run(1)
    again = _small_run(1)
    pooled = _small_run(2)
    assert serial.artifact_hash() == again.artifact_hash() == pooled.artifact_hash()
    for name, array in serial.arrays().items():
        assert pooled.arrays()[name].tobytes() == array.tobytes(), name


@pytest.mark.spec("SPEC§4", "ARCH§14")
def test_another_base_seed_is_another_model() -> None:
    from quant.hmm.train import train_candidate

    data = training_data(three_full(), 1500, 4)
    hashes = set()
    for seed in (R1_BASE_SEED, R1_BASE_SEED + 100):
        result = train_candidate(
            data,
            _candidate(3, "full"),
            thresholds=_thresholds(),
            config=_config([(3, "full")], 2, base_seed=seed),
            provenance=PROVENANCE,
            workers=1,
        )
        assert result.model.training.seeds == (seed, seed + 1)
        hashes.add(result.model.artifact_hash())
    assert len(hashes) == 2


# --- The full run (PLAN B4's "done when") -----------------------------------------------------


@pytest.mark.full
@pytest.mark.spec("SPEC§4", "PLAN§5")
@pytest.mark.parametrize("truth_name", ["three_full", "three_diag"])
def test_r1_selection_on_twenty_thousand_bars(
    truth_name: str, record_property: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """r1's 50 restarts over the 8 candidates on 20,000 bars: selection picks K = 3 and the
    generator's covariance type, the model recovers the generator in standardized units, and
    the time a whole selection takes is measured and reported."""
    from quant.hmm.train import r1_train_config, select_and_train

    truth = TRUTHS[truth_name]()
    data = training_data(truth, 20_000, 7)
    config = r1_train_config()
    started = time.perf_counter()
    result = select_and_train(data, thresholds=_thresholds(), config=config, provenance=PROVENANCE)
    seconds = time.perf_counter() - started
    record_property("selection_seconds", round(seconds, 1))
    with capsys.disabled():
        print(f"\nB4 full K x covariance selection ({truth_name}, 20,000 bars): {seconds:.1f} s")

    chosen = result.selection.chosen
    assert (chosen.n_states, chosen.covariance_type) == (3, truth.covariance_type)
    assert len(result.model.training.seeds) == 50
    assert len(result.model.training.info["selection"]["table"]) == 8
    # The model works in standardized units: z = (x - m) / s.
    m, s = result.model.scaler.mean, result.model.scaler.std
    means = (truth.means - m) / s
    covars = truth.covars / np.outer(s, s)
    params = result.model.params
    perm = list(match_states(means, params.means))
    np.testing.assert_allclose(params.means[perm], means, atol=0.1)
    np.testing.assert_allclose(params.covars[perm], covars, atol=0.15)
    np.testing.assert_allclose(params.transmat[np.ix_(perm, perm)], truth.transmat, atol=0.03)
