"""Model training: the only module that imports hmmlearn, and it calls only `fit` (I2). Built in B4.

**One candidate** (K, covariance type) is trained on a block of rows by `fit_candidate`:

1. The rows are standardized with the given scaler, z = `scaler.standardize(rows, columns=...)`.
2. Restart i, for i = 0 .. n_restarts - 1, uses seed `config.base_seed + i` and fits
   `hmmlearn.hmm.GaussianHMM(n_components=K, covariance_type=..., min_covar=config.min_covar,
   n_iter=config.n_iter, tol=config.tol, random_state=seed, implementation="log",
   init_params="stmc", params="stmc")` on z as one sequence. `fit` is the only hmmlearn method
   ever called: never `score`, `score_samples`, `predict`, `predict_proba`, `decode` or
   `sample` (I2).
3. Its parameters become a `GaussianParams`: `startprob_`, `transmat_`, `means_` and
   `covars_` (hmmlearn returns full (K, d, d) matrices for both types), each covariance
   symmetrized exactly as 0.5 * (Σ + Σᵀ), so a diag model's off-diagonal entries stay exactly 0.
4. Its training log-likelihood is ours, never hmmlearn's: `float(numpy.sum(forward_pass(params,
   scaler, rows, columns=...).loglik))`, the density of the raw rows (Jacobian included).
5. A restart *fails* when hmmlearn raises, `GaussianParams` refuses its parameters, or the
   forward pass raises ModelError. A failed restart is left out of the record's `seeds` and
   `restart_logliks` and listed in `info` instead (index, seed and the exception's type name);
   when every restart fails, the candidate raises ModelError.
6. The winner is `select.choose_restart` over the recorded log-likelihoods in restart order:
   the highest, a tie going to the lowest index, and a structlog warning "hmm optimum
   unstable" when fewer than 3 restarts (the winner included) land within 0.1% of the best.

**Parallelism** (ARCHITECTURE §7.1, §14): `workers=1` runs the restarts one after another in
the calling process. Otherwise they run in a `concurrent.futures.ProcessPoolExecutor` with the
"spawn" context and `workers` processes (None: `os.cpu_count()`), and results are collected
in restart order. Every fit, in a pool or serial, runs inside
`threadpoolctl.threadpool_limits(limits=1)` (single-threaded BLAS and OpenMP; threadpoolctl
comes with scikit-learn, and the green commit lists it in pyproject.toml), so the pool and the
serial path give the same bits and the same artifact hash. `select_and_train` sends every
restart of every candidate, both fits, through one pool. The worker count is never recorded:
it can't change the model.

**Selection** (`select_and_train`, spec §4, the rules in quant.hmm.select): for every candidate
of `config.candidates`, in order,

- the *full-window fit*: `fit_candidate` on every row of the fit window, with the scaler fit on
  them (`Scaler.fit(data.matrix, start=data.fit_start, end=data.fit_end)`), gives the training
  log-likelihood, the BIC and the sanity report;
- the *validation fit*: `fit_candidate` on the rows before `select.validation_start(fit_end,
  config.validation_months)`, with a scaler fit on those rows only, so nothing from the
  validation block reaches the model or its scaler; `select.validation_score` then scores the
  block from the end of that fit's forward pass, with `config.bootstrap_draws` draws and seed
  `config.base_seed`, the same draws for every candidate.

`select.select` picks the candidate. "Refit the winner on the full window" is its full-window
fit, already done: same rows, scaler, seeds and config, so `train_candidate(data, chosen, ...)`
gives the same parameters bit for bit.

**The model** is the chosen fit's parameters, the full-window scaler (`ScalerParams.from_scaler`),
the fit window, the labels (`hmm.labels.label_states(data.returns, trace.state_probs,
thresholds)` with the winner's filtered probabilities over the window) and a TrainingRecord:
`base_seed`, the recorded `seeds` and `restart_logliks` of the full-window fit, `best` from
`choose_restart`, and `info` (`TrainResult` lists its keys). Nothing in it depends on wall time,
paths or the worker count, so the same data, seeds and config give the same artifact hash.

Times are UTC. "Now" is never read (K6).
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import platform
import re
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from typing import Any, Self

import numpy as np
from hmmlearn.hmm import GaussianHMM
from pydantic import model_validator
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from quant.core.canonical import canonical_dumps, sha256_hex
from quant.core.clock import ensure_utc
from quant.core.epoch import to_us
from quant.core.errors import CanonicalJSONError, ModelError
from quant.core.model import StrictModel
from quant.data.frames import Floats, Ints
from quant.features.scaler import Scaler
from quant.features.stream import FeatureMatrix, freeze_arrays
from quant.hmm import select as selection_rules
from quant.hmm.filter import ForwardTrace, forward_pass
from quant.hmm.labels import BARS_PER_YEAR, LabelThresholds, label_states
from quant.hmm.model import GaussianParams, HmmModel, ScalerParams, TrainingRecord
from quant.hmm.select import Candidate, RestartChoice, SanityReport, Selection


class TrainConfig(StrictModel):
    """Everything that defines a training run, recorded in `info["config"]` (so in the hash).

    - `base_seed`: restart i uses `base_seed + i`; the bootstrap uses `base_seed`;
    - `n_restarts`: restarts per fit (spec §4: 50);
    - `candidates`: the selection grid, in the order the table reports it;
      `train_candidate` ignores it;
    - `n_iter`, `tol`, `min_covar`: passed to GaussianHMM (EM's iteration cap, its convergence
      threshold on the total log-likelihood gain of the standardized rows, and the floor added
      to every covariance's diagonal);
    - `bootstrap_draws`: draws of the session-block bootstrap (PLAN B4: 1,000);
    - `validation_months`: the validation block's length (spec §4: the last 12 months).
    """

    base_seed: int
    n_restarts: int
    candidates: tuple[Candidate, ...]
    n_iter: int
    tol: float
    min_covar: float
    bootstrap_draws: int
    validation_months: int

    @model_validator(mode="after")
    def _valid(self) -> Self:
        """Return self when base_seed ≥ 0, n_restarts ≥ 1, `candidates` is non-empty with no
        duplicate and every K ≥ 2, n_iter ≥ 1, tol > 0, min_covar > 0, bootstrap_draws ≥ 2
        and validation_months ≥ 1. Otherwise raise ValueError naming the field (pydantic
        reports a ValidationError)."""
        for name, minimum in (
            ("base_seed", 0),
            ("n_restarts", 1),
            ("n_iter", 1),
            ("bootstrap_draws", 2),
            ("validation_months", 1),
        ):
            if getattr(self, name) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        for name in ("tol", "min_covar"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            not self.candidates
            or len(set(self.candidates)) != len(self.candidates)
            or any(c.n_states < 2 for c in self.candidates)
        ):
            raise ValueError("candidates must be distinct, nonempty and have at least two states")
        return self


def r1_train_config() -> TrainConfig:
    """r1's values (PLAN §5, spec §4): base_seed 20160104, 50 restarts, the candidates K ∈ {2, 3,
    4, 5} × {diag, full} ordered K ascending, diag before full; n_iter 1000, tol 1e-2 (hmmlearn's
    default), min_covar 1e-3 (hmmlearn's default); 1,000 bootstrap draws; a 12-month
    validation block. A new, validated object on each call."""
    return TrainConfig(
        base_seed=20160104,
        n_restarts=50,
        candidates=tuple(
            Candidate(n_states=k, covariance_type=c) for k in (2, 3, 4, 5) for c in ("diag", "full")
        ),
        n_iter=1000,
        tol=1e-2,
        min_covar=1e-3,
        bootstrap_draws=1000,
        validation_months=12,
    )


@dataclass(frozen=True, eq=False)
class TrainingData:
    """The fit window's rows, as the trainer reads them.

    - `matrix`: the warm feature rows of the window (B3's FeatureMatrix), every row's
      available_at in [fit_start, fit_end);
    - `returns` (n,): each row's bar's raw log return ln(C_t / C_{t-1}), for the labels' v and d
      and the report's per-state return and volatility (not a feature);
    - `sessions` (n,): each row's session as `date.toordinal()` of its New York session date,
      nondecreasing: the bootstrap's blocks;
    - `fit_start`, `fit_end`: the fit window [fit_start, fit_end), stored in UTC;
    - `feature_config_hash`: the FeatureConfig hash the rows were computed under.
    """

    matrix: FeatureMatrix
    returns: Floats
    sessions: Ints
    fit_start: datetime
    fit_end: datetime
    feature_config_hash: str

    def __post_init__(self) -> None:
        """Raise ModelError unless: the matrix has n ≥ 2 rows; `returns` is a finite float64
        (n,) array; `sessions` an int64 (n,) array that never decreases; every available_at
        lies in [fit_start, fit_end); fit_start < fit_end; and `feature_config_hash` is 64
        lowercase hex characters. A naive fit_start or fit_end raises ClockError. Store both
        in UTC and the arrays read-only (`freeze_arrays`)."""
        start, end = ensure_utc(self.fit_start), ensure_utc(self.fit_end)
        n = len(self.matrix)
        if (
            n < 2
            or start >= end
            or not isinstance(self.returns, np.ndarray)
            or self.returns.dtype != np.float64
            or self.returns.shape != (n,)
            or not np.isfinite(self.returns).all()
            or not isinstance(self.sessions, np.ndarray)
            or self.sessions.dtype != np.int64
            or self.sessions.shape != (n,)
            or (self.sessions[1:] < self.sessions[:-1]).any()
            or (self.matrix.available_at < to_us(start)).any()
            or (self.matrix.available_at >= to_us(end)).any()
            or re.fullmatch(r"[0-9a-f]{64}", self.feature_config_hash) is None
        ):
            raise ModelError(
                "training data needs aligned returns, sessions, a valid window and hash"
            )
        object.__setattr__(self, "fit_start", start)
        object.__setattr__(self, "fit_end", end)
        freeze_arrays(self, "returns", "sessions")

    def data_hash(self) -> str:
        """`quant.core.canonical.sha256_hex` of {"columns", "fit_start", "fit_end",
        "feature_config_hash", and for each of "t_open", "available_at", "values", "returns"
        and "sessions" {"dtype", "shape", "sha256"}}, each array hashed as HmmModel's are
        (little-endian, C-order bytes). Equal data gives an equal hash; any changed value,
        time or name changes it."""
        arrays = {
            "t_open": self.matrix.t_open,
            "available_at": self.matrix.available_at,
            "values": self.matrix.values,
            "returns": self.returns,
            "sessions": self.sessions,
        }
        return sha256_hex(
            {
                "columns": self.matrix.columns,
                "fit_start": self.fit_start,
                "fit_end": self.fit_end,
                "feature_config_hash": self.feature_config_hash,
                **{
                    name: {
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                        "sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                array, dtype=array.dtype.newbyteorder("<")
                            ).tobytes()
                        ).hexdigest(),
                    }
                    for name, array in arrays.items()
                },
            }
        )


@dataclass(frozen=True, eq=False)
class FailedRestart:
    """A restart that produced no usable model: its index, its seed, and the type name of the
    exception that stopped it."""

    index: int
    seed: int
    error: str


@dataclass(frozen=True, eq=False)
class CandidateFit:
    """One candidate trained on one block of rows (`fit_candidate`).

    - `params`: the winning restart's parameters;
    - `trace`: our forward pass over the rows under `params` (`forward_pass`, from π);
    - `seeds`, `logliks`, `iterations`: one entry per recorded (non-failed) restart, in restart
      order: its seed, its training log-likelihood (Σ l_t), and EM's iteration count
      (`monitor_.iter`);
    - `choice`: `choose_restart(logliks)`; `choice.best` indexes the recorded tuples;
    - `failed`: the failed restarts, in restart order.
    """

    candidate: Candidate
    params: GaussianParams
    trace: ForwardTrace
    seeds: tuple[int, ...]
    logliks: tuple[float, ...]
    iterations: tuple[int, ...]
    choice: RestartChoice
    failed: tuple[FailedRestart, ...]

    @property
    def loglik(self) -> float:
        """The winner's training log-likelihood, `logliks[choice.best]`, which equals
        `float(numpy.sum(trace.loglik))`."""
        return self.logliks[self.choice.best]


def fit_candidate(
    rows: Floats,
    scaler: ScalerParams,
    candidate: Candidate,
    *,
    columns: tuple[str, ...],
    config: TrainConfig,
    workers: int | None = None,
) -> CandidateFit:
    """Fit `candidate` on `rows` (n, d), raw feature rows named by `columns` (exactly
    `scaler.columns`), by the module docstring's steps 1–6 with `config.n_restarts` restarts.

    Raises ModelError when `columns` isn't the scaler's, a row isn't finite, n < K, or every
    restart fails, and when workers < 1.
    """
    return _fit_blocks(((rows, scaler, candidate),), columns, config, workers)[0]


@dataclass(frozen=True, eq=False)
class TrainResult:
    """A trained model and how it was judged.

    - `model`: the HmmModel (module docstring). Its `training.info` holds, as canonical JSON:
      - "data_hash": `data.data_hash()`;
      - "config": `config.model_dump(mode="json")`;
      - "candidate": {"n_states", "covariance_type", "n_params"} of the model;
      - "n_rows", "loglik", "bic": the full-window fit's row count, Σ l_t and BIC;
      - "restarts": {"near_best", "stable", "iterations": [...], "failed": [{"index", "seed",
        "error"}]} of the full-window fit;
      - "sanity": {"passed", "reasons", "occupancy", "durations", "condition"}, one list entry
        per state, an infinite duration written as null;
      - "states": spec §4's per-state report, one object per state in state order:
        {"state", "label", "v", "d", "mean_return" (μ_k · BARS_PER_YEAR), "volatility"
        (sigma_k · sqrt(BARS_PER_YEAR)), "duration_bars" (1 / (1 - a_kk), null when infinite),
        "duration_sessions" (duration_bars / BARS_PER_SESSION, null when infinite),
        "occupancy"}, μ_k and sigma_k by hmm.labels's weighted formulas over the window;
      - "selection": null from `train_candidate`; from `select_and_train` {"rule", "chosen",
        "best_bic", "best_validation", "close" (candidates as {"n_states",
        "covariance_type"}), "se", "validation": {"start" (ISO UTC), "rows", "sessions",
        "draws", "seed"}, "table": one row per candidate in grid order {"n_states",
        "covariance_type", "n_params", "loglik", "bic", "validation_mean", "validation_se",
        "near_best", "stable", "sanity_passed", "sanity_reasons"}, "distribution": the chosen
        candidate's validation l_t as {"mean", "se", "quantiles": {"0.01", "0.05", "0.25",
        "0.5", "0.75", "0.95", "0.99" → numpy.quantile, linear}}};
      - "provenance": the caller's `provenance` mapping, verbatim (code commit, lockfile hash);
      - "versions": {"python", "numpy", "scipy", "scikit-learn", "hmmlearn"}, version strings;
      - "blas": {"name", "version"} of numpy's BLAS, from `numpy.show_config(mode="dicts")`'s
        build dependencies (no paths).
    - `sanity`: the model's SanityReport. `train_candidate` returns a model that fails it
      rather than raising, so a refit job can store and not activate it (spec §5);
    - `selection`: the Selection, or None from `train_candidate`.
    """

    model: HmmModel
    sanity: SanityReport
    selection: Selection | None


def train_candidate(
    data: TrainingData,
    candidate: Candidate,
    *,
    thresholds: LabelThresholds,
    config: TrainConfig,
    provenance: Mapping[str, Any],
    workers: int | None = None,
) -> TrainResult:
    """Train one fixed candidate on the whole fit window: the full-window fit of the module
    docstring, its sanity report and labels, and the model. `config.candidates` is not
    consulted. Raises ModelError as `fit_candidate` and `label_states` do, or when
    `provenance` has no canonical JSON form."""
    scaler = ScalerParams.from_scaler(
        Scaler.fit(data.matrix, start=data.fit_start, end=data.fit_end)
    )
    fit = fit_candidate(
        data.matrix.values,
        scaler,
        candidate,
        columns=data.matrix.columns,
        config=config,
        workers=workers,
    )
    return _build_result(data, scaler, fit, thresholds, config, provenance, None, None)


def select_and_train(
    data: TrainingData,
    *,
    thresholds: LabelThresholds,
    config: TrainConfig,
    provenance: Mapping[str, Any],
    workers: int | None = None,
) -> TrainResult:
    """Spec §4's selection over `config.candidates`, then the chosen candidate's model (module
    docstring). Raises ModelError when the validation block or the rows before it span fewer
    than 2 sessions, when every candidate fails its sanity check (`select.select`), or as
    `train_candidate` does."""
    start = selection_rules.validation_start(data.fit_end, config.validation_months)
    block = data.matrix.available_at >= to_us(start)
    if any(len(np.unique(data.sessions[mask])) < 2 for mask in (block, ~block)):
        raise ModelError("training and validation blocks must each span at least two sessions")
    scaler = ScalerParams.from_scaler(
        Scaler.fit(data.matrix, start=data.fit_start, end=data.fit_end)
    )
    train_scaler = ScalerParams.from_scaler(
        Scaler.fit(data.matrix, start=data.fit_start, end=start)
    )
    rows = data.matrix.values
    blocks = tuple(
        item
        for c in config.candidates
        for item in ((rows, scaler, c), (rows[~block], train_scaler, c))
    )
    fits = _fit_blocks(blocks, data.matrix.columns, config, workers)
    scores = []
    with threadpool_limits(limits=1):
        for i, candidate in enumerate(config.candidates):
            full, train = fits[2 * i : 2 * i + 2]
            validation = selection_rules.validation_score(
                train.params,
                train_scaler,
                rows[~block],
                rows[block],
                data.sessions[block],
                columns=data.matrix.columns,
                draws=config.bootstrap_draws,
                seed=config.base_seed,
            )
            p = candidate.n_free_params(rows.shape[1])
            scores.append(
                selection_rules.CandidateScore(
                    candidate,
                    p,
                    full.loglik,
                    selection_rules.bic(full.loglik, p, len(rows)),
                    validation,
                    selection_rules.sanity_check(full.params, full.trace.state_probs),
                )
            )
        selection = selection_rules.select(tuple(scores), rows.shape[1])
        chosen_index = config.candidates.index(selection.chosen)
        report = _selection_report(selection, fits[::2], start, data.sessions[block], config)
        return _build_result(
            data, scaler, fits[2 * chosen_index], thresholds, config, provenance, selection, report
        )


@dataclass(frozen=True)
class _RestartTask:
    rows: Floats
    scaler: ScalerParams
    candidate: Candidate
    config: TrainConfig
    index: int


@dataclass(frozen=True)
class _RestartResult:
    params: GaussianParams
    loglik: float
    iterations: int


def _restart(task: _RestartTask) -> _RestartResult | FailedRestart:
    """The same fitting and scoring path in the caller and in spawned workers."""
    seed = task.config.base_seed + task.index
    try:
        with threadpool_limits(limits=1):
            z = task.scaler.standardize(task.rows, columns=task.scaler.columns)
            estimator = GaussianHMM(
                n_components=task.candidate.n_states,
                covariance_type=task.candidate.covariance_type,
                min_covar=task.config.min_covar,
                n_iter=task.config.n_iter,
                tol=task.config.tol,
                random_state=seed,
                implementation="log",
                init_params="stmc",
                params="stmc",
            )
            estimator.fit(z)
            covariance = np.asarray(estimator.covars_, dtype=np.float64)
            params = GaussianParams(
                task.candidate.covariance_type,
                np.asarray(estimator.startprob_, dtype=np.float64),
                np.asarray(estimator.transmat_, dtype=np.float64),
                np.asarray(estimator.means_, dtype=np.float64),
                0.5 * (covariance + covariance.transpose(0, 2, 1)),
            )
            trace = forward_pass(params, task.scaler, task.rows, columns=task.scaler.columns)
            loglik = float(np.sum(trace.loglik))
            if not math.isfinite(loglik):
                raise ModelError("training log-likelihood overflows float64")
            return _RestartResult(params, loglik, int(estimator.monitor_.iter))
    except Exception as exc:
        # A bad local optimum must not discard the usable restarts.
        return FailedRestart(task.index, seed, type(exc).__name__)


def _fit_blocks(
    blocks: tuple[tuple[Floats, ScalerParams, Candidate], ...],
    columns: tuple[str, ...],
    config: TrainConfig,
    workers: int | None,
) -> tuple[CandidateFit, ...]:
    """Submit all blocks to one pool and reduce results in seed order."""
    if workers is not None and workers < 1:
        raise ModelError("workers must be positive")
    tasks: list[_RestartTask] = []
    for rows, scaler, candidate in blocks:
        scaler.standardize(rows, columns=columns)
        if rows.ndim != 2 or len(rows) < candidate.n_states or candidate.n_states < 1:
            raise ModelError("a fit needs at least K rows and a positive state count")
        tasks.extend(
            _RestartTask(rows, scaler, candidate, config, i) for i in range(config.n_restarts)
        )
    context = (
        nullcontext(None)
        if workers == 1
        else ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
    )
    with context as pool:
        results = list(map(_restart, tasks) if pool is None else pool.map(_restart, tasks))
    fits = []
    with threadpool_limits(limits=1):
        for block_index, (rows, scaler, candidate) in enumerate(blocks):
            offset = block_index * config.n_restarts
            batch = results[offset : offset + config.n_restarts]
            good = [
                (i, result) for i, result in enumerate(batch) if isinstance(result, _RestartResult)
            ]
            failed = tuple(result for result in batch if isinstance(result, FailedRestart))
            if not good:
                raise ModelError(f"all restarts failed for {candidate}: {failed}")
            logliks = tuple(result.loglik for _, result in good)
            choice = selection_rules.choose_restart(logliks, candidate=candidate)
            params = good[choice.best][1].params
            # Reconstruct to retain array immutability after process transport.
            params = GaussianParams(
                params.covariance_type,
                params.startprob,
                params.transmat,
                params.means,
                params.covars,
            )
            trace = forward_pass(params, scaler, rows, columns=columns)
            fits.append(
                CandidateFit(
                    candidate,
                    params,
                    trace,
                    tuple(config.base_seed + i for i, _ in good),
                    logliks,
                    tuple(result.iterations for _, result in good),
                    choice,
                    failed,
                )
            )
    return tuple(fits)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _selection_report(
    selection: Selection,
    fits: tuple[CandidateFit, ...],
    start: datetime,
    sessions: Ints,
    config: TrainConfig,
) -> dict[str, Any]:
    chosen = next(s for s in selection.scores if s.candidate == selection.chosen)
    return {
        "rule": selection.rule,
        "chosen": selection.chosen.model_dump(mode="json"),
        "best_bic": selection.best_bic.model_dump(mode="json"),
        "best_validation": selection.best_validation.model_dump(mode="json"),
        "close": [c.model_dump(mode="json") for c in selection.close],
        "se": selection.se,
        "validation": {
            "start": start,
            "rows": len(sessions),
            "sessions": len(np.unique(sessions)),
            "draws": config.bootstrap_draws,
            "seed": config.base_seed,
        },
        "table": [
            {
                **score.candidate.model_dump(mode="json"),
                "n_params": score.n_params,
                "loglik": score.loglik,
                "bic": score.bic,
                "validation_mean": score.validation.mean,
                "validation_se": score.validation.se,
                "near_best": fit.choice.near_best,
                "stable": fit.choice.stable,
                "sanity_passed": score.sanity.passed,
                "sanity_reasons": score.sanity.reasons,
            }
            for score, fit in zip(selection.scores, fits, strict=True)
        ],
        "distribution": {
            "mean": chosen.validation.mean,
            "se": chosen.validation.se,
            "quantiles": {
                str(q): float(np.quantile(chosen.validation.loglik, q))
                for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
            },
        },
    }


def _build_result(
    data: TrainingData,
    scaler: ScalerParams,
    fit: CandidateFit,
    thresholds: LabelThresholds,
    config: TrainConfig,
    provenance: Mapping[str, Any],
    selection: Selection | None,
    selection_report: dict[str, Any] | None,
) -> TrainResult:
    with threadpool_limits(limits=1):
        sanity = selection_rules.sanity_check(fit.params, fit.trace.state_probs)
        labels = label_states(data.returns, fit.trace.state_probs, thresholds)
        weights = fit.trace.state_probs.sum(axis=0)
        means = (fit.trace.state_probs * data.returns[:, None]).sum(axis=0) / weights
        deviations = np.sqrt(
            (fit.trace.state_probs * (data.returns[:, None] - means) ** 2).sum(axis=0) / weights
        )
    p = fit.candidate.n_free_params(fit.params.n_features)
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    info = {
        "data_hash": data.data_hash(),
        "config": config.model_dump(mode="json"),
        "candidate": {**fit.candidate.model_dump(mode="json"), "n_params": p},
        "n_rows": len(data.matrix),
        "loglik": fit.loglik,
        "bic": selection_rules.bic(fit.loglik, p, len(data.matrix)),
        "restarts": {
            "near_best": fit.choice.near_best,
            "stable": fit.choice.stable,
            "iterations": fit.iterations,
            "failed": [{"index": f.index, "seed": f.seed, "error": f.error} for f in fit.failed],
        },
        "sanity": {
            "passed": sanity.passed,
            "reasons": sanity.reasons,
            "occupancy": sanity.occupancy,
            "durations": [_finite_or_none(d) for d in sanity.durations],
            "condition": sanity.condition,
        },
        "states": [
            {
                "state": k,
                "label": label.label.value,
                "v": label.v,
                "d": label.d,
                "mean_return": float(means[k] * BARS_PER_YEAR),
                "volatility": float(deviations[k] * np.sqrt(BARS_PER_YEAR)),
                "duration_bars": _finite_or_none(sanity.durations[k]),
                "duration_sessions": _finite_or_none(
                    sanity.durations[k] / selection_rules.BARS_PER_SESSION
                ),
                "occupancy": float(sanity.occupancy[k]),
            }
            for k, label in enumerate(labels.states)
        ],
        "selection": selection_report,
        "provenance": provenance,
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "hmmlearn")},
        },
        "blas": {"name": blas["name"], "version": blas["version"]},
    }
    try:
        # Validate before pydantic serializes arbitrary caller provenance.
        canonical_dumps(info)
        record = TrainingRecord(
            base_seed=config.base_seed,
            seeds=fit.seeds,
            restart_logliks=fit.logliks,
            best=fit.choice.best,
            info=info,
        )
    except (CanonicalJSONError, ValueError) as exc:
        raise ModelError("training report has no canonical JSON form") from exc
    model = HmmModel(
        fit.params, scaler, data.fit_start, data.fit_end, data.feature_config_hash, labels, record
    )
    return TrainResult(model, sanity, selection)
