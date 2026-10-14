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
5. A restart *fails* when it raises one of the expected fit failures, ModelError or ValueError
   (hmmlearn's parameter checks, numpy's LinAlgError, `GaussianParams` refusing the
   parameters, the forward pass). Any other exception (TypeError, AssertionError, ...) is a
   bug and propagates. Seeds can't fail: TrainConfig keeps every seed in [0, SEED_MAX]. A
   failed restart is left out of the record's `seeds` and `restart_logliks` and listed in
   `info` instead (index, seed and the exception's type name). A *fit* whose restarts all
   fail is a failed fit: `fit_candidate` raises ModelError for it, and selection sets its
   candidate aside.
6. The winner is `select.choose_restart` over the recorded log-likelihoods in restart order:
   the highest, a tie going to the lowest index, and a structlog warning "hmm optimum
   unstable" (naming the fit, "full" or "validation") when fewer than 3 restarts (the winner
   included) land within 0.1% of the best, measured on the standardized rows' log-likelihood
   (`jacobian` = n · -Σ log std of the fit's scaler), so the features' units can't change it.

**Parallelism** (ARCHITECTURE §7.1, §14): `workers=1` runs the restarts one after another in
the calling process. Otherwise they run in a `concurrent.futures.ProcessPoolExecutor` with the
"spawn" context and `workers` processes (None: `os.cpu_count()`), and results are collected
in restart order. The pool's workers are started with OMP_NUM_THREADS, OPENBLAS_NUM_THREADS,
MKL_NUM_THREADS and VECLIB_MAXIMUM_THREADS set to 1 in their environment, so each worker's BLAS
is single-threaded from before it imports numpy: Apple's Accelerate reads only
VECLIB_MAXIMUM_THREADS, at load, and threadpoolctl can't limit it. The caller's environment is
restored once the pool closes. Each worker receives the blocks of rows and the config once (the
pool's initializer); a task is just (block index, restart index). Every fit, in a pool or
serial, also runs inside `threadpoolctl.threadpool_limits(limits=1)` (OpenMP and the BLAS
libraries threadpoolctl can see). The serial path runs in the caller's process, whose BLAS
threads were fixed when it imported numpy: ARCHITECTURE §14 has the engine, backtest and
harness set the same variables before that, and then the pool and the serial path give the
same bits and the same artifact hash. `select_and_train` sends every restart of every
candidate, both fits, through one pool. The worker count is never recorded: it can't change
the model.

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

A candidate with a failed fit (either one) is *set aside*: it gets no score, takes no part in
the selection, and is listed in the report with its failed restarts. Selection raises
ModelError only when every candidate is set aside or rejected. `select.select` picks among the
rest. "Refit the winner on the full window" is its full-window fit, already done: same rows,
scaler, seeds and config, so `train_candidate(data, chosen, ...)` gives the same parameters bit
for bit.

**The validation record** (spec §5, ARCHITECTURE §7.2, §7.5): every model, a refit's too,
records its own candidate's validation fit: `train_candidate` runs the same two fits for its one
candidate. It holds the validation block's l_t distribution and the drift baseline
(`select.drift_threshold`), the 1st percentile of the block's 35-bar rolling mean of l_t, which
the live engine compares its own rolling mean with. Both are densities of the raw rows, so the
validation fit's scaler and the live model's don't need reconciling.

**The model** is the chosen fit's parameters, the full-window scaler (`ScalerParams.from_scaler`),
the fit window, the labels (`hmm.labels.label_states(data.returns, trace.state_probs,
thresholds)` with the winner's filtered probabilities over the window; a state with no filtered
mass at all has no label, and the fit is then refused as the sanity reject it is) and a
TrainingRecord:
`base_seed`, the recorded `seeds` and `restart_logliks` of the full-window fit, `best` from
`choose_restart`, and `info` (`TrainResult` lists its keys). Nothing in it depends on wall time,
paths or the worker count, so the same data, seeds and config on the same platform give the
same artifact hash.

Times are UTC. "Now" is never read (K6).
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import platform
import re
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from typing import Any, Final, Self

import numpy as np
import scipy
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
from quant.hmm.select import (
    Candidate,
    CandidateScore,
    FitName,
    RestartChoice,
    SanityReport,
    Selection,
    ValidationScore,
)

# numpy's seeds are 32-bit: every restart's seed base_seed + i must be at most this.
SEED_MAX: Final = 2**32 - 1
# The thread counts a pool worker is started with (ARCHITECTURE §14).
BLAS_THREAD_VARIABLES: Final = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
# What a bad local optimum raises. numpy's LinAlgError is a ValueError. Anything else is a bug.
_FIT_FAILURES: Final = (ModelError, ValueError)


class TrainConfig(StrictModel):
    """Everything that defines a training run, recorded in `info["config"]` (so in the hash).

    - `base_seed`: restart i uses `base_seed + i`, at most SEED_MAX; the bootstrap uses
      `base_seed`;
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
        """Return self when base_seed ≥ 0, n_restarts ≥ 1, base_seed + n_restarts - 1 ≤
        SEED_MAX, `candidates` is non-empty with no duplicate and every K ≥ 2, n_iter ≥ 1,
        tol > 0, min_covar > 0, bootstrap_draws ≥ 2 and validation_months ≥ 1. Otherwise raise
        ValueError naming the field (pydantic reports a ValidationError)."""
        for name, minimum in (
            ("base_seed", 0),
            ("n_restarts", 1),
            ("n_iter", 1),
            ("bootstrap_draws", 2),
            ("validation_months", 1),
        ):
            if getattr(self, name) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.base_seed + self.n_restarts - 1 > SEED_MAX:
            raise ValueError(f"base_seed + n_restarts - 1 must be at most {SEED_MAX}")
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
class FailedFit:
    """A fit whose restarts all failed: its candidate, which fit it was, and the failures."""

    candidate: Candidate
    fit: FitName
    failed: tuple[FailedRestart, ...]


@dataclass(frozen=True, eq=False)
class CandidateFit:
    """One candidate trained on one block of rows (`fit_candidate`).

    - `params`: the winning restart's parameters;
    - `trace`: our forward pass over the rows under `params` (`forward_pass`, from π);
    - `seeds`, `logliks`, `iterations`: one entry per recorded (non-failed) restart, in restart
      order: its seed, its training log-likelihood (Σ l_t), and EM's iteration count
      (`monitor_.iter`);
    - `choice`: `choose_restart(logliks, jacobian=n·J, fit=...)`; `choice.best` indexes the
      recorded tuples;
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
    The fit is named "full" in the instability warning.

    Raises ModelError when `columns` isn't the scaler's, a row isn't finite, n < K, or every
    restart fails, and when workers < 1.
    """
    return _required(
        _fit_blocks((_Block(rows, scaler, candidate, "full"),), columns, config, workers)[0]
    )


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
      - "validation": the model's own candidate's validation fit, on both paths: {"start" (ISO
        UTC), "rows", "sessions", "draws", "seed", "mean", "se", "near_best", "stable" (its
        restarts), "quantiles": {"0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99" →
        numpy.quantile of the block's l_t, linear}, "drift": {"window" (DRIFT_WINDOW),
        "quantile" (DRIFT_QUANTILE), "threshold" (`select.drift_threshold` of the block's
        l_t)}};
      - "selection": null from `train_candidate`; from `select_and_train` {"rule", "chosen",
        "best_bic", "best_validation", "close" (candidates as {"n_states",
        "covariance_type"}), "se", "table": one row per scored candidate in grid order
        {"n_states", "covariance_type", "n_params", "loglik", "bic", "validation_mean",
        "validation_se", "near_best", "stable", "validation_near_best", "validation_stable",
        "sanity_passed", "sanity_reasons"}, "set_aside": one entry per candidate set aside, in
        grid order, {"n_states", "covariance_type", "failed": {fit name: [{"index", "seed",
        "error"}]} for each failed fit}};
      - "provenance": the caller's `provenance` mapping, verbatim (code commit, lockfile hash);
      - "versions": {"python", "numpy", "scipy", "scikit-learn", "hmmlearn"}, version strings;
      - "platform": {"system", "release", "machine", "platform"} from the `platform` module;
      - "blas": the BLAS and LAPACK builds numpy and scipy were compiled against, {"numpy",
        "scipy"} → {"blas", "lapack"} → {"name", "version", "detection method", "openblas
        configuration"}, each a string from `show_config(mode="dicts")` (no paths). Accelerate
        reports no version: the OS release in "platform" identifies it.
    - `sanity`: the model's SanityReport. `train_candidate` returns a model that fails it
      rather than raising, so a refit job can store and not activate it (spec §5), except when
      a state has no filtered mass at all: such a state has no v, d or label, and the fit
      raises ModelError "sanity reject: <reasons>" (ARCHITECTURE §7.3: a sanity reject keeps
      the current model and alerts);
    - `selection`: the Selection, or None from `train_candidate`.
    """

    model: HmmModel
    sanity: SanityReport
    selection: Selection | None


@dataclass(frozen=True, eq=False)
class _Split:
    """The validation split of a fit window: the block's start, its row mask, and the scaler
    fit on the rows before it."""

    start: datetime
    block: np.ndarray[Any, np.dtype[np.bool_]]
    scaler: ScalerParams


def _split(data: TrainingData, config: TrainConfig) -> _Split:
    start = selection_rules.validation_start(data.fit_end, config.validation_months)
    block = data.matrix.available_at >= to_us(start)
    if any(len(np.unique(data.sessions[mask])) < 2 for mask in (block, ~block)):
        raise ModelError("training and validation blocks must each span at least two sessions")
    if int(block.sum()) < selection_rules.DRIFT_WINDOW:
        raise ModelError("the validation block is shorter than the drift window")
    scaler = ScalerParams.from_scaler(Scaler.fit(data.matrix, start=data.fit_start, end=start))
    return _Split(start, block, scaler)


def _full_scaler(data: TrainingData) -> ScalerParams:
    return ScalerParams.from_scaler(Scaler.fit(data.matrix, start=data.fit_start, end=data.fit_end))


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
    docstring, its sanity report and labels, the candidate's validation fit and score (for the
    validation record), and the model. `config.candidates` is not consulted. Raises
    ModelError as `fit_candidate` and `label_states` do, when either fit fails, when the
    validation block or the rows before it span fewer than 2 sessions or the block has fewer
    than DRIFT_WINDOW rows, for a state with no filtered mass (a sanity reject), or when
    `provenance` has no canonical JSON form."""
    scaler, split = _full_scaler(data), _split(data, config)
    rows = data.matrix.values
    train_rows = rows[~split.block]
    full, validation = (
        _required(fit)
        for fit in _fit_blocks(
            (
                _Block(rows, scaler, candidate, "full"),
                _Block(train_rows, split.scaler, candidate, "validation"),
            ),
            data.matrix.columns,
            config,
            workers,
        )
    )
    with threadpool_limits(limits=1):
        score = _validation_score(data, split, validation, config)
        report = _validation_report(score, validation, split, data, config)
        return _build_result(data, scaler, full, thresholds, config, provenance, report, None, None)


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
    than 2 sessions or the block has fewer than DRIFT_WINDOW rows, when every candidate is set
    aside or fails its sanity check (`select.select`), or as `train_candidate` does."""
    scaler, split = _full_scaler(data), _split(data, config)
    rows = data.matrix.values
    train_rows = rows[~split.block]
    blocks = tuple(
        block
        for c in config.candidates
        for block in (
            _Block(rows, scaler, c, "full"),
            _Block(train_rows, split.scaler, c, "validation"),
        )
    )
    fits = _fit_blocks(blocks, data.matrix.columns, config, workers)
    scores: list[CandidateScore] = []
    kept: list[tuple[CandidateFit, CandidateFit]] = []
    set_aside: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for i, candidate in enumerate(config.candidates):
            full, validation = fits[2 * i : 2 * i + 2]
            if not isinstance(full, CandidateFit) or not isinstance(validation, CandidateFit):
                set_aside.append(_set_aside_entry(candidate, (full, validation)))
                continue
            p = candidate.n_free_params(rows.shape[1])
            scores.append(
                CandidateScore(
                    candidate,
                    p,
                    full.loglik,
                    selection_rules.bic(full.loglik, p, len(rows)),
                    _validation_score(data, split, validation, config),
                    selection_rules.sanity_check(full.params, full.trace.state_probs),
                )
            )
            kept.append((full, validation))
        if not scores:
            raise ModelError("no candidate survives: every one has a fit whose restarts all failed")
        selection = selection_rules.select(tuple(scores), rows.shape[1])
        chosen = next(i for i, s in enumerate(scores) if s.candidate == selection.chosen)
        full, validation = kept[chosen]
        return _build_result(
            data,
            scaler,
            full,
            thresholds,
            config,
            provenance,
            _validation_report(scores[chosen].validation, validation, split, data, config),
            selection,
            _selection_report(selection, tuple(kept), set_aside),
        )


@dataclass(frozen=True, eq=False)
class _Block:
    """One fit: its raw rows, the scaler that standardizes them, the candidate, its name."""

    rows: Floats
    scaler: ScalerParams
    candidate: Candidate
    fit: FitName


@dataclass(frozen=True)
class _RestartResult:
    params: GaussianParams
    loglik: float
    iterations: int


# Set in each pool worker, once, by the pool's initializer.
_WORKER: tuple[tuple[_Block, ...], TrainConfig] | None = None


def _init_worker(blocks: tuple[_Block, ...], config: TrainConfig) -> None:
    global _WORKER
    _WORKER = (blocks, config)


def _pooled_restart(key: tuple[int, int]) -> _RestartResult | FailedRestart:
    """Restart key[1] of block key[0], from the blocks the initializer gave this worker."""
    if _WORKER is None:
        raise ModelError("the training worker was not initialized")
    blocks, config = _WORKER
    return _restart(blocks[key[0]], config, key[1])


def _restart(block: _Block, config: TrainConfig, index: int) -> _RestartResult | FailedRestart:
    """The same fitting and scoring path in the caller and in spawned workers."""
    seed = config.base_seed + index
    try:
        with threadpool_limits(limits=1):
            z = block.scaler.standardize(block.rows, columns=block.scaler.columns)
            estimator = GaussianHMM(
                n_components=block.candidate.n_states,
                covariance_type=block.candidate.covariance_type,
                min_covar=config.min_covar,
                n_iter=config.n_iter,
                tol=config.tol,
                random_state=seed,
                implementation="log",
                init_params="stmc",
                params="stmc",
            )
            estimator.fit(z)
            covariance = np.asarray(estimator.covars_, dtype=np.float64)
            params = GaussianParams(
                block.candidate.covariance_type,
                np.asarray(estimator.startprob_, dtype=np.float64),
                np.asarray(estimator.transmat_, dtype=np.float64),
                np.asarray(estimator.means_, dtype=np.float64),
                0.5 * (covariance + covariance.transpose(0, 2, 1)),
            )
            trace = forward_pass(params, block.scaler, block.rows, columns=block.scaler.columns)
            loglik = float(np.sum(trace.loglik))
            if not math.isfinite(loglik):
                raise ModelError("training log-likelihood overflows float64")
            return _RestartResult(params, loglik, int(estimator.monitor_.iter))
    except _FIT_FAILURES as exc:
        # A bad local optimum must not discard the usable restarts; a bug still raises.
        return FailedRestart(index, seed, type(exc).__name__)


@contextmanager
def _single_threaded_children() -> Iterator[None]:
    """Set BLAS_THREAD_VARIABLES to "1" in os.environ, which processes started now inherit,
    and restore the caller's values (or their absence) afterwards."""
    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(BLAS_THREAD_VARIABLES, "1"))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _fit_blocks(
    blocks: tuple[_Block, ...],
    columns: tuple[str, ...],
    config: TrainConfig,
    workers: int | None,
) -> tuple[CandidateFit | FailedFit, ...]:
    """Run every restart of every block, in one pool, and reduce each block in seed order."""
    if workers is not None and workers < 1:
        raise ModelError("workers must be positive")
    for block in blocks:
        block.scaler.standardize(block.rows, columns=columns)
        if block.rows.ndim != 2 or len(block.rows) < block.candidate.n_states:
            raise ModelError("a fit needs at least K rows")
    keys = [(b, i) for b in range(len(blocks)) for i in range(config.n_restarts)]
    if workers == 1:
        results = [_restart(blocks[b], config, i) for b, i in keys]
    else:
        with (
            _single_threaded_children(),
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(blocks, config),
            ) as pool,
        ):
            results = list(pool.map(_pooled_restart, keys))
    fits: list[CandidateFit | FailedFit] = []
    with threadpool_limits(limits=1):
        for block_index, block in enumerate(blocks):
            offset = block_index * config.n_restarts
            batch = results[offset : offset + config.n_restarts]
            good = [
                (i, result) for i, result in enumerate(batch) if isinstance(result, _RestartResult)
            ]
            failed = tuple(result for result in batch if isinstance(result, FailedRestart))
            if not good:
                fits.append(FailedFit(block.candidate, block.fit, failed))
                continue
            logliks = tuple(result.loglik for _, result in good)
            choice = selection_rules.choose_restart(
                logliks,
                candidate=block.candidate,
                jacobian=len(block.rows) * -float(np.log(block.scaler.std).sum()),
                fit=block.fit,
            )
            params = good[choice.best][1].params
            # Reconstruct to retain array immutability after process transport.
            params = GaussianParams(
                params.covariance_type,
                params.startprob,
                params.transmat,
                params.means,
                params.covars,
            )
            trace = forward_pass(params, block.scaler, block.rows, columns=columns)
            fits.append(
                CandidateFit(
                    block.candidate,
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


def _required(fit: CandidateFit | FailedFit) -> CandidateFit:
    if isinstance(fit, FailedFit):
        raise ModelError(f"all restarts failed for {fit.candidate} ({fit.fit}): {fit.failed}")
    return fit


def _failures(failed: tuple[FailedRestart, ...]) -> list[dict[str, Any]]:
    return [{"index": f.index, "seed": f.seed, "error": f.error} for f in failed]


def _set_aside_entry(
    candidate: Candidate, fits: tuple[CandidateFit | FailedFit, ...]
) -> dict[str, Any]:
    return {
        **candidate.model_dump(mode="json"),
        "failed": {f.fit: _failures(f.failed) for f in fits if isinstance(f, FailedFit)},
    }


def _validation_score(
    data: TrainingData, split: _Split, fit: CandidateFit, config: TrainConfig
) -> ValidationScore:
    rows = data.matrix.values
    return selection_rules.validation_score(
        fit.params,
        split.scaler,
        rows[~split.block],
        rows[split.block],
        data.sessions[split.block],
        columns=data.matrix.columns,
        draws=config.bootstrap_draws,
        seed=config.base_seed,
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _validation_report(
    score: ValidationScore,
    fit: CandidateFit,
    split: _Split,
    data: TrainingData,
    config: TrainConfig,
) -> dict[str, Any]:
    sessions = data.sessions[split.block]
    return {
        "start": split.start,
        "rows": len(sessions),
        "sessions": len(np.unique(sessions)),
        "draws": config.bootstrap_draws,
        "seed": config.base_seed,
        "mean": score.mean,
        "se": score.se,
        "near_best": fit.choice.near_best,
        "stable": fit.choice.stable,
        "quantiles": {
            str(q): float(np.quantile(score.loglik, q))
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
        "drift": {
            "window": selection_rules.DRIFT_WINDOW,
            "quantile": selection_rules.DRIFT_QUANTILE,
            "threshold": selection_rules.drift_threshold(score.loglik),
        },
    }


def _selection_report(
    selection: Selection,
    fits: tuple[tuple[CandidateFit, CandidateFit], ...],
    set_aside: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "rule": selection.rule,
        "chosen": selection.chosen.model_dump(mode="json"),
        "best_bic": selection.best_bic.model_dump(mode="json"),
        "best_validation": selection.best_validation.model_dump(mode="json"),
        "close": [c.model_dump(mode="json") for c in selection.close],
        "se": selection.se,
        "table": [
            {
                **score.candidate.model_dump(mode="json"),
                "n_params": score.n_params,
                "loglik": score.loglik,
                "bic": score.bic,
                "validation_mean": score.validation.mean,
                "validation_se": score.validation.se,
                "near_best": full.choice.near_best,
                "stable": full.choice.stable,
                "validation_near_best": validation.choice.near_best,
                "validation_stable": validation.choice.stable,
                "sanity_passed": score.sanity.passed,
                "sanity_reasons": score.sanity.reasons,
            }
            for score, (full, validation) in zip(selection.scores, fits, strict=True)
        ],
        "set_aside": set_aside,
    }


def _blas_builds() -> dict[str, Any]:
    """numpy's and scipy's BLAS and LAPACK builds, without paths (TrainResult)."""
    fields = ("name", "version", "detection method", "openblas configuration")
    builds: dict[str, Any] = {}
    for library in (np, scipy):
        built = library.show_config(mode="dicts")["Build Dependencies"]
        builds[library.__name__] = {
            api: {field: str(built[api].get(field)) for field in fields}
            for api in ("blas", "lapack")
        }
    return builds


def _build_result(
    data: TrainingData,
    scaler: ScalerParams,
    fit: CandidateFit,
    thresholds: LabelThresholds,
    config: TrainConfig,
    provenance: Mapping[str, Any],
    validation_report: dict[str, Any],
    selection: Selection | None,
    selection_report: dict[str, Any] | None,
) -> TrainResult:
    with threadpool_limits(limits=1):
        sanity = selection_rules.sanity_check(fit.params, fit.trace.state_probs)
        weights = fit.trace.state_probs.sum(axis=0)
        if (weights <= 0).any():
            # No filtered mass: no v, d or label. Occupancy 0 is already among the reasons.
            raise ModelError("sanity reject: " + "; ".join(sanity.reasons))
        labels = label_states(data.returns, fit.trace.state_probs, thresholds)
        means = (fit.trace.state_probs * data.returns[:, None]).sum(axis=0) / weights
        deviations = np.sqrt(
            (fit.trace.state_probs * (data.returns[:, None] - means) ** 2).sum(axis=0) / weights
        )
    p = fit.candidate.n_free_params(fit.params.n_features)
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
            "failed": _failures(fit.failed),
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
        "validation": validation_report,
        "selection": selection_report,
        "provenance": provenance,
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "hmmlearn")},
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "blas": _blas_builds(),
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
