"""Model selection and the sanity rejects (spec §4): pure float64 numpy over fitted candidates.

Nothing here imports hmmlearn (only quant.hmm.train does, I2) or eval. `quant.hmm.train` fits
each candidate and calls these functions; they are separate so each rule can be tested on
hand-built numbers.

**Candidates** are (K, covariance type) pairs. A candidate's complexity is its number of free
parameters p (`n_free_params`) for d features:

    p = (K - 1)            start probabilities π
      + K (K - 1)          transition matrix A, rows summing to 1
      + K d                means
      + K d                covariances, diag
        or K d (d + 1) / 2 covariances, full (symmetric)

The scaler's 2d values are the same for every candidate and are not counted.

**"Simplest"** orders candidates by the key (p, K, covariance type with diag before full): the
fewest free parameters first, a tie going to fewer states, then to diag (`simplicity_key`).

**BIC** = -2 LL + p ln n, where LL is the candidate's training log-likelihood on the whole fit
window (Σ l_t from our forward pass, `hmm.filter.forward_pass`, the scaler's Jacobian
included) and n the window's row count. Lower is better.

**Validation** (spec §4): the validation block is the rows of the last `validation_months`
calendar months of the fit window (`validation_start`); the candidate is fit again on the rows
before it, and scored by the mean one-step predictive log-likelihood of the block's rows,
continuing that fit's forward pass from the last training row (`validation_score`). Higher is
better.

**Standard errors** come from a session-block bootstrap of that mean (`bootstrap_means`): the
blocks are the validation block's sessions, one block per session, every bar of it.

**Close** (spec §4): a candidate is close on BIC when BIC ≤ best BIC + BIC_CLOSE, and close on
validation when its mean ≥ best mean - SE, SE being the bootstrap standard error of the best
candidate's mean. Both comparisons are inclusive. The pick is the simplest candidate close on
both; if none is, the simpler of the two winners (`select`). The bests are taken over the
candidates that pass the sanity rejects only; rejected candidates are never picked.

**Sanity rejects** (spec §4, `sanity_check`): a model is rejected when any state has an
occupancy under OCCUPANCY_MIN, an expected duration under DURATION_MIN bars, or a covariance
condition number above CONDITION_MAX. All three thresholds are strict: a value exactly at one
passes.
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

import numpy as np
import structlog

from quant.core.clock import ensure_utc
from quant.core.errors import ModelError
from quant.core.model import StrictModel
from quant.data.frames import Floats, Ints
from quant.hmm.filter import forward_pass
from quant.hmm.labels import BARS_PER_YEAR
from quant.hmm.model import CovarianceType, GaussianParams, ScalerParams

# Spec §4: within 10 BIC points of the best is close.
BIC_CLOSE: Final = 10.0
# Spec §4's sanity rejects.
OCCUPANCY_MIN: Final = 0.02
DURATION_MIN: Final = 2.0
CONDITION_MAX: Final = 1e6
# Spec §4's instability warning: fewer than NEAR_BEST_MIN restarts (the winner included) whose
# training log-likelihood is within NEAR_BEST_RTOL of the best's, relative to |best|.
NEAR_BEST_RTOL: Final = 1e-3
NEAR_BEST_MIN: Final = 3
# Bars per session, for durations in sessions (spec §4's report): the labels' annualization.
BARS_PER_SESSION: Final = BARS_PER_YEAR // 252

SelectionRule = Literal["close_on_both", "simpler_winner"]


class Candidate(StrictModel):
    """One point of the selection grid: K hidden states, and the covariance type."""

    n_states: int
    covariance_type: CovarianceType

    def n_free_params(self, n_features: int) -> int:
        """`n_free_params(self.n_states, n_features, self.covariance_type)`."""
        return n_free_params(self.n_states, n_features, self.covariance_type)


def n_free_params(n_states: int, n_features: int, covariance_type: CovarianceType) -> int:
    """p of the module docstring: (K-1) + K(K-1) + Kd + (Kd for "diag", Kd(d+1)/2 for "full").
    K = 3, d = 2 gives 20 for diag and 23 for full. Raises ModelError for K < 1 or d < 1."""
    if n_states < 1 or n_features < 1:
        raise ModelError("parameter counts need positive states and features")
    k, d = n_states, n_features
    covariance = k * d if covariance_type == "diag" else k * d * (d + 1) // 2
    return (k - 1) + k * (k - 1) + k * d + covariance


def simplicity_key(candidate: Candidate, n_features: int) -> tuple[int, int, int]:
    """(p, K, 0 for diag or 1 for full): sorting by it puts the simplest candidate first."""
    return (
        candidate.n_free_params(n_features),
        candidate.n_states,
        int(candidate.covariance_type == "full"),
    )


def bic(loglik: float, n_params: int, n_rows: int) -> float:
    """-2 * loglik + n_params * math.log(n_rows), exactly that expression. Raises ModelError
    when loglik isn't finite or n_rows < 1."""
    if not math.isfinite(loglik) or n_rows < 1:
        raise ModelError("BIC needs a finite log-likelihood and positive row count")
    return -2 * loglik + n_params * math.log(n_rows)


@dataclass(frozen=True, eq=False)
class RestartChoice:
    """Which restart wins (spec §4, ARCHITECTURE §7.1).

    - `best`: the index of the highest log-likelihood, a tie going to the lowest index
      (`int(numpy.argmax(logliks))`);
    - `near_best`: how many restarts, the winner included, have |LL_i - LL_best| ≤
      NEAR_BEST_RTOL * |LL_best| (inclusive);
    - `stable`: near_best ≥ NEAR_BEST_MIN.
    """

    best: int
    near_best: int
    stable: bool


def choose_restart(logliks: tuple[float, ...], *, candidate: Candidate) -> RestartChoice:
    """The RestartChoice for one candidate's restart log-likelihoods, in restart order.

    When the result isn't stable, log one structlog warning, event "hmm optimum unstable",
    with the fields n_states, covariance_type, near_best, restarts (len(logliks)) and
    best_loglik. A stable result logs nothing. Raises ModelError for an empty tuple or a value
    that isn't finite.
    """
    if not logliks or not np.isfinite(logliks).all():
        raise ModelError("restarts need finite log-likelihoods")
    best = int(np.argmax(logliks))
    value = logliks[best]
    near = sum(abs(ll - value) <= NEAR_BEST_RTOL * abs(value) for ll in logliks)
    stable = near >= NEAR_BEST_MIN
    if not stable:
        structlog.get_logger().warning(
            "hmm optimum unstable",
            n_states=candidate.n_states,
            covariance_type=candidate.covariance_type,
            near_best=near,
            restarts=len(logliks),
            best_loglik=value,
        )
    return RestartChoice(best, near, stable)


def validation_start(fit_end: datetime, months: int) -> datetime:
    """The start of the validation block: `fit_end` moved back `months` calendar months, the
    same day and time of day in UTC, the day clamped to the target month's last day (2020-03-31
    back one month is 2020-02-29). Rows with available_at ≥ it are the block. A naive
    `fit_end` raises ClockError; months < 1 raises ModelError."""
    end = ensure_utc(fit_end)
    if months < 1:
        raise ModelError("validation months must be positive")
    year, month = divmod(end.year * 12 + end.month - 1 - months, 12)
    month += 1
    if year < 1:
        raise ModelError("validation start precedes the supported calendar")
    return end.replace(
        year=year, month=month, day=min(end.day, calendar.monthrange(year, month)[1])
    )


@dataclass(frozen=True, eq=False)
class ValidationScore:
    """A candidate's validation result.

    - `loglik` (m,): l_t of each validation row, from `forward_pass(params, scaler,
      validation_rows, columns=..., previous=train_trace.log_probs[-1])`, i.e. bit for bit the
      tail of one forward pass over the training rows followed by the validation rows;
    - `mean`: float(numpy.mean(loglik));
    - `se`: the bootstrap standard error of `mean` (`bootstrap_se`).
    """

    loglik: Floats
    mean: float
    se: float


def validation_score(
    params: GaussianParams,
    scaler: ScalerParams,
    train_rows: Floats,
    validation_rows: Floats,
    validation_sessions: Ints,
    *,
    columns: tuple[str, ...],
    draws: int,
    seed: int,
) -> ValidationScore:
    """Score a model fit on `train_rows` on the validation block that follows them: run
    `forward_pass` over `train_rows`, continue it over `validation_rows` with `previous` the
    last training row's log p, and bootstrap the mean (`bootstrap_se(loglik,
    validation_sessions, draws=draws, seed=seed)`). `columns` names both blocks' columns.
    Raises ModelError as forward_pass does, and as bootstrap_se does."""
    train = forward_pass(params, scaler, train_rows, columns=columns)
    trace = forward_pass(
        params, scaler, validation_rows, columns=columns, previous=train.log_probs[-1]
    )
    return ValidationScore(
        trace.loglik,
        float(np.mean(trace.loglik)),
        bootstrap_se(trace.loglik, validation_sessions, draws=draws, seed=seed),
    )


def bootstrap_means(loglik: Floats, sessions: Ints, *, draws: int, seed: int) -> Floats:
    """The session-block bootstrap distribution of mean(loglik), shape (draws,).

    `sessions` (m,) is each row's session id (int64, nondecreasing); the blocks are the runs of
    equal ids, S of them, in order. With picks = numpy.random.default_rng(seed).integers(0, S,
    size=(draws, S)), draw b's value is Σ_{j in picks[b]} sum_j / Σ_{j in picks[b]} count_j,
    sum_j and count_j being block j's total l_t and bar count (a block picked twice counts
    twice). The same seed and sessions give the same picks for every candidate, so all are
    resampled on the same draws.

    Raises ModelError when the shapes differ, `sessions` decreases, S < 2, draws < 2, or a
    value isn't finite.
    """
    if (
        not isinstance(loglik, np.ndarray)
        or loglik.dtype != np.float64
        or loglik.ndim != 1
        or not np.isfinite(loglik).all()
        or not isinstance(sessions, np.ndarray)
        or sessions.dtype != np.int64
        or sessions.shape != loglik.shape
        or len(sessions) < 2
        or (sessions[1:] < sessions[:-1]).any()
        or draws < 2
        or seed < 0
    ):
        raise ModelError("bootstrap needs finite log-likelihoods and ordered session blocks")
    starts = np.r_[0, np.flatnonzero(sessions[1:] != sessions[:-1]) + 1]
    if len(starts) < 2:
        raise ModelError("bootstrap needs at least two sessions")
    totals = np.add.reduceat(loglik, starts)
    counts = np.diff(np.r_[starts, len(loglik)])
    picks = np.random.default_rng(seed).integers(0, len(starts), size=(draws, len(starts)))
    result: Floats = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return result


def bootstrap_se(loglik: Floats, sessions: Ints, *, draws: int, seed: int) -> float:
    """float(numpy.std(bootstrap_means(...), ddof=1)): the standard error of mean(loglik)."""
    return float(np.std(bootstrap_means(loglik, sessions, draws=draws, seed=seed), ddof=1))


@dataclass(frozen=True, eq=False)
class SanityReport:
    """Spec §4's sanity checks of one model on its fit window, one entry per state.

    - `occupancy` (K,): mean filtered probability, Σ_t p_tk / n;
    - `durations` (K,): expected duration 1 / (1 - a_kk) in bars, +inf where a_kk = 1;
    - `condition` (K,): `numpy.linalg.cond(covars[k])`, the 2-norm condition number of the
      stored (standardized) covariance;
    - `reasons`: one string per failing check and state, in state order and, within a state,
      occupancy, duration, condition: "state {k}: occupancy {x} < 0.02", "state {k}: duration
      {x} < 2 bars" or "state {k}: condition number {x} > 1e6";
    - `passed`: no reasons.
    """

    occupancy: Floats
    durations: Floats
    condition: Floats
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons


def sanity_check(params: GaussianParams, state_probs: Floats) -> SanityReport:
    """The SanityReport of `params` with `state_probs` (n, K) its filtered probabilities over
    the fit window (`forward_pass(...).state_probs`). A state fails when occupancy <
    OCCUPANCY_MIN, duration < DURATION_MIN or condition > CONDITION_MAX; a value exactly at a
    threshold passes. Raises ModelError when `state_probs` isn't a finite float64 (n, K) array
    with n ≥ 1."""
    if (
        not isinstance(state_probs, np.ndarray)
        or state_probs.dtype != np.float64
        or state_probs.ndim != 2
        or state_probs.shape[1] != params.n_states
        or len(state_probs) < 1
        or not np.isfinite(state_probs).all()
    ):
        raise ModelError("sanity checks need finite float64 probabilities of shape (n, K)")
    occupancy = state_probs.mean(axis=0)
    with np.errstate(divide="ignore"):
        durations = 1 / (1 - np.diag(params.transmat))
    condition = np.array([np.linalg.cond(c) for c in params.covars], dtype=np.float64)
    reasons = []
    for k in range(params.n_states):
        if occupancy[k] < OCCUPANCY_MIN:
            reasons.append(f"state {k}: occupancy {occupancy[k]} < 0.02")
        if durations[k] < DURATION_MIN:
            reasons.append(f"state {k}: duration {durations[k]} < 2 bars")
        if condition[k] > CONDITION_MAX:
            reasons.append(f"state {k}: condition number {condition[k]} > 1e6")
    return SanityReport(occupancy, durations, condition, tuple(reasons))


@dataclass(frozen=True, eq=False)
class CandidateScore:
    """One row of spec §4's BIC and validation table.

    - `candidate`, `n_params` (p for the data's d);
    - `loglik`: the full-window fit's training log-likelihood, and `bic` from it;
    - `validation`: the validation fit's ValidationScore;
    - `sanity`: the full-window fit's SanityReport.
    """

    candidate: Candidate
    n_params: int
    loglik: float
    bic: float
    validation: ValidationScore
    sanity: SanityReport


@dataclass(frozen=True, eq=False)
class Selection:
    """The outcome of spec §4's rule over `scores` (in grid order).

    - `best_bic`: the passing candidate with the lowest BIC, a tie going to the simplest;
    - `best_validation`: the passing candidate with the highest validation mean, a tie going to
      the simplest;
    - `se`: best_validation's bootstrap standard error, the width of "close";
    - `close`: the passing candidates close on both scores, simplest first;
    - `chosen`: close[0], or the simpler of best_bic and best_validation when `close` is empty;
    - `rule`: "close_on_both" or "simpler_winner", whichever produced `chosen`.
    """

    scores: tuple[CandidateScore, ...]
    best_bic: Candidate
    best_validation: Candidate
    se: float
    close: tuple[Candidate, ...]
    chosen: Candidate
    rule: SelectionRule


def select(scores: tuple[CandidateScore, ...], n_features: int) -> Selection:
    """Apply the module docstring's rule. "Simplest" is `simplicity_key(..., n_features)`.
    Candidates whose sanity check failed take no part. Raises ModelError when `scores` is empty
    or every candidate failed its sanity check, or two scores share a
    candidate."""
    if len({s.candidate for s in scores}) != len(scores):
        raise ModelError("selection candidates must be distinct")
    passing = [s for s in scores if s.sanity.passed]
    if not passing:
        raise ModelError("selection needs a candidate that passes sanity checks")
    best_bic = min(passing, key=lambda s: (s.bic, simplicity_key(s.candidate, n_features)))
    best_val = min(
        passing, key=lambda s: (-s.validation.mean, simplicity_key(s.candidate, n_features))
    )
    se = best_val.validation.se
    close = tuple(
        sorted(
            (
                s.candidate
                for s in passing
                if s.bic <= best_bic.bic + BIC_CLOSE
                and s.validation.mean >= best_val.validation.mean - se
            ),
            key=lambda c: simplicity_key(c, n_features),
        )
    )
    chosen = (
        close[0]
        if close
        else min(
            (best_bic.candidate, best_val.candidate), key=lambda c: simplicity_key(c, n_features)
        )
    )
    return Selection(
        scores,
        best_bic.candidate,
        best_val.candidate,
        se,
        close,
        chosen,
        "close_on_both" if close else "simpler_winner",
    )
