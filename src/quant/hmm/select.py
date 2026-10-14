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

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

from quant.core.model import StrictModel
from quant.data.frames import Floats, Ints
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
        raise NotImplementedError("red:B4")


def n_free_params(n_states: int, n_features: int, covariance_type: CovarianceType) -> int:
    """p of the module docstring: (K-1) + K(K-1) + Kd + (Kd for "diag", Kd(d+1)/2 for "full").
    K = 3, d = 2 gives 20 for diag and 23 for full. Raises ModelError for K < 1 or d < 1."""
    raise NotImplementedError("red:B4")


def simplicity_key(candidate: Candidate, n_features: int) -> tuple[int, int, int]:
    """(p, K, 0 for diag or 1 for full): sorting by it puts the simplest candidate first."""
    raise NotImplementedError("red:B4")


def bic(loglik: float, n_params: int, n_rows: int) -> float:
    """-2 * loglik + n_params * math.log(n_rows), exactly that expression. Raises ModelError
    when loglik isn't finite or n_rows < 1."""
    raise NotImplementedError("red:B4")


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
    raise NotImplementedError("red:B4")


def validation_start(fit_end: datetime, months: int) -> datetime:
    """The start of the validation block: `fit_end` moved back `months` calendar months, the
    same day and time of day in UTC, the day clamped to the target month's last day (2020-03-31
    back one month is 2020-02-29). Rows with available_at ≥ it are the block. A naive
    `fit_end` raises ClockError; months < 1 raises ModelError."""
    raise NotImplementedError("red:B4")


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
    raise NotImplementedError("red:B4")


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
    raise NotImplementedError("red:B4")


def bootstrap_se(loglik: Floats, sessions: Ints, *, draws: int, seed: int) -> float:
    """float(numpy.std(bootstrap_means(...), ddof=1)): the standard error of mean(loglik)."""
    raise NotImplementedError("red:B4")


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
    raise NotImplementedError("red:B4")


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
    raise NotImplementedError("red:B4")
