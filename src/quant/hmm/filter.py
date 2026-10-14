"""Our log-space forward filter. Never a smoother or Viterbi (spec §6, I2).

**The recursion.** Each feature row x_t is standardized with the model's frozen scaler,
z_t = (x_t - mean) / std. With log b_k(z) the Gaussian log-density of state k
(`emission_logpdf`), π the start probabilities and A the transition matrix:

- the prior is log π on the first bar, and log(p_{t-1} A)_k = logsumexp_j(log p_{t-1,j} +
  log a_jk) after it;
- e_t = log b(z_t) - max_k log b_k(z_t), the emissions centered on their largest, so that an
  outlier's densities (about -5e17 at 1e9) can't round the prior away;
- a_t = e_t + prior, the unnormalized log posterior, and c_t = logsumexp_k a_tk;
- log p_t = a_t - c_t, the filtered log-probabilities: p_t ∝ b(z_t) ⊙ (p_{t-1} A);
- the predictive log-likelihood is l_t = max_k log b_k(z_t) + c_t + J, with
  J = -Σ_i log std_i the scaler's Jacobian, so that l_t = log p(x_t | x_<t) is the density of
  the raw row (spec §6), and models with different scalers score the same rows on one scale.
  The first bar's is log p(x_0).

Zero entries of π or A are log 0 = -inf and simply stay -inf (no warning may escape: use
`numpy.errstate(divide="ignore")` around the logs). Only past rows reach p_t, so the filter
can't look ahead.

**Per bar**, `ForwardFilter.step` returns every output of spec §6 (`FilterOutput`): the filtered
state and label probabilities, the next-bar probabilities q_t = p_t A and their label sums, the
normalized label entropy H_t, the predictive log-likelihood l_t, the top (most probable) label
and its expected remaining duration. The top label is not the switching engine's active label
(spec §8), which has hysteresis; `ForwardFilter.remaining` gives any label's duration. The label
view comes from quant.hmm.labels.

**Columns.** Every entry point takes the rows' column names with them, and refuses names that
aren't exactly the model scaler's (`ScalerParams.standardize`), so a row of the same width from
another feature config can't be filtered.

**State** is the last log p_t and a bar count. It snapshots to canonical JSON for EngineState
(`FilterState`, ARCHITECTURE §4.3), and a restored filter continues bit for bit.

**Model swap** (spec §6): a new model's filter starts from π on the first row of its own fit
window and runs forward to the swap time (`ForwardFilter.warmed`), so it only ever sees the
past.

The emissions come from the stored means and covariances through each covariance's inverse
Cholesky factor (`GaussianParams.whitening`, scipy.linalg), checked when the model is built, so
every covariance a model accepts evaluates. This module and everything it imports never touch
hmmlearn or eval (K3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant.core.errors import ModelError
from quant.core.model import StrictModel
from quant.core.regime import Label
from quant.data.frames import Floats
from quant.hmm.labels import Durations, label_entropy, label_probs, top_label
from quant.hmm.model import GaussianParams, HmmModel, ScalerParams


def _check_rows(rows: Floats, width: int) -> None:
    # Check the container here; run() validates values as each row is consumed.
    if (
        not isinstance(rows, np.ndarray)
        or rows.dtype != np.float64
        or rows.ndim != 2
        or rows.shape[1] != width
    ):
        raise ModelError(f"rows must be a float64 matrix with {width} columns")


class _Emissions:
    """Each state's whitening factor, computed once; every row is evaluated the same way."""

    def __init__(self, params: GaussianParams) -> None:
        self._means = params.means
        self._inverse, half_logdet = params.whitening()
        self._offset = -0.5 * params.n_features * np.log(2 * np.pi) - half_logdet

    def row(self, z: Floats) -> Floats:
        with np.errstate(over="ignore", invalid="ignore"):
            white = np.einsum("kij,kj->ki", self._inverse, z - self._means)
            result: Floats = self._offset - 0.5 * np.einsum("ki,ki->k", white, white)
        # The quadratic form is ≥ 0; NaN can only come from an overflow (inf - inf), where the
        # density is 0 to float64.
        result[np.isnan(result)] = -np.inf
        return result


def _logs(params: GaussianParams) -> tuple[Floats, Floats]:
    with np.errstate(divide="ignore"):
        return np.log(params.startprob), np.log(params.transmat)


def _jacobian(scaler: ScalerParams) -> float:
    return -float(np.log(scaler.std).sum())


def _predict(log_probs: Floats, log_a: Floats) -> Floats:
    """log(p A) from log p: logsumexp_j(log p_j + log a_jk) for each k, -inf where every term
    is. Plain numpy, since scipy's logsumexp costs tens of microseconds a call."""
    joint = log_probs[:, None] + log_a
    peak = np.max(joint, axis=0)
    peak[~np.isfinite(peak)] = 0.0
    with np.errstate(divide="ignore"):
        result: Floats = np.log(np.exp(joint - peak).sum(axis=0)) + peak
    return result


def _advance(prior: Floats, emissions: Floats, jacobian: float) -> tuple[Floats, float]:
    top = float(np.max(emissions))
    if not np.isfinite(top):
        raise ModelError("predictive log-likelihood is not finite")
    # Center the emissions before the prior goes in: at an outlier they are about -5e17, and
    # adding log π to that would round the prior away.
    joint = prior + (emissions - top)
    peak = float(np.max(joint))
    if not np.isfinite(peak):
        raise ModelError("predictive log-likelihood is not finite")
    shifted = joint - peak
    normalizer = float(np.log(np.exp(shifted).sum()))
    loglik = top + (peak + normalizer) + jacobian
    if not np.isfinite(loglik):
        raise ModelError("predictive log-likelihood is not finite")
    return shifted - normalizer, loglik


def _check_log_probs(log_probs: Floats, k: int) -> Floats:
    """A copy of `log_probs` when it is a float64 (K,) vector of log-probabilities: no NaN or
    +inf, and exp summing to 1 within 1e-9. Raises ModelError otherwise."""
    if (
        not isinstance(log_probs, np.ndarray)
        or log_probs.dtype != np.float64
        or log_probs.shape != (k,)
        or np.isnan(log_probs).any()
        or np.isposinf(log_probs).any()
    ):
        raise ModelError(f"log probabilities must be a float64 vector of {k} entries")
    with np.errstate(over="ignore", invalid="ignore"):
        total = np.exp(log_probs).sum()
    if not np.isfinite(total) or abs(total - 1.0) > 1e-9:
        raise ModelError("log probabilities must sum to one")
    return log_probs.copy()


def emission_logpdf(params: GaussianParams, z: Floats) -> Floats:
    """log b_k(z_t) for every row and state: shape (n, K) for `z` of shape (n, d).

    Column k is the Gaussian log-density of N(params.means[k], params.covars[k]) at each row,
    -½ d log 2π - h_k - ½ |W_k (z - μ_k)|² with (W, h) = `params.whitening()`, and -inf where
    the quadratic form overflows float64.
    Row t of the result depends only on z[t], and is bit-identical to the result for z[t:t+1]
    alone. Raises ModelError when `z` isn't a float64 (n, d) array of finite values.
    """
    _check_rows(z, params.n_features)
    if not np.isfinite(z).all():
        raise ModelError("emission rows must be finite")
    emissions = _Emissions(params)
    result = np.empty((len(z), params.n_states), dtype=np.float64)
    for t, row in enumerate(z):
        result[t] = emissions.row(row)
    return result


@dataclass(frozen=True, eq=False)
class ForwardTrace:
    """The forward filter's state outputs over a block of rows: row t is bar t.

    - `log_probs` (n, K): log p_t, -inf where a state is impossible;
    - `state_probs` (n, K): p_t = exp(log p_t);
    - `loglik` (n,): the predictive log-likelihood l_t, Jacobian included;
    - `emissions` (n, K): log b_k(z_t), equal to `emission_logpdf` of the standardized rows.
    """

    log_probs: Floats
    state_probs: Floats
    loglik: Floats
    emissions: Floats


def forward_pass(
    params: GaussianParams,
    scaler: ScalerParams,
    rows: Floats,
    *,
    columns: tuple[str, ...],
    previous: Floats | None = None,
) -> ForwardTrace:
    """Run the recursion of the module docstring over `rows` (n, d), raw feature rows whose
    columns `columns` names, which must be exactly `scaler.columns`. Used to label a model on
    its fit window and to score a validation block (B4b) before any HmmModel exists.

    `previous` is log p of the bar just before rows[0] (`ForwardTrace.log_probs[-1]` of the pass
    over the rows before), so the first prior is log(p A) and the pass continues that one; None
    starts from π. A validation block is scored by continuing from its training rows: restarting
    from π, which EM fits to the first training row, would misprice the block's first bars.

    Bar t's values are bit-identical to what `ForwardFilter.step` gives on the same rows from a
    fresh filter (with `previous`, from a filter that consumed the rows before), and a pass over
    rows[:m] continued over rows[m:] equals one pass over all of them. Raises ModelError for
    n = 0, `columns` other than the scaler's, a shape or dtype that doesn't fit, a value that
    isn't finite, a bar whose l_t isn't finite, or a `previous` that isn't a float64 (K,) vector of log-probabilities (no NaN or
    +inf, exp summing to 1 within 1e-9).
    """
    _check_rows(rows, params.n_features)
    if not len(rows) or len(scaler.columns) != params.n_features:
        raise ModelError("forward pass needs rows and a scaler matching the model")
    z = scaler.standardize(rows, columns=columns)
    emissions = _Emissions(params)
    prior, log_a = _logs(params)
    if previous is not None:
        prior = _predict(_check_log_probs(previous, params.n_states), log_a)
    jacobian = _jacobian(scaler)
    logs = np.empty((len(rows), params.n_states), dtype=np.float64)
    densities = np.empty((len(rows), params.n_states), dtype=np.float64)
    likelihoods = np.empty(len(rows), dtype=np.float64)
    for t, row in enumerate(z):
        densities[t] = emissions.row(row)
        logs[t], likelihoods[t] = _advance(prior, densities[t], jacobian)
        prior = _predict(logs[t], log_a)
    return ForwardTrace(logs, np.exp(logs), likelihoods, densities)


@dataclass(frozen=True, eq=False)
class FilterOutput:
    """Every spec §6 output for one bar.

    - `state_probs` (K,): the filtered probabilities p_t, summing to 1.
    - `next_state_probs` (K,): q_t = p_t A, the probabilities for the next bar.
    - `label_probs`, `next_label_probs`: `hmm.labels.label_probs` of p_t and of q_t, keyed by
      the model's labels in rule order; a label the model lacks is absent (probability 0).
    - `entropy`: `hmm.labels.label_entropy` of `label_probs`, in [0, 1].
    - `loglik`: l_t = log p(x_t | x_<t), the density of the raw row (Jacobian included).
    - `top_label`: `hmm.labels.top_label(label_probs)`, the most probable label. Not the
      switching engine's active label (spec §8), which has hysteresis.
    - `top_label_remaining`: `hmm.labels.remaining_duration(p_t, A, labels, top_label)`, in
      bars, the current one included; None when the label covers every state.
    """

    state_probs: Floats
    next_state_probs: Floats
    label_probs: dict[Label, float]
    next_label_probs: dict[Label, float]
    entropy: float
    loglik: float
    top_label: Label
    top_label_remaining: float | None


class FilterState(StrictModel):
    """A ForwardFilter's state as plain JSON data, for EngineState (ARCHITECTURE §4.3).

    - `model_hash`: the artifact hash of the model the filter runs.
    - `bars`: rows consumed so far.
    - `log_probs`: log p_t after the last row, one entry per state, None standing for log 0
      (-inf, which JSON can't hold). None itself before the first row.

    `FilterState.model_validate(canonical_loads(canonical_dumps(state)))` reads it back.
    """

    model_hash: str
    bars: int
    log_probs: list[float | None] | None


class ForwardFilter:
    """The streaming forward filter for one model. Mutable: `step` advances it in place.
    EngineState keeps `snapshot()`, and `restore` rebuilds the filter from one."""

    def __init__(self, model: HmmModel) -> None:
        """A filter under `model` that has consumed no row: its first step uses π.

        Everything that depends only on the model is computed here, the labels' exit times
        among it (`hmm.labels.Durations`). So a model with a label the chain can reach but
        never leave (short of every state) raises ModelError here, not on some later bar."""
        self._model = model
        self._bars = 0
        self._log_probs: Floats | None = None
        self._log_start, self._log_a = _logs(model.params)
        self._emissions = _Emissions(model.params)
        self._jacobian = _jacobian(model.scaler)
        self._durations = Durations(model.params.startprob, model.params.transmat, model.labels)
        self._model_hash = model.artifact_hash()

    @classmethod
    def warmed(cls, model: HmmModel, rows: Floats, *, columns: tuple[str, ...]) -> ForwardFilter:
        """The filter a model swap starts from (spec §6): a new filter stepped over `rows`, the
        model's own feature rows from its fit window's first row up to (not including) the swap
        time, in order, with `columns` naming their columns. Its state is exactly that of
        `ForwardFilter(model)` after `step` on each row. Rows of shape (0, d) give a fresh
        filter. Raises as `step` does, and for `columns` other than the scaler's even with no
        rows."""
        result = cls(model)
        _check_rows(rows, model.params.n_features)
        if columns != model.scaler.columns:
            raise ModelError("rows must have the model scaler's columns")
        for row in rows:
            result.step(row, columns=columns)
        return result

    @property
    def model(self) -> HmmModel:
        return self._model

    @property
    def bars(self) -> int:
        """Rows consumed so far."""
        return self._bars

    def step(self, row: Floats, *, columns: tuple[str, ...]) -> FilterOutput:
        """Consume the next feature row (d,), raw, with `columns` naming its columns, and
        return its outputs.

        Raises ModelError, leaving the state exactly as it was, when `columns` isn't exactly
        the model scaler's, the row isn't a float64 array of shape (d,), a value isn't finite,
        or l_t isn't finite. A NaN feature is a model error, so rule 6 applies (spec §3).
        """
        if not isinstance(row, np.ndarray) or row.shape != (self.model.params.n_features,):
            raise ModelError("a filter step needs one feature row")
        z = self.model.scaler.standardize(row, columns=columns)
        prior = (
            self._log_start if self._log_probs is None else _predict(self._log_probs, self._log_a)
        )
        logs, loglik = _advance(prior, self._emissions.row(z), self._jacobian)
        probs = np.exp(logs)
        next_probs = probs @ self.model.params.transmat
        table = self.model.labels
        labels = label_probs(probs, table)
        top = top_label(labels)
        output = FilterOutput(
            state_probs=probs,
            next_state_probs=next_probs,
            label_probs=labels,
            next_label_probs=label_probs(next_probs, table),
            entropy=label_entropy(labels, table),
            loglik=loglik,
            top_label=top,
            top_label_remaining=self._durations.remaining(probs, top),
        )
        self._log_probs = logs
        self._bars += 1
        return output

    def run(self, rows: Floats, *, columns: tuple[str, ...]) -> list[FilterOutput]:
        """`step` on each row of `rows` (n, d) in order, `columns` naming their columns,
        returning the outputs. Equal, bit for bit, to calling `step` n times. On an error, the
        rows before the bad one stay consumed; `columns` other than the scaler's consume none."""
        _check_rows(rows, self.model.params.n_features)
        if columns != self.model.scaler.columns:
            raise ModelError("rows must have the model scaler's columns")
        return [self.step(row, columns=columns) for row in rows]

    def remaining(self, label: Label) -> float | None:
        """The expected remaining duration of `label` from the current p_t, in bars, the current
        one included (`hmm.labels.remaining_duration`): what the switching engine reads for its
        active label, which need not be the top one. Raises ModelError before the first row, or
        as `remaining_duration` does (a label the model lacks, or one with no probability)."""
        if self._log_probs is None:
            raise ModelError("the filter has consumed no row")
        return self._durations.remaining(np.exp(self._log_probs), label)

    def snapshot(self) -> FilterState:
        """The state now. Two filters under one model that consumed the same rows give equal
        snapshots with identical canonical JSON."""
        return FilterState(
            model_hash=self._model_hash,
            bars=self.bars,
            log_probs=(
                None
                if self._log_probs is None
                else [None if np.isneginf(p) else float(p) for p in self._log_probs]
            ),
        )

    @classmethod
    def restore(cls, model: HmmModel, state: FilterState) -> ForwardFilter:
        """The filter `state` describes. It continues exactly as the filter that took the
        snapshot would have: the same outputs, bit for bit, and the same later snapshots.

        Raises ModelError when `state.model_hash` isn't `model.artifact_hash()`, `bars` < 0,
        `log_probs` is None while `bars` > 0 or set while `bars` is 0, `log_probs` doesn't have
        K entries, or exp(log_probs) doesn't sum to 1 within 1e-9.
        """
        result = cls(model)
        if (
            state.model_hash != result._model_hash
            or state.bars < 0
            or (state.log_probs is None) != (state.bars == 0)
        ):
            raise ModelError("filter snapshot does not match its model or bar count")
        if state.log_probs is not None:
            logs = np.array(
                [float("-inf") if p is None else p for p in state.log_probs], dtype=np.float64
            )
            result._log_probs = _check_log_probs(logs, model.params.n_states)
        result._bars = state.bars
        return result
