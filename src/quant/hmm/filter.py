"""Our log-space forward filter. Never a smoother or Viterbi (spec §6, I2).

**The recursion.** Each feature row x_t is standardized with the model's frozen scaler,
z_t = (x_t - mean) / std, and every density below is of z_t (no Jacobian term). With
log b_k(z) the Gaussian log-density of state k (`emission_logpdf`), π the start probabilities
and A the transition matrix:

- the prior is log π on the first bar, and log(p_{t-1} A)_k = logsumexp_j(log p_{t-1,j} +
  log a_jk) after it;
- a_t = log b(z_t) + prior, the unnormalized log posterior;
- the predictive log-likelihood is l_t = logsumexp_k a_tk = log p(x_t | x_<t) (the first bar's
  is log p(x_0));
- log p_t = a_t - l_t, the filtered log-probabilities: p_t ∝ b(x_t) ⊙ (p_{t-1} A).

Zero entries of π or A are log 0 = -inf and simply stay -inf (no warning may escape: use
`numpy.errstate(divide="ignore")` around the logs). Only past rows reach p_t, so the filter
can't look ahead.

**Per bar**, `ForwardFilter.step` returns every output of spec §6 (`FilterOutput`): the filtered
state and label probabilities, the next-bar probabilities q_t = p_t A and their label sums, the
normalized label entropy H_t, the predictive log-likelihood l_t, the active (most probable)
label and its expected remaining duration. The label view comes from quant.hmm.labels.

**State** is the last log p_t and a bar count. It snapshots to canonical JSON for EngineState
(`FilterState`, ARCHITECTURE §4.3), and a restored filter continues bit for bit.

**Model swap** (spec §6): a new model's filter starts from π on the first row of its own fit
window and runs forward to the swap time (`ForwardFilter.warmed`), so it only ever sees the
past.

The emissions are computed with scipy from the stored means and covariances; this module and
everything it imports never touch hmmlearn or eval (K3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from quant.core.errors import ModelError
from quant.core.model import StrictModel
from quant.core.regime import Label
from quant.hmm.labels import label_entropy, label_probs, remaining_duration, top_label
from quant.hmm.model import Floats, GaussianParams, HmmModel, ScalerParams


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
    """Factor each stored covariance once, then evaluate every row the same way."""

    def __init__(self, params: GaussianParams) -> None:
        self._gaussians = tuple(
            multivariate_normal(params.means[k], params.covars[k]) for k in range(params.n_states)
        )

    def row(self, z: Floats) -> Floats:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.array([gaussian.logpdf(z) for gaussian in self._gaussians], dtype=np.float64)


def _logs(params: GaussianParams) -> tuple[Floats, Floats]:
    with np.errstate(divide="ignore"):
        return np.log(params.startprob), np.log(params.transmat)


def _advance(prior: Floats, emissions: Floats) -> tuple[Floats, float]:
    joint = prior + emissions
    peak = float(np.max(joint))
    if not np.isfinite(peak):
        raise ModelError("predictive log-likelihood is not finite")
    # Subtract the peak before normalizing. At an outlier, subtracting the rounded
    # likelihood (possibly 1e18 in magnitude) would discard the normalization.
    shifted = joint - peak
    normalizer = float(logsumexp(shifted))
    loglik = peak + normalizer
    if not np.isfinite(loglik):
        raise ModelError("predictive log-likelihood is not finite")
    return shifted - normalizer, loglik


def emission_logpdf(params: GaussianParams, z: Floats) -> Floats:
    """log b_k(z_t) for every row and state: shape (n, K) for `z` of shape (n, d).

    Column k is `scipy.stats.multivariate_normal.logpdf(z, params.means[k], params.covars[k])`.
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
    - `loglik` (n,): the predictive log-likelihood l_t.
    """

    log_probs: Floats
    state_probs: Floats
    loglik: Floats


def forward_pass(params: GaussianParams, scaler: ScalerParams, rows: Floats) -> ForwardTrace:
    """Run the recursion of the module docstring from π over `rows` (n, d), raw feature rows in
    `scaler.columns` order. Used to label a model on its fit window and to score a validation
    block (B4b) before any HmmModel exists.

    Bar t's values are bit-identical to what `ForwardFilter.step` gives on the same rows from a
    fresh filter. Raises ModelError for n = 0, a shape or dtype that doesn't fit, a value that
    isn't finite, or a bar whose l_t isn't finite.
    """
    _check_rows(rows, params.n_features)
    if not len(rows) or len(scaler.columns) != params.n_features:
        raise ModelError("forward pass needs rows and a scaler matching the model")
    z = scaler.standardize(rows)
    emissions = _Emissions(params)
    prior, log_a = _logs(params)
    logs = np.empty((len(rows), params.n_states), dtype=np.float64)
    likelihoods = np.empty(len(rows), dtype=np.float64)
    for t, row in enumerate(z):
        logs[t], likelihoods[t] = _advance(prior, emissions.row(row))
        prior = logsumexp(logs[t, :, None] + log_a, axis=0)
    return ForwardTrace(logs, np.exp(logs), likelihoods)


@dataclass(frozen=True, eq=False)
class FilterOutput:
    """Every spec §6 output for one bar.

    - `state_probs` (K,): the filtered probabilities p_t, summing to 1.
    - `next_state_probs` (K,): q_t = p_t A, the probabilities for the next bar.
    - `label_probs`, `next_label_probs`: `hmm.labels.label_probs` of p_t and of q_t, keyed by
      the model's labels in rule order; a label the model lacks is absent (probability 0).
    - `entropy`: `hmm.labels.label_entropy` of `label_probs`, in [0, 1].
    - `loglik`: l_t = log p(x_t | x_<t).
    - `active_label`: `hmm.labels.top_label(label_probs)`.
    - `remaining_duration`: `hmm.labels.remaining_duration(p_t, A, labels, active_label)`, in
      bars, the current one included; None when the label covers every state.
    """

    state_probs: Floats
    next_state_probs: Floats
    label_probs: dict[Label, float]
    next_label_probs: dict[Label, float]
    entropy: float
    loglik: float
    active_label: Label
    remaining_duration: float | None


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
        """A filter under `model` that has consumed no row: its first step uses π."""
        self._model = model
        self._bars = 0
        self._log_probs: Floats | None = None
        self._log_start, self._log_a = _logs(model.params)
        self._emissions = _Emissions(model.params)
        self._model_hash = model.artifact_hash()

    @classmethod
    def warmed(cls, model: HmmModel, rows: Floats) -> ForwardFilter:
        """The filter a model swap starts from (spec §6): a new filter stepped over `rows`, the
        model's own feature rows from its fit window's first row up to (not including) the swap
        time, in order. Its state is exactly that of `ForwardFilter(model)` after `step` on each
        row. Rows of shape (0, d) give a fresh filter. Raises as `step` does."""
        result = cls(model)
        _check_rows(rows, model.params.n_features)
        for row in rows:
            result.step(row)
        return result

    @property
    def model(self) -> HmmModel:
        return self._model

    @property
    def bars(self) -> int:
        """Rows consumed so far."""
        return self._bars

    def step(self, row: Floats) -> FilterOutput:
        """Consume the next feature row (d,), raw and in the scaler's column order, and return
        its outputs.

        Raises ModelError, leaving the state exactly as it was, when the row isn't a float64
        array of shape (d,), a value isn't finite, or l_t isn't finite. A NaN feature is a model
        error, so rule 6 applies (spec §3).
        """
        if not isinstance(row, np.ndarray) or row.shape != (self.model.params.n_features,):
            raise ModelError("a filter step needs one feature row")
        z = self.model.scaler.standardize(row)
        prior = (
            self._log_start
            if self._log_probs is None
            else logsumexp(self._log_probs[:, None] + self._log_a, axis=0)
        )
        logs, loglik = _advance(prior, self._emissions.row(z))
        probs = np.exp(logs)
        next_probs = probs @ self.model.params.transmat
        table = self.model.labels
        labels = label_probs(probs, table)
        active = top_label(labels)
        output = FilterOutput(
            state_probs=probs,
            next_state_probs=next_probs,
            label_probs=labels,
            next_label_probs=label_probs(next_probs, table),
            entropy=label_entropy(labels, table),
            loglik=loglik,
            active_label=active,
            remaining_duration=remaining_duration(probs, self.model.params.transmat, table, active),
        )
        self._log_probs = logs
        self._bars += 1
        return output

    def run(self, rows: Floats) -> list[FilterOutput]:
        """`step` on each row of `rows` (n, d) in order, returning the outputs. Equal, bit for
        bit, to calling `step` n times. On an error, the rows before the bad one stay consumed."""
        _check_rows(rows, self.model.params.n_features)
        return [self.step(row) for row in rows]

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
            with np.errstate(over="ignore", invalid="ignore"):
                total = np.exp(logs).sum()
            if (
                logs.shape != (model.params.n_states,)
                or np.isnan(logs).any()
                or np.isposinf(logs).any()
                or not np.isfinite(total)
                or abs(total - 1.0) > 1e-9
            ):
                raise ModelError("filter snapshot log probabilities must sum to one")
            result._log_probs = logs
        result._bars = state.bars
        return result
