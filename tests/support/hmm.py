"""Known HMMs, draws from them, and naive references for the filter and smoother (B4).

The references work in probability space with plain numpy and none of the code under test:
densities from the textbook formula (inverse and determinant), the scaled forward recursion,
and the scaled forward-backward of Rabiner. For a few bars, `brute_force` enumerates every
state path, which checks the references themselves. hmmlearn appears here only to build the
posteriors T-NL3 compares against; the I2 scan covers src/quant, not tests.

The references are exact enough for normal data. They underflow on outliers far from every
state, which is what the log-space filter is for.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from quant.core.regime import Label

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]

FIT_START = datetime(2016, 1, 4, 14, 30, tzinfo=UTC)
FIT_END = datetime(2019, 1, 1, tzinfo=UTC)  # D4: the first model's window, 2016-01 to 2018-12
FEATURE_HASH = "ab" * 32


@dataclass(frozen=True)
class Truth:
    """An HMM as plain arrays: what generates the data, and what the references read."""

    covariance_type: str
    startprob: Floats
    transmat: Floats
    means: Floats
    covars: Floats  # (K, d, d), full for both types

    @property
    def k(self) -> int:
        return len(self.startprob)

    @property
    def d(self) -> int:
        return int(self.means.shape[1])


def _f(values: Any) -> Floats:
    return np.array(values, dtype=np.float64)


def three_full() -> Truth:
    """Three well-separated states over two features, full covariances."""
    return Truth(
        "full",
        _f([0.5, 0.3, 0.2]),
        _f([[0.90, 0.07, 0.03], [0.10, 0.85, 0.05], [0.10, 0.10, 0.80]]),
        _f([[0.0, 0.0], [1.5, -1.0], [-2.0, 2.0]]),
        _f(
            [
                [[1.0, 0.3], [0.3, 0.8]],
                [[0.6, -0.2], [-0.2, 1.2]],
                [[2.0, 0.5], [0.5, 1.5]],
            ]
        ),
    )


def two_diag() -> Truth:
    """Two overlapping states over three features, diagonal covariances."""
    return Truth(
        "diag",
        _f([0.6, 0.4]),
        _f([[0.95, 0.05], [0.08, 0.92]]),
        _f([[0.2, 0.0, -0.1], [-0.5, 0.4, 0.3]]),
        np.stack([np.diag(_f([1.0, 0.7, 1.3])), np.diag(_f([2.0, 1.5, 0.9]))]),
    )


def four_sparse() -> Truth:
    """Four states with zeros in π and A: state 3 can't start, 0 never jumps to 2, 2 never
    returns to 0. Zeros are log 0 = -inf in the log-space filter."""
    return Truth(
        "full",
        _f([0.6, 0.4, 0.0, 0.0]),
        _f(
            [
                [0.90, 0.10, 0.00, 0.00],
                [0.05, 0.85, 0.10, 0.00],
                [0.00, 0.05, 0.85, 0.10],
                [0.10, 0.00, 0.10, 0.80],
            ]
        ),
        _f([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -2.0]]),
        np.stack([np.eye(2) * s for s in (0.5, 1.0, 1.5, 2.5)]),
    )


TRUTHS = {"three_full": three_full, "two_diag": two_diag, "four_sparse": four_sparse}


def sample(truth: Truth, n: int, seed: int) -> tuple[Ints, Floats]:
    """n bars drawn from the HMM: the hidden states and the observations (n, d)."""
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=np.int64)
    states[0] = rng.choice(truth.k, p=truth.startprob)
    for t in range(1, n):
        states[t] = rng.choice(truth.k, p=truth.transmat[states[t - 1]])
    x = np.empty((n, truth.d), dtype=np.float64)
    for t in range(n):
        s = states[t]
        x[t] = rng.multivariate_normal(truth.means[s], truth.covars[s])
    return states, x


def density(truth: Truth, x: Floats) -> Floats:
    """b_k(x) for every state, from the textbook formula: shape (K,)."""
    out = np.empty(truth.k)
    for k in range(truth.k):
        cov = truth.covars[k]
        dev = x - truth.means[k]
        quad = float(dev @ np.linalg.inv(cov) @ dev)
        norm = math.sqrt((2 * math.pi) ** truth.d * np.linalg.det(cov))
        out[k] = math.exp(-0.5 * quad) / norm
    return out


@dataclass(frozen=True)
class Reference:
    probs: Floats  # (n, K) filtered p_t
    next_probs: Floats  # (n, K) q_t = p_t A
    loglik: Floats  # (n,) log p(x_t | x_<t)


def naive_filter(truth: Truth, x: Floats) -> Reference:
    """The scaled forward recursion in probability space."""
    n = len(x)
    probs = np.empty((n, truth.k))
    loglik = np.empty(n)
    prior = truth.startprob
    for t in range(n):
        joint = density(truth, x[t]) * prior
        total = joint.sum()
        probs[t] = joint / total
        loglik[t] = math.log(total)
        prior = probs[t] @ truth.transmat
    return Reference(probs, probs @ truth.transmat, loglik)


def naive_smoother(truth: Truth, x: Floats) -> Floats:
    """Rabiner's scaled forward-backward in probability space: P(s_t | x_1..x_n), (n, K)."""
    n = len(x)
    alpha = np.empty((n, truth.k))
    scale = np.empty(n)
    b = np.array([density(truth, row) for row in x])
    prior = truth.startprob
    for t in range(n):
        joint = b[t] * prior
        scale[t] = joint.sum()
        alpha[t] = joint / scale[t]
        prior = alpha[t] @ truth.transmat
    beta = np.ones((n, truth.k))
    for t in range(n - 2, -1, -1):
        beta[t] = truth.transmat @ (b[t + 1] * beta[t + 1]) / scale[t + 1]
    gamma = alpha * beta
    return gamma / gamma.sum(axis=1, keepdims=True)


def brute_force(truth: Truth, x: Floats) -> tuple[Floats, Floats]:
    """(filtered, smoothed) marginals by enumerating every state path: K**n paths, so keep n
    small. filtered[t] is P(s_t | x_0..x_t), smoothed[t] is P(s_t | x_0..x_{n-1})."""
    n, k = len(x), truth.k
    b = np.array([density(truth, row) for row in x])
    filtered = np.empty((n, k))
    for t in range(n):
        mass = np.zeros(k)
        for path in itertools.product(range(k), repeat=t + 1):
            p = truth.startprob[path[0]] * b[0, path[0]]
            for i in range(1, t + 1):
                p *= truth.transmat[path[i - 1], path[i]] * b[i, path[i]]
            mass[path[-1]] += p
        filtered[t] = mass / mass.sum()
    smoothed = np.zeros((n, k))
    for path in itertools.product(range(k), repeat=n):
        p = truth.startprob[path[0]] * b[0, path[0]]
        for i in range(1, n):
            p *= truth.transmat[path[i - 1], path[i]] * b[i, path[i]]
        for t in range(n):
            smoothed[t, path[t]] += p
    return filtered, smoothed / smoothed.sum(axis=1, keepdims=True)


def hmmlearn_model(truth: Truth) -> Any:
    """hmmlearn's GaussianHMM with the truth's parameters set directly, never fit."""
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(n_components=truth.k, covariance_type=truth.covariance_type)
    model.n_features = truth.d
    model.startprob_ = truth.startprob.copy()
    model.transmat_ = truth.transmat.copy()
    model.means_ = truth.means.copy()
    if truth.covariance_type == "diag":
        model.covars_ = np.array([np.diag(c) for c in truth.covars])
    else:
        model.covars_ = truth.covars.copy()
    return model


def hmmlearn_posteriors(truth: Truth, x: Floats) -> Floats:
    """hmmlearn's predict_proba: smoothed posteriors, which T-NL3 tells apart from the filter."""
    return np.asarray(hmmlearn_model(truth).predict_proba(x), dtype=np.float64)


def hmmlearn_score(truth: Truth, x: Floats) -> float:
    """hmmlearn's total log-likelihood log p(x_0..x_{n-1})."""
    return float(hmmlearn_model(truth).score(x))


# --- Building models under test (these call the stubs) -----------------------------------


def params_of(truth: Truth) -> Any:
    from quant.hmm.model import GaussianParams

    return GaussianParams(
        truth.covariance_type,  # type: ignore[arg-type]
        truth.startprob.copy(),
        truth.transmat.copy(),
        truth.means.copy(),
        truth.covars.copy(),
    )


def identity_scaler(d: int) -> Any:
    from quant.hmm.model import ScalerParams

    return ScalerParams(identity_columns(d), np.zeros(d, dtype=np.float64), np.ones(d))


def identity_columns(d: int) -> tuple[str, ...]:
    """The column names of `identity_scaler(d)`: f0, f1, ..."""
    return tuple(f"f{i}" for i in range(d))


def columns_of(rows: Floats) -> tuple[str, ...]:
    """The column names `identity_scaler` gives rows (n, d) or a row (d,) of this width."""
    return identity_columns(rows.shape[-1])


def training_record(base_seed: int = 20160104) -> Any:
    from quant.hmm.model import TrainingRecord

    return TrainingRecord(
        base_seed=base_seed,
        seeds=(base_seed, base_seed + 1, base_seed + 2),
        restart_logliks=(-1234.5, -1200.25, -1201.0),
        best=1,
        info={"data_hash": "cd" * 32, "note": "synthetic"},
    )


def label_table(labels: list[Label], v: list[float] | None = None) -> Any:
    from quant.hmm.labels import LabelTable, StateLabel, r1_thresholds

    vs = v or [1.0] * len(labels)
    return LabelTable(
        thresholds=r1_thresholds(),
        states=tuple(
            StateLabel(v=vi, d=0.25 * i, label=label)
            for i, (vi, label) in enumerate(zip(vs, labels, strict=True))
        ),
    )


DEFAULT_LABELS = {
    "three_full": [Label.CHOP, Label.CALM_UP, Label.STRESS],
    "two_diag": [Label.CALM_UP, Label.STRESS],
    "four_sparse": [Label.CHOP, Label.CALM_UP, Label.CHOP, Label.CRASH],
}


def model_of(
    truth: Truth,
    labels: list[Label] | None = None,
    scaler: Any = None,
    *,
    fit_end: datetime = FIT_END,
    training: Any = None,
) -> Any:
    """An HmmModel with the truth's parameters, an identity scaler unless one is given, the D4
    fit window and a fixed training record."""
    from quant.hmm.model import HmmModel

    if labels is None:
        labels = [Label.CHOP] * truth.k
    return HmmModel(
        params=params_of(truth),
        scaler=scaler if scaler is not None else identity_scaler(truth.d),
        fit_start=FIT_START,
        fit_end=fit_end,
        feature_config_hash=FEATURE_HASH,
        labels=label_table(labels),
        training=training if training is not None else training_record(),
    )
