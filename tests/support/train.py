"""Synthetic training data and slow references for training and selection (B4b).

`training_data` lays bars drawn from a known HMM (tests.support.hmm) on a synthetic calendar of
weekday sessions of BARS_PER_SESSION bars, in the 1990s so nobody mistakes them for market data.
The references re-derive, with plain Python loops and none of the code under test, the
quantities quant.hmm.select defines: free-parameter counts, BIC, the session-block bootstrap,
the sanity checks and the selection rule. `match_states` undoes the relabeling a fit is free to
make, so recovered parameters can be compared with the generator's.
"""

from __future__ import annotations

import itertools
import math
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np

from tests.support.hmm import FEATURE_HASH, Floats, Ints, Truth, _f, identity_columns, sample

BARS_PER_SESSION = 7
FIRST_SESSION = date(1990, 1, 2)
R1_BASE_SEED = 20160104
RETURN_SCALE = 0.002  # raw log return of a bar = RETURN_SCALE * its first feature


def three_diag() -> Truth:
    """three_full's states without their correlations: the generator for 'selection picks the
    generator's covariance type' when that type is diag."""
    return Truth(
        "diag",
        _f([0.5, 0.3, 0.2]),
        _f([[0.90, 0.07, 0.03], [0.10, 0.85, 0.05], [0.10, 0.10, 0.80]]),
        _f([[0.0, 0.0], [1.5, -1.0], [-2.0, 2.0]]),
        np.stack([np.diag(_f([1.0, 0.8])), np.diag(_f([0.6, 1.2])), np.diag(_f([2.0, 1.5]))]),
    )


def sessions(count: int) -> list[date]:
    """The first `count` weekdays from FIRST_SESSION."""
    days: list[date] = []
    day = FIRST_SESSION
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def calendar(n: int) -> tuple[Ints, Ints, Ints, datetime, datetime]:
    """(t_open_us, available_at_us, session_ordinals, fit_start, fit_end) for n bars: session
    s's bar h opens at 14:30 UTC + h hours and is available an hour later. The fit window runs
    from the first session's midnight to the midnight after the last session."""
    days = sessions(math.ceil(n / BARS_PER_SESSION))
    opens: list[int] = []
    ordinals: list[int] = []
    for day in days:
        base = datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC)
        for h in range(BARS_PER_SESSION):
            opens.append(int((base + timedelta(hours=h)).timestamp()) * 1_000_000)
            ordinals.append(day.toordinal())
    t_open = np.array(opens[:n], dtype=np.int64)
    available = t_open + 3_600_000_000
    start = datetime(days[0].year, days[0].month, days[0].day, tzinfo=UTC)
    last = days[-1] + timedelta(days=1)
    end = datetime(last.year, last.month, last.day, tzinfo=UTC)
    return t_open, available, np.array(ordinals[:n], dtype=np.int64), start, end


def training_data(truth: Truth, n: int, seed: int, *, feature_hash: str = FEATURE_HASH) -> Any:
    """quant.hmm.train.TrainingData for n bars drawn from `truth` with `seed`: the features are
    the draws (columns f0, f1, ...), and each bar's raw return is RETURN_SCALE times its first
    feature, so the generator's states differ in mean and volatility."""
    from quant.features.stream import FeatureMatrix
    from quant.hmm.train import TrainingData

    _, x = sample(truth, n, seed)
    t_open, available, ordinals, start, end = calendar(n)
    matrix = FeatureMatrix(identity_columns(truth.d), t_open, available, x)
    return TrainingData(
        matrix=matrix,
        returns=RETURN_SCALE * x[:, 0].copy(),
        sessions=ordinals,
        fit_start=start,
        fit_end=end,
        feature_config_hash=feature_hash,
    )


def match_states(truth_means: Floats, fitted_means: Floats) -> tuple[int, ...]:
    """The permutation `perm` minimizing Σ_k |fitted_means[perm[k]] - truth_means[k]|²: fitted
    state perm[k] plays the generator's state k. Brute force over K! orders."""
    k = len(truth_means)
    best: tuple[float, tuple[int, ...]] | None = None
    for perm in itertools.permutations(range(k)):
        cost = float(
            sum(np.sum((fitted_means[p] - truth_means[i]) ** 2) for i, p in enumerate(perm))
        )
        if best is None or cost < best[0]:
            best = (cost, perm)
    assert best is not None
    return best[1]


def ref_n_params(k: int, d: int, covariance_type: str) -> int:
    """Free parameters of a Gaussian HMM: π (K-1), A (K(K-1)), means (Kd), and the covariances,
    Kd for diag and Kd(d+1)/2 for full."""
    covariances = k * d if covariance_type == "diag" else k * d * (d + 1) // 2
    return (k - 1) + k * (k - 1) + k * d + covariances


def ref_bic(loglik: float, n_params: int, n_rows: int) -> float:
    return -2.0 * loglik + n_params * math.log(n_rows)


def ref_bootstrap(loglik: Floats, session_ids: Ints, draws: int, seed: int) -> Floats:
    """The session-block bootstrap of mean(loglik), loop by loop: blocks are whole sessions in
    order of first appearance; draw b picks S of the S blocks uniformly with replacement, the
    picks being row b of numpy.random.default_rng(seed).integers(0, S, size=(draws, S)), and
    its statistic is the mean of l_t over every bar of the picked blocks (a block picked twice
    counts twice)."""
    blocks: list[list[float]] = []
    last = None
    for value, sid in zip(loglik.tolist(), session_ids.tolist(), strict=True):
        if sid != last:
            blocks.append([])
            last = sid
        blocks[-1].append(value)
    picks = np.random.default_rng(seed).integers(0, len(blocks), size=(draws, len(blocks)))
    means = np.empty(draws)
    for b in range(draws):
        total = 0.0
        count = 0
        for j in picks[b]:
            total += math.fsum(blocks[j])
            count += len(blocks[j])
        means[b] = total / count
    return means


def ref_se(means: Floats) -> float:
    """The sample standard deviation of the draws (ddof = 1)."""
    mu = math.fsum(means) / len(means)
    return math.sqrt(math.fsum((m - mu) ** 2 for m in means) / (len(means) - 1))


def ref_sanity(transmat: Floats, covars: Floats, state_probs: Floats) -> dict[str, list[bool]]:
    """Spec §4's rejects per state, True where the state fails: occupancy (mean filtered
    probability) under 0.02, expected duration 1 / (1 - a_kk) under 2 bars, covariance
    condition number (largest over smallest eigenvalue) above 1e6."""
    k = len(transmat)
    occupancy = [math.fsum(state_probs[:, j]) / len(state_probs) < 0.02 for j in range(k)]
    duration = [transmat[j, j] < 1.0 and 1.0 / (1.0 - transmat[j, j]) < 2.0 for j in range(k)]
    eig = [np.linalg.eigvalsh(c) for c in covars]
    condition = [float(e.max() / e.min()) > 1e6 for e in eig]
    return {"occupancy": occupancy, "duration": duration, "condition": condition}


def r1_candidates() -> list[tuple[int, str]]:
    """Spec §4's grid in the documented order: K ascending, diag before full."""
    return [(k, c) for k in (2, 3, 4, 5) for c in ("diag", "full")]
