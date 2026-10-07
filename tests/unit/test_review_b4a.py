"""B4a review: the filter's and smoother's numerics on outliers, durations of closed labels,
ill-conditioned covariances, continuing a forward pass, the Jacobian in l_t and degenerate
volatility (BUILD_LOG, B4)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from quant.core.errors import ModelError
from quant.core.regime import Label
from tests.support.hmm import (
    DEFAULT_LABELS,
    Truth,
    density,
    identity_scaler,
    model_of,
    naive_filter,
    params_of,
    sample,
    three_full,
)


def _f(values: Any) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def _unit(means: list[float], startprob: list[float], transmat: list[list[float]]) -> Truth:
    """A one-feature HMM with unit variances."""
    return Truth(
        "full", _f(startprob), _f(transmat), _f([[m] for m in means]), np.ones((len(means), 1, 1))
    )


@pytest.mark.xfail(strict=True, reason="red:B4", raises=AssertionError)
@pytest.mark.spec("SPEC§6")
def test_identical_emissions_keep_the_prior_on_an_outlier() -> None:
    """Codex R1: when every state gives a row the same density, the posterior is the prior, however
    far out the row is. Adding the prior to emissions of -5e17 before centering rounds it away."""
    from quant.hmm.filter import ForwardFilter, forward_pass

    truth = _unit([0.0, 0.0], [0.9, 0.1], [[0.8, 0.2], [0.3, 0.7]])
    rows = _f([[1e9], [-1e9], [0.5]])
    trace = forward_pass(params_of(truth), identity_scaler(1), rows)
    prior = truth.startprob
    for t in range(len(rows)):
        np.testing.assert_allclose(trace.state_probs[t], prior, rtol=0, atol=1e-12)
        prior = prior @ truth.transmat
    outputs = ForwardFilter(model_of(truth)).run(rows)
    assert np.array([o.state_probs for o in outputs]).tobytes() == trace.state_probs.tobytes()


@pytest.mark.xfail(strict=True, reason="red:B4", raises=AssertionError)
@pytest.mark.spec("A4")
def test_the_smoother_keeps_the_transitions_on_an_outlier() -> None:
    """Codex R3: an outlier's emissions (about -5e17) must not swamp log a_jk + log β. With two
    bars, P(s_0 = j | x_0, x_1) ∝ π_j b_j(x_0) Σ_k a_jk b_k(x_1), and at x_1 = 1e9 state 1's
    density beats state 0's by a factor e^(2e9), so the sum is a_j1 b_1(x_1)."""
    from quant.eval.smoother import smooth

    truth = _unit([0.0, 2.0], [0.5, 0.5], [[0.99, 0.01], [0.1, 0.9]])
    posteriors = smooth(params_of(truth), identity_scaler(1), _f([[0.0], [1e9]]))
    first = truth.startprob * density(truth, _f([0.0])) * truth.transmat[:, 1]
    np.testing.assert_allclose(posteriors[0], first / first.sum(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(posteriors[1], [0.0, 1.0], rtol=0, atol=1e-12)


@pytest.mark.xfail(strict=True, reason="red:B4", raises=ModelError)
@pytest.mark.spec("SPEC§4", "SPEC§6")
def test_a_closed_state_the_chain_cant_reach_doesnt_block_the_duration() -> None:
    """Codex R2, Claude C1: CHOP is states 0 and 1, and state 1 is absorbing but unreachable (π
    and every transition into it are 0). Its row makes I - A_SS singular, yet from state 0 CHOP
    lasts 1 / (1 - 0.9) = 10 bars."""
    from quant.hmm.filter import ForwardFilter
    from quant.hmm.labels import remaining_duration

    truth = _unit(
        [0.0, 1.0, 3.0], [0.5, 0.0, 0.5], [[0.9, 0.0, 0.1], [0.0, 1.0, 0.0], [0.1, 0.0, 0.9]]
    )
    model = model_of(truth, [Label.CHOP, Label.CHOP, Label.STRESS])
    table, a = model.labels, truth.transmat
    assert remaining_duration(_f([1.0, 0.0, 0.0]), a, table, Label.CHOP) == pytest.approx(10.0)
    assert remaining_duration(_f([0.3, 0.0, 0.7]), a, table, Label.CHOP) == pytest.approx(10.0)
    rows = _f([[0.0], [0.1], [3.0], [3.1], [-0.2]])
    outputs = ForwardFilter(model).run(rows)
    assert [o.active_label for o in outputs] == [Label.CHOP] * 2 + [Label.STRESS] * 2 + [Label.CHOP]
    for out in outputs:
        assert out.state_probs[1] == 0.0
        assert out.remaining_duration == pytest.approx(10.0, rel=1e-12)


@pytest.mark.xfail(strict=True, reason="red:B4", raises=pytest.fail.Exception)
@pytest.mark.spec("SPEC§4", "SPEC§6")
def test_a_label_the_chain_can_never_leave_is_refused_when_the_filter_is_built() -> None:
    """Claude C1: a label short of every state, holding a closed class the chain can reach, has
    no finite duration. The filter refuses the model when it is built, rather than failing on
    whichever bar first gives that label the most probability."""
    from quant.hmm.filter import ForwardFilter

    absorbing = _unit(
        [0.0, 1.0, 3.0], [0.5, 0.5, 0.0], [[0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.1, 0.1, 0.8]]
    )
    # Codex R2's matrix: CHOP's state 1 is absorbing, and reachable from STRESS's state 2.
    reached = _unit(
        [0.0, 1.0, 3.0], [1.0, 0.0, 0.0], [[0.9, 0.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.1, 0.9]]
    )
    # Two states that only swap with each other: no single absorbing state, still closed.
    pair = _unit(
        [0.0, 1.0, 3.0], [0.4, 0.3, 0.3], [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.1, 0.1, 0.8]]
    )
    for truth, labels in (
        (absorbing, [Label.CHOP, Label.CALM_UP, Label.STRESS]),
        (reached, [Label.CHOP, Label.CHOP, Label.STRESS]),
        (pair, [Label.CHOP, Label.CHOP, Label.STRESS]),
    ):
        with pytest.raises(ModelError):
            ForwardFilter(model_of(truth, labels))
    # One label over every state is never left, and that is fine: its duration is None.
    ForwardFilter(model_of(absorbing, [Label.CHOP] * 3))


@pytest.mark.xfail(strict=True, reason="red:B4", raises=np.linalg.LinAlgError)
@pytest.mark.spec("SPEC§6")
def test_an_ill_conditioned_covariance_the_model_accepts_is_one_the_filter_evaluates() -> None:
    """Claude C2: diag(1, 1e-11) is positive definite, so GaussianParams accepts it; scipy's
    multivariate_normal then refused it with a bare LinAlgError. Every covariance the model
    accepts must give the textbook density."""
    from quant.eval.smoother import smooth
    from quant.hmm.filter import ForwardFilter, emission_logpdf, forward_pass

    base = three_full()
    covars = base.covars.copy()
    covars[0] = np.diag([1.0, 1e-11])
    truth = Truth("full", base.startprob, base.transmat, base.means, covars)
    model = model_of(truth, DEFAULT_LABELS["three_full"])
    _, x = sample(truth, 200, seed=301)
    outputs = ForwardFilter(model).run(x)
    reference = naive_filter(truth, x)
    probs = np.array([o.state_probs for o in outputs])
    np.testing.assert_allclose(probs, reference.probs, rtol=0, atol=1e-10)
    np.testing.assert_allclose([o.loglik for o in outputs], reference.loglik, rtol=1e-12)
    textbook = np.log([density(truth, row) for row in x[:20]])
    np.testing.assert_allclose(emission_logpdf(model.params, x[:20]), textbook, rtol=1e-12)
    assert forward_pass(model.params, model.scaler, x).state_probs.tobytes() == probs.tobytes()
    assert np.isfinite(smooth(model.params, model.scaler, x)).all()


@pytest.mark.xfail(strict=True, reason="red:B4", raises=TypeError)
@pytest.mark.spec("SPEC§6")
def test_a_forward_pass_continues_from_the_rows_before_it() -> None:
    """Claude C3: B4b scores a validation block by continuing the filter from the training rows,
    not by restarting it from π. `previous` is the last bar's log p, and the continued pass equals
    the matching rows of one pass over both blocks, and a warmed filter's outputs, bit for bit."""
    from quant.hmm.filter import ForwardFilter, forward_pass

    truth = three_full()
    model = model_of(truth, DEFAULT_LABELS["three_full"])
    params, scaler = model.params, model.scaler
    _, x = sample(truth, 500, seed=311)
    whole = forward_pass(params, scaler, x)
    train = forward_pass(params, scaler, x[:300])
    later = forward_pass(params, scaler, x[300:], previous=train.log_probs[-1])
    assert later.log_probs.tobytes() == whole.log_probs[300:].tobytes()
    assert later.loglik.tobytes() == whole.loglik[300:].tobytes()
    warmed = ForwardFilter.warmed(model, x[:300]).run(x[300:])
    assert np.array([o.loglik for o in warmed]).tobytes() == later.loglik.tobytes()

    # Restarting from π instead gets the first validation bar's l_t wrong.
    fresh = forward_pass(params, scaler, x[300:])
    assert fresh.loglik[0] != later.loglik[0]
    # None is the same as starting from π.
    assert (
        forward_pass(params, scaler, x[300:], previous=None).loglik.tobytes()
        == fresh.loglik.tobytes()
    )

    lp = train.log_probs[-1]
    for bad in (lp[:2], lp + 0.1, np.where(lp == lp.max(), np.nan, lp), lp.astype(np.float32)):
        with pytest.raises(ModelError):
            forward_pass(params, scaler, x[300:], previous=bad)


@pytest.mark.xfail(strict=True, reason="red:B4", raises=AssertionError)
@pytest.mark.spec("SPEC§3", "SPEC§6")
def test_the_predictive_log_likelihood_is_the_density_of_the_raw_features() -> None:
    """Claude C4: spec §6's l_t is log p(x_t | x_<t) for the raw row x_t, so it carries the
    scaler's Jacobian, -Σ log std. Then refits with different scalers score the same data on one
    scale: x = z std + mean is an HMM with means μ std + mean and covariances std Σ std."""
    from quant.hmm.filter import ForwardFilter, forward_pass
    from quant.hmm.model import ScalerParams

    truth = three_full()
    _, z = sample(truth, 400, seed=321)
    for mean, std in (
        ([0.0, 0.0], [1.0, 1.0]),
        ([0.001, -2.5], [0.02, 3.0]),
        ([5.0, 1.0], [10.0, 0.5]),
    ):
        m, s = _f(mean), _f(std)
        raw = z * s + m
        in_x = Truth(
            "full",
            truth.startprob,
            truth.transmat,
            truth.means * s + m,
            truth.covars * np.outer(s, s),
        )
        reference = naive_filter(in_x, raw)
        model = model_of(truth, DEFAULT_LABELS["three_full"], ScalerParams(("a", "b"), m, s))
        loglik = np.array([o.loglik for o in ForwardFilter(model).run(raw)])
        np.testing.assert_allclose(loglik, reference.loglik, rtol=0, atol=1e-10)
        trace = forward_pass(model.params, model.scaler, raw)
        assert trace.loglik.tobytes() == loglik.tobytes()
        assert math.isclose(loglik.sum(), reference.loglik.sum(), rel_tol=1e-12)


@pytest.mark.xfail(strict=True, reason="red:B4", raises=pytest.fail.Exception)
@pytest.mark.spec("SPEC§4")
def test_a_volatility_that_is_only_rounding_is_undefined() -> None:
    """Codex R5: 3,000 returns all equal to 0.001 have no volatility, but summation leaves a
    residue near 1e-19, which made v = 337 and d about 6e14 (STRESS). A state whose weighted
    returns are all equal is the same case."""
    from quant.hmm.labels import state_stats

    n = 3000
    with pytest.raises(ModelError):
        state_stats(np.full(n, 0.001), np.tile([0.5, 0.5], (n, 1)))
    even = np.arange(n) % 2 == 0
    returns = np.where(even, 0.001, np.random.default_rng(331).normal(0.0, 0.01, n))
    probs = np.stack([even, ~even], axis=1).astype(np.float64)
    with pytest.raises(ModelError):
        state_stats(returns, probs)
    # Real dispersion still measures: the same returns with the states mixed.
    v, d = state_stats(returns, np.tile([0.5, 0.5], (n, 1)))
    assert np.isfinite(v).all()
    assert np.isfinite(d).all()
