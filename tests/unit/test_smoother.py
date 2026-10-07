"""B4: eval.smoother, our forward-backward smoother, for evaluation only (I2, A4)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.support.hmm import (
    TRUTHS,
    brute_force,
    identity_scaler,
    naive_smoother,
    params_of,
    sample,
)


@pytest.mark.spec("A4", "I2")
@pytest.mark.parametrize("name", sorted(TRUTHS))
def test_the_smoother_matches_the_naive_reference(name: str) -> None:
    from quant.eval.smoother import smooth
    from quant.hmm.filter import forward_pass

    truth = TRUTHS[name]()
    params, scaler = params_of(truth), identity_scaler(truth.d)
    _, x = sample(truth, 1500, seed=201)
    posteriors = smooth(params, scaler, x)
    assert posteriors.shape == (len(x), truth.k)
    assert posteriors.dtype == np.float64
    np.testing.assert_allclose(posteriors, naive_smoother(truth, x), rtol=0, atol=1e-10)
    np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, rtol=0, atol=1e-12)

    # Its last row is the filter's last p_t: nothing comes after it to smooth with.
    last = forward_pass(params, scaler, x).state_probs[-1]
    np.testing.assert_allclose(posteriors[-1], last, rtol=0, atol=1e-12)

    _, few = sample(truth, 6, seed=202)
    _, exact = brute_force(truth, few)
    np.testing.assert_allclose(smooth(params, scaler, few), exact, rtol=0, atol=1e-12)


@pytest.mark.spec("A4")
def test_the_smoother_stays_finite_on_outliers() -> None:
    from quant.eval.smoother import smooth

    truth = TRUTHS["three_full"]()
    _, x = sample(truth, 2000, seed=203)
    x[500] = [1e9, -1e9]
    x[501] = [40.0, 40.0]
    posteriors = smooth(params_of(truth), identity_scaler(truth.d), x)
    assert np.isfinite(posteriors).all()
    np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, rtol=0, atol=1e-12)
