"""The forward-backward smoother, for evaluation only (I2, A4).

Smoothed posteriors P(s_t | x_1..x_n) use every row of a block, later ones included, so they
must never reach a decision: K3 keeps the decision path from importing eval, and the I2 scan
keeps this module's name out of everything but eval/. Calibration scores filtered labels
against them (A4), and the nightly review uses them to find wrong state calls.

This is our own implementation, in log space, over the stored parameters. Its forward half is
`quant.hmm.filter.forward_pass`, so its last row is the filter's last p_t. hmmlearn is not
imported here (only hmm.train may, I2).
"""

from __future__ import annotations

from quant.hmm.model import Floats, GaussianParams, ScalerParams


def smooth(params: GaussianParams, scaler: ScalerParams, rows: Floats) -> Floats:
    """Smoothed state posteriors, shape (n, K): row t is P(s_t | x_1..x_n), summing to 1.

    Standardize `rows` (n, d) with `scaler`, run `forward_pass` for log alpha, then the backward
    recursion in log space: log β_{n-1} = 0 and
    log β_t,j = logsumexp_k(log a_jk + log b_k(z_{t+1}) + log β_{t+1},k) - l_{t+1}, with l the
    forward pass's predictive log-likelihoods. The posterior is exp(log p_t + log β_t),
    renormalized to sum to 1. Raises as `forward_pass` does.
    """
    raise NotImplementedError("red:B4")
