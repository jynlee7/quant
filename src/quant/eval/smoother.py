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

import numpy as np
from scipy.special import logsumexp

from quant.core.errors import ModelError
from quant.data.frames import Floats
from quant.hmm.filter import forward_pass
from quant.hmm.model import GaussianParams, ScalerParams


def smooth(
    params: GaussianParams, scaler: ScalerParams, rows: Floats, *, columns: tuple[str, ...]
) -> Floats:
    """Smoothed state posteriors, shape (n, K): row t is P(s_t | x_1..x_n), summing to 1.

    Run `forward_pass` over `rows` (n, d), whose columns `columns` names, for log alpha and the emissions log b(z_t), then the
    backward recursion in log space: log β_{n-1} = 0 and

        log β_t,j = logsumexp_k(log a_jk + e_{t+1},k + log β_{t+1},k) - c_t,

    with e_{t+1} = log b(z_{t+1}) - max_k log b_k(z_{t+1}), the emissions centered as the filter
    centers them, and c_t the largest entry of the logsumexp, so β stays at most 0. Centering
    first keeps an outlier's emissions (about -5e17) from rounding log a_jk + log β away. Any
    constant per bar cancels: the posterior is exp(log p_t + log β_t), renormalized to sum to
    1. Raises as `forward_pass` does.
    """
    trace = forward_pass(params, scaler, rows, columns=columns)
    with np.errstate(divide="ignore"):
        log_a = np.log(params.transmat)
    beta = np.zeros(params.n_states, dtype=np.float64)
    result = np.empty_like(trace.state_probs)
    for t in range(len(rows) - 1, -1, -1):
        joint = trace.log_probs[t] + beta
        joint -= np.max(joint)
        result[t] = np.exp(joint - logsumexp(joint))
        if t:
            emissions = trace.emissions[t] - np.max(trace.emissions[t])
            beta = logsumexp(log_a + (emissions + beta), axis=1)
            peak = np.max(beta)
            if not np.isfinite(peak):
                raise ModelError("the backward recursion found no path")
            beta = beta - peak
    return result
