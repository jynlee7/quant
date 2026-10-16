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
from quant.hmm.filter import ForwardTrace, forward_pass
from quant.hmm.model import GaussianParams, ScalerParams


def smooth(
    params: GaussianParams, scaler: ScalerParams, rows: Floats, *, columns: tuple[str, ...]
) -> Floats:
    """Smoothed state posteriors, shape (n, K): row t is P(s_t | x_1..x_n), summing to 1.

    `smooth_trace(params, forward_pass(params, scaler, rows, columns=columns))`: the forward
    pass over `rows` (n, d), whose columns `columns` names, then the backward recursion. Raises
    as `forward_pass` does.
    """
    return smooth_trace(params, forward_pass(params, scaler, rows, columns=columns))


def smooth_trace(params: GaussianParams, trace: ForwardTrace) -> Floats:
    """Smoothed state posteriors, shape (n, K), from `trace`, the forward pass `forward_pass`
    made under `params` over n rows: log alpha (`trace.log_probs`) and the emissions log b(z_t)
    (`trace.emissions`). A caller that needs the filter's p_t as well runs the forward pass
    once and hands its trace here (`quant.eval.calibration.model_series`).

    The backward recursion runs in log space: log β_{n-1} = 0 and

        log β_t,j = logsumexp_k(log a_jk + e_{t+1},k + log β_{t+1},k) - c_t,

    with e_{t+1} = log b(z_{t+1}) - max_k log b_k(z_{t+1}), the emissions centered as the filter
    centers them, and c_t the largest entry of the logsumexp, so β stays at most 0. Centering
    first keeps an outlier's emissions (about -5e17) from rounding log a_jk + log β away. Any
    constant per bar cancels: the posterior is exp(log p_t + log β_t), renormalized to sum to
    1. Raises ModelError when the trace's shapes don't fit `params`, or the backward recursion
    finds no path.
    """
    k = params.n_states
    n = len(trace.log_probs)
    if trace.log_probs.shape != (n, k) or trace.emissions.shape != (n, k):
        raise ModelError("the forward trace doesn't fit the model's states")
    with np.errstate(divide="ignore"):
        log_a = np.log(params.transmat)
    beta = np.zeros(k, dtype=np.float64)
    result = np.empty_like(trace.state_probs)
    for t in range(n - 1, -1, -1):
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
