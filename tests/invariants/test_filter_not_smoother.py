"""T-NL3: our filter is not the smoother (spec §6, I2).

On synthetic HMM data, the filter's p_t differs from hmmlearn's smoothed posteriors at interior
bars and matches them at the last bar, where filtering and smoothing coincide. The same check
applied to eval.smoother fails, which shows it can tell a filter that secretly smooths. This is
the runtime defense behind the static I2 scan (BUILD_LOG B0, open items).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from tests.support.hmm import (
    DEFAULT_LABELS,
    TRUTHS,
    columns_of,
    hmmlearn_posteriors,
    identity_scaler,
    model_of,
    params_of,
    sample,
)

Floats = npt.NDArray[np.float64]


def looks_filtered(probs: Floats, smoothed: Floats) -> bool:
    """T-NL3's verdict: most interior bars differ from the smoothed posteriors (by more than
    1e-6) and some by a lot (0.05), and the last bar matches them within 1e-10."""
    gaps = np.abs(probs - smoothed).max(axis=1)
    interior = gaps[1:-1]
    return bool((interior > 1e-6).mean() > 0.5 and interior.max() > 0.05 and gaps[-1] <= 1e-10)


@pytest.mark.spec("T-NL3", "I2", "SPEC§6")
@pytest.mark.parametrize("name", sorted(TRUTHS))
def test_the_filter_differs_from_the_smoother_inside_and_matches_at_the_end(name: str) -> None:
    from quant.eval.smoother import smooth
    from quant.hmm.filter import ForwardFilter

    truth = TRUTHS[name]()
    _, x = sample(truth, 600, seed=301)
    theirs = hmmlearn_posteriors(truth, x)

    ours = np.array(
        [
            o.state_probs
            for o in ForwardFilter(model_of(truth, DEFAULT_LABELS[name])).run(
                x, columns=columns_of(x)
            )
        ]
    )
    assert looks_filtered(ours, theirs)

    # The check bites: smoothed posteriors fail it, because they match at every bar.
    smoothed = smooth(params_of(truth), identity_scaler(truth.d), x, columns=columns_of(x))
    np.testing.assert_allclose(smoothed, theirs, rtol=0, atol=1e-9)
    assert not looks_filtered(smoothed, theirs)
