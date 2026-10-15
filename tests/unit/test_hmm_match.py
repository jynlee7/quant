"""B7a: label matching after a refit (spec §5, ARCHITECTURE §7.3).

The Hungarian algorithm pairs old and new states by the symmetric KL divergence of their
Gaussians, compared in raw feature units, since the two models standardize with different
scalers. A model whose states are permuted is matched back exactly, inherited labels included.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pytest

from quant.core.errors import ModelError
from quant.core.regime import Label
from quant.hmm.model import GaussianParams, HmmModel, ScalerParams
from tests.support.hmm import (
    FEATURE_HASH,
    FIT_END,
    FIT_START,
    Truth,
    identity_columns,
    label_table,
    three_full,
    training_record,
    two_diag,
)

OLD_LABELS = {
    "three_full": [Label.CHOP, Label.CALM_UP, Label.STRESS],
    "two_diag": [Label.CALM_UP, Label.STRESS],
}
TRUTHS = {"three_full": three_full, "two_diag": two_diag}


def scaler(d: int, mean: list[float], std: list[float]) -> ScalerParams:
    return ScalerParams(
        identity_columns(d), np.array(mean, dtype=np.float64), np.array(std, dtype=np.float64)
    )


def model(truth: Truth, labels: list[Label], scale: ScalerParams) -> HmmModel:
    """An HmmModel whose states, in raw units under `scale`, are the truth's Gaussians: the
    stored means and covariances are the truth's standardized with `scale`."""
    std = scale.std
    covars = truth.covars / np.outer(std, std)
    covars = 0.5 * (covars + covars.transpose(0, 2, 1))
    params = GaussianParams(
        truth.covariance_type,  # type: ignore[arg-type]
        truth.startprob.copy(),
        truth.transmat.copy(),
        (truth.means - scale.mean) / std,
        covars,
    )
    return HmmModel(
        params=params,
        scaler=scale,
        fit_start=FIT_START,
        fit_end=FIT_END,
        feature_config_hash=FEATURE_HASH,
        labels=label_table(labels),
        training=training_record(),
    )


def permuted(truth: Truth, perm: tuple[int, ...]) -> Truth:
    """The same HMM with new state j being the truth's state perm[j]."""
    p = list(perm)
    return Truth(
        truth.covariance_type,
        truth.startprob[p],
        truth.transmat[np.ix_(p, p)],
        truth.means[p],
        truth.covars[p],
    )


def ref_symmetric_kl(ma: Any, ca: Any, mb: Any, cb: Any) -> float:
    """KL(a||b) + KL(b||a) from the textbook formula with explicit inverses."""
    ia, ib = np.linalg.inv(ca), np.linalg.inv(cb)
    dm = ma - mb
    d = len(ma)
    return float(0.5 * (np.trace(ib @ ca) + np.trace(ia @ cb) + dm @ (ia + ib) @ dm) - d)


CASES = [
    (name, perm)
    for name in ("three_full", "two_diag")
    for perm in itertools.permutations(range(TRUTHS[name]().k))
]


@pytest.mark.spec("SPEC§5", "ARCH§7.3")
@pytest.mark.parametrize(("name", "perm"), CASES)
def test_a_permuted_model_is_matched_back_exactly(name: str, perm: tuple[int, ...]) -> None:
    from quant.hmm.match import match_states

    truth = TRUTHS[name]()
    old = model(truth, OLD_LABELS[name], scaler(truth.d, [0.0] * truth.d, [1.0] * truth.d))
    # The refit standardizes with another scaler, and its rule-derived labels are all CHOP.
    new_scale = scaler(
        truth.d, [0.3 * (i + 1) for i in range(truth.d)], [1.7 + i for i in range(truth.d)]
    )
    new = model(permuted(truth, perm), [Label.CHOP] * truth.k, new_scale)

    match = match_states(old, new)
    assert match.old_of_new == perm
    assert match.new_of_old == tuple(int(i) for i in np.argsort(perm))
    assert [s.label for s in match.inherited.states] == [OLD_LABELS[name][p] for p in perm]
    assert [s.label for s in match.derived.states] == [Label.CHOP] * truth.k
    # The inherited table keeps the new fit's own v and d.
    for inherited, derived in zip(match.inherited.states, match.derived.states, strict=True):
        assert (inherited.v, inherited.d) == (derived.v, derived.d)
    assert match.cost == pytest.approx(0.0, abs=1e-9)


@pytest.mark.spec("SPEC§5")
def test_divergence_is_the_symmetric_kl_in_raw_units() -> None:
    from quant.hmm.match import divergence_matrix, raw_gaussians, symmetric_kl

    truth = three_full()
    old = model(truth, OLD_LABELS["three_full"], scaler(2, [0.5, -1.0], [2.0, 0.5]))
    other = Truth(
        "full",
        truth.startprob,
        truth.transmat,
        truth.means + np.array([[0.2, -0.1], [0.0, 0.3], [-0.4, 0.0]]),
        truth.covars * 1.3,
    )
    new = model(other, [Label.CHOP] * 3, scaler(2, [-0.2, 0.4], [0.7, 3.0]))

    means, covars = raw_gaussians(old.params, old.scaler)
    np.testing.assert_allclose(means, truth.means, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(covars, truth.covars, rtol=1e-12, atol=1e-12)

    matrix = divergence_matrix(old, new)
    assert matrix.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            expected = ref_symmetric_kl(
                truth.means[i], truth.covars[i], other.means[j], other.covars[j]
            )
            assert matrix[i, j] == pytest.approx(expected, rel=1e-9)
            assert symmetric_kl(
                truth.means[i], truth.covars[i], other.means[j], other.covars[j]
            ) == pytest.approx(expected, rel=1e-9)
    assert symmetric_kl(
        truth.means[0], truth.covars[0], truth.means[0], truth.covars[0]
    ) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.spec("SPEC§5")
@pytest.mark.parametrize("fault", ["states", "columns", "feature hash"])
def test_models_that_do_not_agree_are_not_matched(fault: str) -> None:
    from quant.hmm.match import match_states

    old = model(three_full(), OLD_LABELS["three_full"], scaler(2, [0.0, 0.0], [1.0, 1.0]))
    if fault == "states":
        new = model(two_diag(), OLD_LABELS["two_diag"], scaler(3, [0.0] * 3, [1.0] * 3))
    else:
        base = model(three_full(), OLD_LABELS["three_full"], scaler(2, [0.0, 0.0], [1.0, 1.0]))
        new = HmmModel(
            params=base.params,
            scaler=ScalerParams(("x", "y"), base.scaler.mean, base.scaler.std)
            if fault == "columns"
            else base.scaler,
            fit_start=base.fit_start,
            fit_end=base.fit_end,
            feature_config_hash="cd" * 32 if fault == "feature hash" else FEATURE_HASH,
            labels=base.labels,
            training=base.training,
        )
    with pytest.raises(ModelError):
        match_states(old, new)


@pytest.mark.spec("SPEC§5", "ARCH§7.2")
def test_the_stored_refit_carries_inherited_labels_and_records_the_match() -> None:
    from quant.hmm.match import inherit_labels, match_states

    truth = three_full()
    old = model(truth, OLD_LABELS["three_full"], scaler(2, [0.0, 0.0], [1.0, 1.0]))
    perm = (2, 0, 1)
    new = model(
        permuted(truth, perm),
        [Label.CHOP, Label.STRESS, Label.CHOP],
        scaler(2, [0.1, 0.2], [1.5, 0.8]),
    )
    match = match_states(old, new)
    previous = old.artifact_hash()
    stored = inherit_labels(new, match, previous=previous)

    assert stored.labels == match.inherited
    assert [s.label for s in stored.labels.states] == [Label.STRESS, Label.CHOP, Label.CALM_UP]
    for name, array in new.arrays().items():
        assert stored.arrays()[name].tobytes() == array.tobytes(), name
    assert (stored.fit_start, stored.fit_end) == (new.fit_start, new.fit_end)
    info = stored.training.info
    recorded = info["match"]
    assert recorded["previous"] == previous
    assert tuple(recorded["old_of_new"]) == perm
    assert recorded["derived"] == ("CHOP", "STRESS", "CHOP")
    np.testing.assert_array_equal(np.array(recorded["divergence"]), match.divergence)
    for key, value in new.training.info.items():
        assert info[key] == value
    assert stored.artifact_hash() != new.artifact_hash()

    with pytest.raises(ModelError):
        inherit_labels(new, match, previous="not a hash")
    with pytest.raises(ModelError):
        inherit_labels(stored, match, previous=previous)  # its labels aren't the derived ones


@pytest.mark.spec("SPEC§5")
@pytest.mark.parametrize("fault", ["not a permutation", "divergence shape", "table size", "nan"])
def test_a_state_match_checks_itself(fault: str) -> None:
    from quant.hmm.match import StateMatch

    table = label_table([Label.CHOP, Label.CALM_UP])
    pairs, divergence, other = (0, 1), np.zeros((2, 2)), table
    if fault == "not a permutation":
        pairs = (0, 0)
    elif fault == "divergence shape":
        divergence = np.zeros((2, 3))
    elif fault == "table size":
        other = label_table([Label.CHOP])
    else:
        divergence = np.array([[0.0, math.nan], [1.0, 0.0]])
    with pytest.raises(ModelError):
        StateMatch(old_of_new=pairs, divergence=divergence, derived=table, inherited=other)
