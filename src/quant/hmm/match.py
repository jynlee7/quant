"""Label matching after a refit (spec §5, ARCHITECTURE §7.3). Built in B7.

A refit's states come out in whatever order EM found them. Each new state is paired with one
old state by the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`, minimizing the
total) on the symmetric KL divergence between their Gaussians, and inherits that old state's
label.

**Raw units** (design call, B7): the old and new models standardize with different scalers, so
their stored means and covariances are in different units. Both are un-standardized first
(`raw_gaussians`): state k's raw mean is μ_k · std + mean and its raw covariance Σ_k ⊙ (std
stdᵀ), elementwise, with the model's own scaler. A diag model's covariances stay diagonal; the
full (K, d, d) matrices are used either way.

**Symmetric KL** of two d-dimensional Gaussians a and b (`symmetric_kl`):

    KL(a‖b) + KL(b‖a) = ½ [tr(Σ_b⁻¹ Σ_a) + tr(Σ_a⁻¹ Σ_b)
                           + (μ_a - μ_b)ᵀ (Σ_a⁻¹ + Σ_b⁻¹) (μ_a - μ_b)] - d

(the log-determinants cancel). It is ≥ 0, symmetric, and 0 for equal Gaussians.

**Labels**: the new fit's rule-derived labels (B4's `label_states`, which `train_candidate`
puts in the model) are kept as `StateMatch.derived`, beside the inherited ones. A difference
between the two is one of the drift checks (`quant.hmm.drift`). The model a refit stores
carries the inherited labels (`inherit_labels`).

This module never imports hmmlearn or eval (K3): it reads stored parameters only.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant.data.frames import Floats
from quant.hmm.labels import LabelTable
from quant.hmm.model import GaussianParams, HmmModel, ScalerParams


@dataclass(frozen=True, eq=False)
class StateMatch:
    """The pairing of a refit's states with the previous model's.

    - `old_of_new` (K,): new state j plays old state `old_of_new[j]`; a permutation of
      0..K-1;
    - `divergence` (K, K): `divergence[i, j]` is the symmetric KL between old state i and new
      state j, in raw units (`divergence_matrix`);
    - `derived`: the new fit's rule-derived label table (the labels `train_candidate` gave it);
    - `inherited`: the table the stored model carries: state j keeps `derived`'s v and d and
      takes the label of old state `old_of_new[j]`; the thresholds are `derived`'s.

    `__post_init__` raises ModelError unless `old_of_new` is a tuple permutation of 0..K-1,
    `divergence` a finite float64 (K, K) array (stored read-only), and both tables have K
    states.
    """

    old_of_new: tuple[int, ...]
    divergence: Floats
    derived: LabelTable
    inherited: LabelTable

    def __post_init__(self) -> None:
        raise NotImplementedError("B7a: StateMatch checks")

    @property
    def new_of_old(self) -> tuple[int, ...]:
        """The inverse permutation: old state i is played by new state `new_of_old[i]`."""
        raise NotImplementedError("B7a: StateMatch.new_of_old")

    @property
    def cost(self) -> float:
        """The matched pairs' total divergence, math.fsum of divergence[old_of_new[j], j]."""
        raise NotImplementedError("B7a: StateMatch.cost")


def raw_gaussians(params: GaussianParams, scaler: ScalerParams) -> tuple[Floats, Floats]:
    """(means (K, d), covariances (K, d, d)) of every state in raw feature units: `params.means
    * scaler.std + scaler.mean` and `params.covars * numpy.outer(scaler.std, scaler.std)`,
    exactly those expressions (module docstring). ModelError when the scaler's width isn't the
    model's."""
    raise NotImplementedError("B7a: raw_gaussians")


def symmetric_kl(mean_a: Floats, cov_a: Floats, mean_b: Floats, cov_b: Floats) -> float:
    """KL(a‖b) + KL(b‖a) for two Gaussians with means (d,) and covariances (d, d) (module
    docstring). The inverses come from Cholesky factors (scipy.linalg). Raises ModelError for
    mismatched shapes, a value that isn't finite, a covariance that isn't positive definite,
    or a result that isn't finite."""
    raise NotImplementedError("B7a: symmetric_kl")


def divergence_matrix(old: HmmModel, new: HmmModel) -> Floats:
    """(K, K): entry [i, j] is `symmetric_kl` between old state i and new state j, both in raw
    units (`raw_gaussians` with each model's own scaler). Raises as `match_states` does."""
    raise NotImplementedError("B7a: divergence_matrix")


def match_states(old: HmmModel, new: HmmModel) -> StateMatch:
    """Pair `new`'s states with `old`'s (module docstring): `linear_sum_assignment` on
    `divergence_matrix(old, new)`, rows old and columns new, so new state j plays the old
    state assigned to column j. `derived` is `new.labels`; `inherited` is built from it and
    `old.labels` as StateMatch describes.

    A model whose states are a permutation of `old`'s Gaussians in raw units (whatever its
    scaler) is matched back exactly, labels included. Raises ModelError when the two models
    differ in K, in feature columns or in feature-config hash."""
    raise NotImplementedError("B7a: match_states")


def inherit_labels(new: HmmModel, match: StateMatch, *, previous: str) -> HmmModel:
    """The model a refit stores: `new` with `labels` = `match.inherited`, and its training
    record's `info` extended by one key, "match": {"previous": `previous` (the old model's
    artifact hash), "old_of_new": [...], "divergence": the K × K list of lists, "derived":
    [the rule-derived label of each new state, as its string value]}. Everything else (params,
    scaler, fit window, the other info entries, B4's per-state "label" among them, which is
    the rule-derived one) is unchanged. Raises ModelError when `match.derived` isn't
    `new.labels` (equal by value), `info` already has "match", or `previous` isn't 64
    lowercase hex characters."""
    raise NotImplementedError("B7a: inherit_labels")
