"""Auto-labels from v and d, and the label view of the filter's state probabilities (spec §4, §6).

**Labeling a model** (once, after training, on the fit window). For each hidden state k, with
r_t the raw log return of bar t (ln C_t / C_{t-1}, not a feature) and p_tk the *filtered*
probability of state k at bar t (`hmm.filter.forward_pass` over the same bars):

- w_k = Σ_t p_tk, the state's weight;
- μ_k = Σ_t p_tk r_t / w_k, its weighted mean return;
- sigma_k = sqrt(Σ_t p_tk (r_t - μ_k)² / w_k), its weighted volatility (population form);
- sigma_all = sqrt(mean((r - mean(r))²)), the volatility of every bar, unweighted (population form).

Both are annualized with N = BARS_PER_YEAR bars a year: a mean scales by N and a volatility by
sqrt(N). So v = sigma_k sqrt(N) / (sigma_all sqrt(N)) = sigma_k / sigma_all, and
d = (μ_k N) / (sigma_k sqrt(N)) = sqrt(N) μ_k / sigma_k. The rules are then checked in order, with the
thresholds from config (`LabelThresholds`):

| Order | Label   | Rule                                        |
|-------|---------|---------------------------------------------|
| 1     | CRASH   | v ≥ crash_v and d < 0                       |
| 2     | STRESS  | v ≥ stress_v, or d ≤ stress_d               |
| 3     | CALM_UP | v < stress_v and d ≥ calm_d                 |
| 4     | CHOP    | everything else                             |

Several states can share a label, and a model with no crash-like state has no CRASH label.

**The label view** (every bar). A label's probability is the sum of its states' probabilities.
The normalized label entropy is H = -Σ_l P_l ln P_l / ln L over the model's L labels (terms
with P_l = 0 count 0), so H is 1 when the model's labels are equally likely whatever L is, and
0 for a model with one label. The active label is the most probable one. Its expected remaining
duration is the expected number of bars, the current one included, before the chain first
leaves the label's states, starting from the filtered distribution within them. For a label of
one state k that is 1 / (1 - a_kk), spec §4's expected duration.

Everything here is pure float64 numpy. Nothing imports hmmlearn or eval (K3).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Self

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from quant.core.errors import ModelError
from quant.core.model import StrictModel
from quant.core.regime import Label

Floats = npt.NDArray[np.float64]

# Annualization: 252 sessions a year of 7 bars each (six hourly bars and the closing half
# hour). Half days are ignored. Only d depends on it, since it cancels from v.
BARS_PER_YEAR: Final = 252 * 7


class LabelThresholds(StrictModel):
    """The rule thresholds, fixed in config before any out-of-sample run (spec §4). CRASH's
    `d < 0` is a sign test, not a threshold, so it isn't configurable."""

    crash_v: float
    stress_v: float
    stress_d: float
    calm_d: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Return self when 0 < stress_v < crash_v and stress_d < 0 < calm_d. Otherwise raise
        ValueError naming the values, which pydantic reports as a ValidationError."""
        if not (0 < self.stress_v < self.crash_v and self.stress_d < 0 < self.calm_d):
            raise ValueError(f"unordered label thresholds: {self.model_dump()}")
        return self


def r1_thresholds() -> LabelThresholds:
    """Spec §4's thresholds, which r1 keeps (PLAN §5): crash_v 1.75, stress_v 1.25, stress_d
    -0.5 and calm_d 0.5. A new, validated object on each call."""
    return LabelThresholds(crash_v=1.75, stress_v=1.25, stress_d=-0.5, calm_d=0.5)


class StateLabel(StrictModel):
    """One hidden state's row of the label table: its v and d (finite floats, annualized as in
    the module docstring) and the label in force for it.

    For a first model the label is `classify(v, d, thresholds)`. After a refit a state inherits
    its matched predecessor's label (spec §5, built in B7), which can differ from the rule's;
    that difference is a drift signal, so the table stores the label rather than deriving it.
    """

    v: float
    d: float
    label: Label


class LabelTable(StrictModel):
    """The label map of one model: `states[k]` is hidden state k's row, and `thresholds` the
    rule thresholds it was derived under. Stored in the model artifact's meta.json."""

    thresholds: LabelThresholds
    states: tuple[StateLabel, ...]

    def label_of(self, state: int) -> Label:
        """The label in force for hidden state `state`. IndexError outside 0 ≤ state < K."""
        if not 0 <= state < len(self.states):
            raise IndexError(state)
        return self.states[state].label

    @property
    def labels(self) -> tuple[Label, ...]:
        """The distinct labels the table uses, in rule order (`tuple(Label)` order). A table
        labeled [CHOP, CALM_UP, CHOP] gives (CALM_UP, CHOP)."""
        used = {state.label for state in self.states}
        return tuple(label for label in Label if label in used)

    def states_of(self, label: Label) -> tuple[int, ...]:
        """The hidden states with `label`, ascending; () for a label the table doesn't use."""
        return tuple(k for k, state in enumerate(self.states) if state.label == label)


def classify(v: float, d: float, thresholds: LabelThresholds) -> Label:
    """The first rule of the module docstring's table that holds for (v, d), checked in order,
    with the inequalities exactly as written: v = crash_v with d < 0 is CRASH, v = stress_v is
    STRESS, d = stress_d is STRESS, d = calm_d with v < stress_v is CALM_UP. Raises ModelError
    when v or d isn't finite, or v < 0."""
    if not np.isfinite(v) or not np.isfinite(d) or v < 0:
        raise ModelError("label statistics must be finite, with v >= 0")
    if v >= thresholds.crash_v and d < 0:
        return Label.CRASH
    if v >= thresholds.stress_v or d <= thresholds.stress_d:
        return Label.STRESS
    if v < thresholds.stress_v and d >= thresholds.calm_d:
        return Label.CALM_UP
    return Label.CHOP


def state_stats(returns: Floats, state_probs: Floats) -> tuple[Floats, Floats]:
    """(v, d), each a float64 array of shape (K,), from the formulas in the module docstring.

    `returns` has shape (n,) and `state_probs` shape (n, K), row t being bar t's filtered
    probabilities. Raises ModelError when the shapes don't agree, n < 2, any value isn't finite,
    a probability is negative, sigma_all is 0, or a state has w_k = 0 or sigma_k = 0 (its d is
    undefined).
    """
    if (
        not isinstance(returns, np.ndarray)
        or not isinstance(state_probs, np.ndarray)
        or returns.dtype != np.float64
        or state_probs.dtype != np.float64
        or returns.ndim != 1
        or state_probs.ndim != 2
        or len(returns) < 2
        or state_probs.shape[0] != len(returns)
        or state_probs.shape[1] < 1
        or not np.isfinite(returns).all()
        or not np.isfinite(state_probs).all()
        or (state_probs < 0).any()
    ):
        raise ModelError("state statistics need finite returns and matching probabilities")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        weights = state_probs.sum(axis=0)
        mu = (state_probs * returns[:, None]).sum(axis=0) / weights
        sigma = np.sqrt((state_probs * (returns[:, None] - mu) ** 2).sum(axis=0) / weights)
        sigma_all = returns.std()
        v = sigma / sigma_all
        d = np.sqrt(BARS_PER_YEAR) * mu / sigma
    if (weights <= 0).any() or (sigma <= 0).any() or sigma_all <= 0:
        raise ModelError("state statistics need positive weight and volatility")
    if not np.isfinite(v).all() or not np.isfinite(d).all():
        raise ModelError("state statistics overflow float64")
    return v, d


def label_states(returns: Floats, state_probs: Floats, thresholds: LabelThresholds) -> LabelTable:
    """The label table of a first model: `state_stats`, then `classify` per state. Raises as
    `state_stats` does."""
    v, d = state_stats(returns, state_probs)
    return LabelTable(
        thresholds=thresholds,
        states=tuple(
            StateLabel(v=float(vi), d=float(di), label=classify(float(vi), float(di), thresholds))
            for vi, di in zip(v, d, strict=True)
        ),
    )


def label_probs(state_probs: Floats, table: LabelTable) -> dict[Label, float]:
    """Each of the table's labels (keys in `table.labels` order) mapped to the sum of its
    states' probabilities, added in ascending state order. Labels the table doesn't use are
    absent: their probability is 0. Raises ModelError when `state_probs` isn't shape (K,) for
    the table's K."""
    if state_probs.shape != (len(table.states),):
        raise ModelError("state probabilities must have one entry per state")
    return {
        label: sum(float(state_probs[k]) for k in table.states_of(label)) for label in table.labels
    }


def label_entropy(probs: Mapping[Label, float], table: LabelTable) -> float:
    """H = -Σ P_l ln P_l / ln L over the table's L labels, with 0 ln 0 = 0, clipped to [0, 1]
    against rounding. 0.0 when L = 1. `probs` is a `label_probs` result for the same table."""
    if len(table.labels) <= 1:
        return 0.0
    entropy = -sum(p * np.log(p) for label in table.labels if (p := probs[label]) > 0)
    return float(np.clip(entropy / np.log(len(table.labels)), 0.0, 1.0))


def top_label(probs: Mapping[Label, float]) -> Label:
    """The most probable label. A tie goes to the label earliest in rule order (CRASH first),
    the most cautious. Raises ModelError for an empty mapping."""
    if not probs:
        raise ModelError("an empty label distribution has no active label")
    return max((label for label in Label if label in probs), key=probs.__getitem__)


def remaining_duration(
    state_probs: Floats, transmat: Floats, table: LabelTable, label: Label
) -> float | None:
    """The expected remaining duration of `label`, in bars, the current bar included (module
    docstring): with S = table.states_of(label), w = state_probs[S] / Σ state_probs[S] and A_SS
    the transition matrix restricted to S, it is w · x where (I - A_SS) x = 1 (numpy.linalg.solve).
    A label of one state k gives 1 / (1 - a_kk).

    None when S is every state: the chain can never leave the label. Raises ModelError when
    `label` isn't in the table, Σ state_probs[S] is 0, the system is singular (a closed subset
    of S), or the result isn't finite.
    """
    states = table.states_of(label)
    if not states:
        raise ModelError(f"model has no {label} states")
    mass = state_probs[list(states)]
    if mass.sum() <= 0:
        raise ModelError(f"{label} has no probability mass")
    if len(states) == len(table.states):
        return None
    sub = transmat[np.ix_(states, states)]
    try:
        duration = float(
            (mass / mass.sum()) @ np.linalg.solve(np.eye(len(states)) - sub, np.ones(len(states)))
        )
    except np.linalg.LinAlgError as exc:
        raise ModelError(f"{label} contains a closed subset") from exc
    if not np.isfinite(duration):
        raise ModelError(f"{label} duration is not finite")
    return duration
