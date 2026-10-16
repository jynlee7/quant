"""Drift: the four checks at a refit, and the live monitor (spec §5, D6, ARCHITECTURE §7.5).
Built in B7.

**At a refit** (`refit_drift`), over the matched states (`quant.hmm.match.StateMatch`, new
state j playing old state o(j) = `old_of_new[j]`), any one of these is drift: it freezes new
entries and alerts (C5), while open positions keep their exits and stops.

- "transition": for some new states j, other, |a_new[j, other] - a_old[o(j), o(other)]| > `transition`;
- "duration": for some new state j, the expected duration 1 / (1 - a_kk) changes by more than
  `duration` times the old one: |D_new - D_old| > duration · D_old, with a = a_new[j, j] and
  b = a_old[o(j), o(j)];
- "mean": for some new state j and feature i, |m_j[i] - μ_old[o(j), i]| > `mean`, where m_j is
  new state j's mean in the *previous* model's standardized units: `((μ_new[j] · std_new +
  mean_new) - mean_old) / std_old`, exactly that expression, except that when the two scalers
  are equal (same columns, and mean and std equal element for element) μ_new[j] is used as it
  is;
- "label": for some new state j, the label it inherited differs from its rule-derived one
  (`match.inherited` against `match.derived`).

**Edges** (design call, B7; recorded in BUILD_LOG): spec §5 says "more than", so a value
exactly at a threshold is not drift, and the smallest one above it is. Every comparison is
made on the decimal values of the floats involved (`Decimal(repr(x))`, as B4's sanity checks,
`quant.hmm.select._exact`), in a decimal context of at least 60 digits so every step is exact:

- transition: |A - B| > T, with A, B the two probabilities' decimal values;
- duration: with u = 1 - B and w = 1 - A (exact), |D_new - D_old| > r · D_old is |u - w| > r · w
  for u, w ≥ 0 (multiply through by u·w): this form also covers an infinite duration (a = 1
  or b = 1) without dividing by zero. So b = 0.93, a = 0.86 (D 14.29 → 7.14, exactly half) is
  not drift, though the float computation says it is;
- mean: |M - U| > T on the decimal values of the converted mean m_j[i] (a float) and of
  μ_old[o(j), i].

PLAN B7's red-test line "each drift rule fires at its threshold and not just below it" means
exactly this: it fires just past the threshold and not at it.

**Live** (`LiveDriftMonitor`, ARCHITECTURE §7.5): the rolling mean of the last DRIFT_WINDOW
(35) predictive log-likelihoods l_t under the running model, from full windows only, each mean
`math.fsum(window) / DRIFT_WINDOW` (an exactly rounded sum, as `quant.hmm.select.
drift_threshold` computes the baseline). It fires when the mean is strictly below the model's
`validation.drift.threshold` from its meta.json (`training.info`). A model swap empties the
window, so the new model's l_t are never mixed with the old one's. Both sides are densities of
the raw rows, so no rescaling is needed (v1.6). The monitor only reports: the engine records
the freeze (`quant.store.ops.OpsStore.record_freeze`, FreezeCause.LIVE_DRIFT) and alerts (B13).

This module is on the decision path: it never imports hmmlearn or eval (K3).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import localcontext
from typing import Literal

import numpy as np
from pydantic import field_validator

from quant.core.errors import ModelError
from quant.core.model import StrictModel
from quant.data.frames import Floats
from quant.hmm.match import StateMatch
from quant.hmm.model import HmmModel
from quant.hmm.select import DRIFT_WINDOW, _exact

DriftRule = Literal["transition", "duration", "mean", "label"]


class DriftThresholds(StrictModel):
    """D6's thresholds for the refit drift checks (module docstring), config in code like
    `quant.hmm.labels.r1_thresholds()`.

    - `transition`: the largest allowed move of a matched transition probability;
    - `duration`: the largest allowed change of a matched expected duration, as a fraction of
      the old duration;
    - `mean`: the largest allowed move of a matched mean on any feature, in the previous
      model's standardized units.

    Every value must be finite and > 0 (ValueError, which pydantic reports as a
    ValidationError)."""

    transition: float
    duration: float
    mean: float

    @field_validator("transition", "duration", "mean")
    @classmethod
    def _positive(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("drift thresholds must be positive and finite")
        return value


def r1_drift_thresholds() -> DriftThresholds:
    """Spec §5's values, which r1 keeps (D6): transition 0.10, duration 0.5 (50%), mean 0.5. A
    new, validated object on each call."""
    return DriftThresholds(transition=0.10, duration=0.5, mean=0.5)


@dataclass(frozen=True)
class DriftFinding:
    """One check that fired.

    - `rule`: which check;
    - `new_state`: the new state j it concerns (for "transition", the row j);
    - `other`: "transition": the column other; "mean": the feature index i; None otherwise;
    - `detail`: a sentence naming the states, both values and the threshold, for the alert and
      the refit event's reasons, e.g. "transition 0->1 moved from 0.18 to 0.2800000000000001,
      more than 0.1".
    """

    rule: DriftRule
    new_state: int
    other: int | None
    detail: str


@dataclass(frozen=True, eq=False)
class DriftReport:
    """The refit drift checks' result.

    - `findings`: every check that fired, ordered by rule ("transition", "duration", "mean",
      "label"), then new state, then `other`;
    - `transition_moves` (K, K): |a_new[j, other] - a_old[o(j), o(other)]| as floats, for the report;
    - `duration_changes` (K,): |D_new - D_old| / D_old as floats (inf when exactly one of them
      is infinite, 0 when both are), for the report;
    - `mean_moves` (K, d): |m_j[i] - μ_old[o(j), i]| as floats, in the previous model's
      standardized units.

    The decisions are the decimal comparisons of the module docstring, never these floats.
    """

    findings: tuple[DriftFinding, ...]
    transition_moves: Floats
    duration_changes: Floats
    mean_moves: Floats

    @property
    def passed(self) -> bool:
        """No finding."""
        return not self.findings

    @property
    def reasons(self) -> tuple[str, ...]:
        """Each finding's `detail`, in order: what the refit event and the alert record."""
        return tuple(f.detail for f in self.findings)


def refit_drift(
    old: HmmModel, new: HmmModel, match: StateMatch, thresholds: DriftThresholds
) -> DriftReport:
    """The four refit checks of the module docstring, with `match` pairing `new`'s states with
    `old`'s. Only `new`'s params and scaler are read; its labels may be the inherited or the
    rule-derived table, since the label check reads `match`. Raises ModelError when the models
    differ in K or in feature columns, or `match` has another K."""
    k = new.params.n_states
    if (
        old.params.n_states != k
        or old.scaler.columns != new.scaler.columns
        or len(match.old_of_new) != k
    ):
        raise ModelError("incompatible drift models or match")
    pairs = list(match.old_of_new)
    oa, na = old.params.transmat[np.ix_(pairs, pairs)], new.params.transmat
    same = np.array_equal(old.scaler.mean, new.scaler.mean) and np.array_equal(
        old.scaler.std, new.scaler.std
    )
    means = (
        new.params.means
        if same
        else ((new.params.means * new.scaler.std + new.scaler.mean) - old.scaler.mean)
        / old.scaler.std
    )
    old_means = old.params.means[pairs]
    findings: list[DriftFinding] = []
    durations = np.zeros(k, dtype=np.float64)
    with localcontext() as context:
        context.prec = 80
        for j in range(k):
            for other in range(k):
                if abs(_exact(float(na[j, other])) - _exact(float(oa[j, other]))) > _exact(
                    thresholds.transition
                ):
                    findings.append(
                        DriftFinding(
                            "transition",
                            j,
                            other,
                            f"transition {j}->{other} moved from {oa[j, other]} to {na[j, other]}, more than {thresholds.transition}",
                        )
                    )
        for j in range(k):
            u, w = 1 - _exact(float(oa[j, j])), 1 - _exact(float(na[j, j]))
            durations[j] = (
                (0.0 if u == w else math.inf) if not u or not w else float(abs(u - w) / w)
            )
            if abs(u - w) > _exact(thresholds.duration) * w:
                findings.append(
                    DriftFinding(
                        "duration",
                        j,
                        None,
                        f"duration state {j}: self-transition {oa[j, j]} to {na[j, j]}, relative change exceeds {thresholds.duration}",
                    )
                )
        for j in range(k):
            for i in range(new.params.n_features):
                if abs(_exact(float(means[j, i])) - _exact(float(old_means[j, i]))) > _exact(
                    thresholds.mean
                ):
                    findings.append(
                        DriftFinding(
                            "mean",
                            j,
                            i,
                            f"mean state {j} feature {i} moved from {old_means[j, i]} to {means[j, i]}, more than {thresholds.mean}",
                        )
                    )
    for j, (derived, inherited) in enumerate(
        zip(match.derived.states, match.inherited.states, strict=True)
    ):
        if derived.label != inherited.label:
            findings.append(
                DriftFinding(
                    "label",
                    j,
                    None,
                    f"label state {j}: derived {derived.label}, inherited {inherited.label}",
                )
            )
    return DriftReport(tuple(findings), np.abs(na - oa), durations, np.abs(means - old_means))


@dataclass(frozen=True)
class LiveDriftReading:
    """One bar of the live monitor.

    - `loglik`: the bar's l_t;
    - `mean`: the rolling mean of the last DRIFT_WINDOW l_t under the running model, None
      until it has that many;
    - `threshold`: the running model's `validation.drift.threshold`;
    - `fired`: `mean is not None and mean < threshold`;
    - `model_hash`: the running model's artifact hash.
    """

    loglik: float
    mean: float | None
    threshold: float
    fired: bool
    model_hash: str


class LiveDriftMonitor:
    """The live drift check for the running model (module docstring). Mutable: `update`
    advances it, `swap` restarts it under another model."""

    def __init__(self, model: HmmModel) -> None:
        """A monitor with an empty window under `model`. Its threshold is
        `model.training.info["validation"]["drift"]["threshold"]`; raises ModelError when that
        is missing or not a finite float, or when the record's "window" isn't DRIFT_WINDOW."""
        self.swap(model)

    @property
    def model_hash(self) -> str:
        return self._model_hash

    @property
    def threshold(self) -> float:
        return self._threshold

    def update(self, loglik: float) -> LiveDriftReading:
        """Add the next bar's l_t (`hmm.filter.FilterOutput.loglik`) and read the monitor. The
        window keeps the last DRIFT_WINDOW values; the mean is `math.fsum(window) /
        DRIFT_WINDOW` once it is full. Raises ModelError, leaving the window unchanged, when
        `loglik` isn't a finite float."""
        if (
            not isinstance(loglik, (int, float))
            or isinstance(loglik, bool)
            or not math.isfinite(loglik)
        ):
            raise ModelError("loglik must be finite")
        self._window.append(float(loglik))
        mean = math.fsum(self._window) / DRIFT_WINDOW if len(self._window) == DRIFT_WINDOW else None
        return LiveDriftReading(
            loglik,
            mean,
            self.threshold,
            mean is not None and mean < self.threshold,
            self.model_hash,
        )

    def swap(self, model: HmmModel) -> None:
        """Run under `model` from now on: empty the window and take its threshold, as
        `__init__` does (and raising as it does, leaving the monitor unchanged)."""
        validation = model.training.info.get("validation")
        baseline = validation.get("drift") if isinstance(validation, Mapping) else None
        threshold = baseline.get("threshold") if isinstance(baseline, Mapping) else None
        if (
            not isinstance(baseline, Mapping)
            or baseline.get("window") != DRIFT_WINDOW
            or not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(threshold)
        ):
            raise ModelError("model needs a finite 35-bar drift baseline")
        model_hash = model.artifact_hash()
        self._threshold = float(threshold)
        self._model_hash = model_hash
        self._window: deque[float] = deque(maxlen=DRIFT_WINDOW)
