"""Health, rule 6 (spec §8, C2, C4, C6).

**An unhealthy bar** is a decision whose inputs can't be trusted. Its `Cause`:

- FEATURES: a FeatureError from `features_for` or `indicators_for` (a missing, stale or
  incomplete decision bar, a warm-up row, a NaN feature, a bad bar);
- MODEL: any other ModelError from the scaler or the filter;
- MODEL_PROVENANCE: the governing model's `fit_end >= tau` (T-NL2: its fit window must end before
  the decision);
- CALIBRATION_PROVENANCE: the calibration's `as_of >= tau` (`Calibration.require_before`);
- RELEASE: the release fails to load (a ConfigError from `quant.decide.playbook.load_release`);
- MISSED: a decision the engine missed while down, replayed on restart without trading (C6).

**Effect.** On an unhealthy bar the target position is 0 and entries are blocked, on that bar:
rules 4-6 never wait for hysteresis. Normal operation resumes on the `recovery_bars`-th (r1: the
second) consecutive healthy bar: unhealthy at t, healthy at t+1 (still not operating: target 0,
entries blocked), healthy at t+2 (operating). The streak is capped at `recovery_bars`, so the
state stays fixed-size.

**Cold start** (C4, design call 5) begins with a full healthy streak; the warm-up's rows are
unhealthy anyway (`features_for` raises for them).

**Drift freeze is not rule 6**: it blocks entries and increases only, while exits and stops keep
working (C5). It arrives in decide as a flag, never as a Cause.

Everything here is pure. The switching state (quant.decide.switching) carries the streak.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Cause(StrEnum):
    """Why a bar is unhealthy (module docstring). The record names it."""

    FEATURES = "features"
    MODEL = "model"
    MODEL_PROVENANCE = "model_provenance"
    CALIBRATION_PROVENANCE = "calibration_provenance"
    RELEASE = "release"
    MISSED = "missed"


@dataclass(frozen=True)
class Health:
    """One bar's verdict under rule 6.

    - `cause`: None for a healthy bar;
    - `streak`: consecutive healthy bars ending at this one (0 on an unhealthy bar), capped at
      the recovery count;
    - `operating`: normal operation on this bar: healthy and `streak` >= the recovery count.
      When False the target is 0 and entries are blocked.
    """

    cause: Cause | None
    streak: int
    operating: bool

    @property
    def healthy(self) -> bool:
        """`cause is None`."""
        raise NotImplementedError


def cause_of(error: BaseException) -> Cause:
    """The Cause a decision-time error stands for: FEATURES for a FeatureError, MODEL for any
    other ModelError, RELEASE for a ConfigError. Any other exception is a bug, not a health
    failure: it is re-raised as it is."""
    raise NotImplementedError


def provenance_cause(
    tau: datetime, *, model_fit_end: datetime, calibration_as_of: datetime | None
) -> Cause | None:
    """MODEL_PROVENANCE when `model_fit_end >= tau`, else CALIBRATION_PROVENANCE when a
    calibration is given and `calibration_as_of >= tau`, else None (no calibration at all is
    healthy: sizing then uses C13's failed cap). ClockError for a naive datetime."""
    raise NotImplementedError


def health_step(streak: int, cause: Cause | None, *, recovery_bars: int) -> Health:
    """Rule 6 on one bar, from the previous bar's `streak`: an unhealthy bar (a cause) gives
    streak 0, not operating; a healthy one gives `min(streak + 1, recovery_bars)`, operating
    when that is `recovery_bars`. ValueError, its message naming `recovery_bars` or `streak`,
    for `recovery_bars < 1` or a streak outside [0, recovery_bars]. The cold streak is
    `recovery_bars` (C4), so a run's first healthy bar operates."""
    raise NotImplementedError
