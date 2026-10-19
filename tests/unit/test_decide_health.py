"""B8a: health, rule 6 (spec §8, C2, C4, C6).

An unhealthy bar names its cause: a feature or model error, a model or calibration that isn't
strictly older than the decision (T-NL2's rule), a release that fails to load, or a missed
decision replayed after a restart. It stops normal operation on that bar, and normal operation
resumes on the second consecutive healthy bar. A cold start has a full healthy streak.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from quant.core.errors import ClockError, ConfigError, FeatureError, ModelError

RED = pytest.mark.xfail(strict=True, reason="red:B8", raises=NotImplementedError)

TAU = datetime(2018, 3, 6, 15, 46, tzinfo=UTC)


@RED
@pytest.mark.spec("SPEC§8")
def test_errors_map_to_causes() -> None:
    from quant.decide.health import Cause, cause_of

    assert cause_of(FeatureError("stale bar")) == Cause.FEATURES
    assert cause_of(ModelError("nan state")) == Cause.MODEL
    assert cause_of(ConfigError("bad playbook")) == Cause.RELEASE
    bug = ZeroDivisionError("a bug, not a health failure")
    with pytest.raises(ZeroDivisionError):
        cause_of(bug)


@RED
@pytest.mark.spec("SPEC§8", "SPEC§6")
def test_a_model_or_calibration_not_older_than_tau_is_unhealthy() -> None:
    from quant.decide.health import Cause, provenance_cause

    before = TAU - timedelta(microseconds=1)
    assert provenance_cause(TAU, model_fit_end=before, calibration_as_of=before) is None
    assert provenance_cause(TAU, model_fit_end=before, calibration_as_of=None) is None
    assert provenance_cause(TAU, model_fit_end=TAU, calibration_as_of=None) == (
        Cause.MODEL_PROVENANCE
    )
    assert provenance_cause(TAU, model_fit_end=before, calibration_as_of=TAU) == (
        Cause.CALIBRATION_PROVENANCE
    )
    # The model is checked first.
    late = TAU + timedelta(hours=1)
    assert provenance_cause(TAU, model_fit_end=late, calibration_as_of=late) == (
        Cause.MODEL_PROVENANCE
    )
    with pytest.raises(ClockError):
        provenance_cause(TAU.replace(tzinfo=None), model_fit_end=before, calibration_as_of=None)


@RED
@pytest.mark.spec("SPEC§8", "C2", "C6")
def test_operation_resumes_on_the_second_healthy_bar() -> None:
    from quant.decide.health import Cause, health_step

    streak = 2  # cold (C4)
    seen = []
    for cause in [None, Cause.FEATURES, None, None, None, Cause.MISSED, Cause.MISSED, None, None]:
        health = health_step(streak, cause, recovery_bars=2)
        streak = health.streak
        seen.append((health.healthy, health.operating, health.streak))
    assert seen == [
        (True, True, 2),
        (False, False, 0),  # unhealthy at t: target 0, entries blocked, on that bar
        (True, False, 1),  # healthy at t+1: still not operating
        (True, True, 2),  # healthy at t+2: resumes
        (True, True, 2),  # the streak is capped
        (False, False, 0),  # a missed decision replayed on restart is unhealthy (C6)
        (False, False, 0),
        (True, False, 1),
        (True, True, 2),
    ]


@RED
@pytest.mark.spec("SPEC§8", "C4")
def test_a_cold_start_operates_on_its_first_healthy_bar() -> None:
    from quant.decide.config import SwitchingConfig
    from quant.decide.health import health_step
    from quant.decide.switching import SwitchState
    from tests.support.decide import SWITCHING

    assert isinstance(SWITCHING, SwitchingConfig)
    cold = SwitchState.cold(SWITCHING)
    assert cold.health_streak == SWITCHING.health_recovery_bars == 2
    assert health_step(cold.health_streak, None, recovery_bars=2).operating


@RED
@pytest.mark.spec("SPEC§8")
@pytest.mark.parametrize(("streak", "recovery"), [(-1, 2), (3, 2), (0, 0)])
def test_health_step_refuses_impossible_arguments(streak: int, recovery: int) -> None:
    from quant.decide.health import health_step

    with pytest.raises(ValueError, match=r"streak|recovery_bars"):
        health_step(streak, None, recovery_bars=recovery)


@RED
@pytest.mark.spec("SPEC§8")
@pytest.mark.parametrize("recovery", [1, 3])
def test_the_recovery_count_comes_from_config(recovery: int) -> None:
    from quant.decide.health import Cause, health_step

    health = health_step(recovery, Cause.MODEL, recovery_bars=recovery)
    operating = []
    for _ in range(recovery):
        health = health_step(health.streak, None, recovery_bars=recovery)
        operating.append(health.operating)
    assert operating == [False] * (recovery - 1) + [True]
