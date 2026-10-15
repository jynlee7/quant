"""B7a: drift at a refit and the live drift monitor (spec §5, D6, ARCHITECTURE §7.5).

Each refit rule fires just past its threshold and not at it, compared on the decimal values of
the numbers ("more than", spec §5). The edge values are chosen so float arithmetic would get
them wrong: 0.28 - 0.18 is 0.10000000000000003 in floats, exactly 0.1 in decimals.

The live monitor's mean is an exactly rounded sum over full 35-bar windows, it fires strictly
below the model's threshold, and a model swap empties its window.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from quant.core.errors import ModelError
from quant.core.regime import Label
from quant.hmm.model import GaussianParams, HmmModel, ScalerParams, TrainingRecord
from tests.support.hmm import (
    FEATURE_HASH,
    FIT_END,
    FIT_START,
    identity_columns,
    identity_scaler,
    label_table,
)

RED = pytest.mark.xfail(strict=True, reason="red:B7", raises=NotImplementedError)
LABELS = [Label.CALM_UP, Label.STRESS]

# The old model: two states over two features, identity scaler (so standardized = raw).
OLD_A = [[0.82, 0.18], [0.07, 0.93]]
OLD_MEANS = [[0.6, 0.0], [-1.0, 0.5]]


def _f(values: Any) -> Any:
    return np.array(values, dtype=np.float64)


def two_state(
    transmat: Any = OLD_A,
    means: Any = OLD_MEANS,
    *,
    scaler: ScalerParams | None = None,
    labels: list[Label] = LABELS,
    training: TrainingRecord | None = None,
) -> HmmModel:
    params = GaussianParams(
        "diag",
        _f([0.5, 0.5]),
        _f(transmat),
        _f(means),
        np.stack([np.diag(_f([1.0, 0.8])), np.diag(_f([0.6, 1.2]))]),
    )
    return HmmModel(
        params=params,
        scaler=scaler if scaler is not None else identity_scaler(2),
        fit_start=FIT_START,
        fit_end=FIT_END,
        feature_config_hash=FEATURE_HASH,
        labels=label_table(labels),
        training=training if training is not None else _record(-1.0),
    )


def _record(threshold: Any, window: int = 35) -> TrainingRecord:
    return TrainingRecord(
        base_seed=1,
        seeds=(1,),
        restart_logliks=(-100.0,),
        best=0,
        info={
            "validation": {"drift": {"window": window, "quantile": 0.01, "threshold": threshold}}
        },
    )


def identity_match(derived: list[Label] = LABELS, inherited: list[Label] = LABELS) -> Any:
    from quant.hmm.match import StateMatch

    return StateMatch(
        old_of_new=(0, 1),
        divergence=np.zeros((2, 2)),
        derived=label_table(derived),
        inherited=label_table(inherited),
    )


def _drift(new: HmmModel, match: Any = None) -> Any:
    from quant.hmm.drift import r1_drift_thresholds, refit_drift

    return refit_drift(two_state(), new, match or identity_match(), r1_drift_thresholds())


def _up(x: float) -> float:
    return float(np.nextafter(x, math.inf))


def _down(x: float) -> float:
    return float(np.nextafter(x, -math.inf))


@RED
@pytest.mark.spec("D6", "SPEC§5")
def test_r1_drift_thresholds_are_spec_five() -> None:
    import pydantic

    from quant.hmm.drift import DriftThresholds, r1_drift_thresholds

    thresholds = r1_drift_thresholds()
    assert (thresholds.transition, thresholds.duration, thresholds.mean) == (0.10, 0.5, 0.5)
    assert r1_drift_thresholds() is not thresholds
    for bad in (0.0, -0.1, math.inf):
        with pytest.raises(pydantic.ValidationError):
            DriftThresholds(transition=bad, duration=0.5, mean=0.5)


@RED
@pytest.mark.spec("SPEC§5", "D6")
def test_an_unchanged_refit_passes() -> None:
    report = _drift(two_state())
    assert report.passed
    assert report.findings == ()
    assert report.reasons == ()
    np.testing.assert_array_equal(report.transition_moves, np.zeros((2, 2)))
    np.testing.assert_array_equal(report.duration_changes, np.zeros(2))
    np.testing.assert_array_equal(report.mean_moves, np.zeros((2, 2)))


# Each rule's edge: (the new model at the threshold, the new model just past it).
EDGES = {
    # 0.18 -> 0.28 is exactly 0.1 in decimals (0.10000000000000003 in floats).
    "transition": (
        lambda: two_state([[0.72, 0.28], [0.07, 0.93]]),
        lambda: two_state([[1 - _up(0.28), _up(0.28)], [0.07, 0.93]]),
    ),
    # a_11 0.93 -> 0.86: the duration halves from 14.29 to 7.14 bars, exactly 50% in
    # decimals (the float computation says more); the transition moves only 0.07.
    "duration": (
        lambda: two_state([[0.82, 0.18], [0.14, 0.86]]),
        lambda: two_state([[0.82, 0.18], [1 - _down(0.86), _down(0.86)]]),
    ),
    # 0.6 -> 1.1 is exactly 0.5 in decimals (0.5000000000000001 in floats).
    "mean": (
        lambda: two_state(means=[[1.1, 0.0], [-1.0, 0.5]]),
        lambda: two_state(means=[[_up(1.1), 0.0], [-1.0, 0.5]]),
    ),
}


@RED
@pytest.mark.spec("SPEC§5", "D6")
@pytest.mark.parametrize("rule", sorted(EDGES))
def test_each_refit_drift_rule_fires_just_past_its_threshold_and_not_at_it(rule: str) -> None:
    at, past = EDGES[rule]
    assert _drift(at()).passed, f"{rule} fired exactly at its threshold"
    report = _drift(past())
    assert not report.passed
    assert {f.rule for f in report.findings} == {rule}
    assert report.reasons == tuple(f.detail for f in report.findings)
    assert all(f.new_state in (0, 1) for f in report.findings)


@RED
@pytest.mark.spec("SPEC§5", "D6")
def test_a_duration_that_becomes_infinite_is_drift() -> None:
    report = _drift(two_state([[0.82, 0.18], [0.0, 1.0]]))
    assert "duration" in {f.rule for f in report.findings}


@RED
@pytest.mark.spec("SPEC§5")
def test_a_label_that_differs_from_the_one_it_inherited_is_drift() -> None:
    assert _drift(two_state(), identity_match()).passed
    report = _drift(two_state(), identity_match(derived=[Label.CALM_UP, Label.CHOP]))
    assert [(f.rule, f.new_state) for f in report.findings] == [("label", 1)]


@RED
@pytest.mark.spec("SPEC§5")
def test_drift_compares_matched_states_not_indices() -> None:
    from quant.hmm.match import StateMatch

    # The new fit found the old states in the other order: no drift once matched.
    swapped = two_state([[0.93, 0.07], [0.18, 0.82]], [[-1.0, 0.5], [0.6, 0.0]])
    match = StateMatch(
        old_of_new=(1, 0),
        divergence=np.zeros((2, 2)),
        derived=label_table([Label.STRESS, Label.CALM_UP]),
        inherited=label_table([Label.STRESS, Label.CALM_UP]),
    )
    assert _drift(swapped, match).passed
    assert not _drift(swapped).passed  # paired by index, everything moved


@RED
@pytest.mark.spec("SPEC§5")
def test_mean_drift_is_measured_in_the_previous_models_standardized_units() -> None:
    old_scaler = ScalerParams(identity_columns(2), _f([0.0, 0.0]), _f([2.0, 1.0]))
    old = two_state(scaler=old_scaler)  # state 0's raw mean on feature 0 is 1.2
    new_scaler = ScalerParams(identity_columns(2), _f([1.0, 0.0]), _f([0.25, 1.0]))

    def new_with_raw(raw: float) -> HmmModel:
        means = [[(raw - 1.0) / 0.25, 0.0], [(-2.0 - 1.0) / 0.25, 0.5]]
        return two_state(means=means, scaler=new_scaler)

    from quant.hmm.drift import r1_drift_thresholds, refit_drift

    # Raw 1.2 -> 2.0 is 0.4 old standardized units (0.8 / 2): no drift, though it is 3.2 of
    # the new model's.
    report = refit_drift(old, new_with_raw(2.0), identity_match(), r1_drift_thresholds())
    assert report.passed, report.reasons
    assert report.mean_moves[0, 0] == pytest.approx(0.4)
    # Raw 1.2 -> 2.4 is 0.6 old standardized units: drift on feature 0 of state 0.
    report = refit_drift(old, new_with_raw(2.4), identity_match(), r1_drift_thresholds())
    assert [(f.rule, f.new_state, f.other) for f in report.findings] == [("mean", 0, 0)]


# --- The live monitor -------------------------------------------------------------------------


def _monitor(threshold: float) -> Any:
    from quant.hmm.drift import LiveDriftMonitor

    return LiveDriftMonitor(two_state(training=_record(threshold)))


@RED
@pytest.mark.spec("SPEC§5", "ARCH§7.5")
def test_the_live_monitor_reads_full_windows_only() -> None:
    monitor = _monitor(-1.0)
    for _ in range(34):
        reading = monitor.update(-1e6)  # far below the threshold, but no full window yet
        assert reading.mean is None
        assert not reading.fired
    reading = monitor.update(-1e6)
    assert reading.mean == -1e6
    assert reading.fired
    assert reading.threshold == -1.0
    # The window rolls: 35 values of 0 later, only the last 35 count.
    for _ in range(35):
        reading = monitor.update(0.0)
    assert reading.mean == 0.0
    assert not reading.fired


@RED
@pytest.mark.spec("ARCH§7.5")
def test_the_live_mean_is_an_exactly_rounded_sum_over_35() -> None:
    values = [1e16, 1.0, -1e16] + [0.0] * 32  # a naive running sum loses the 1.0
    expected = math.fsum(values) / 35
    assert expected == 1 / 35
    monitor = _monitor(-1.0)
    for value in values:
        reading = monitor.update(value)
    assert reading.mean == expected


@RED
@pytest.mark.spec("SPEC§5", "ARCH§7.5")
def test_the_live_monitor_fires_strictly_below_its_threshold() -> None:
    values = [-3.0 + 0.01 * i for i in range(35)]
    mean = math.fsum(values) / 35
    at = _monitor(mean)
    above = _monitor(_up(mean))
    for value in values:
        at_reading, above_reading = at.update(value), above.update(value)
    assert at_reading.mean == mean
    assert not at_reading.fired  # exactly at the threshold: not below it
    assert above_reading.fired


@RED
@pytest.mark.spec("SPEC§5", "ARCH§7.5", "SPEC§6")
def test_a_model_swap_resets_the_window() -> None:
    first = two_state(training=_record(-1.0))
    second = two_state(training=_record(-5.0), labels=[Label.CHOP, Label.STRESS])
    from quant.hmm.drift import LiveDriftMonitor

    monitor = LiveDriftMonitor(first)
    for i in range(35):
        assert monitor.update(-2.0).fired is (i == 34)  # fires once the window is full
    monitor.swap(second)
    assert monitor.model_hash == second.artifact_hash()
    assert monitor.threshold == -5.0
    for _ in range(34):
        reading = monitor.update(-10.0)
        assert reading.mean is None  # the old model's l_t are gone
    reading = monitor.update(-10.0)
    assert (reading.mean, reading.fired, reading.model_hash) == (
        -10.0,
        True,
        second.artifact_hash(),
    )


@RED
@pytest.mark.spec("ARCH§7.5")
@pytest.mark.parametrize("fault", ["no validation", "window", "threshold"])
def test_the_live_monitor_needs_the_models_drift_baseline(fault: str) -> None:
    from quant.hmm.drift import LiveDriftMonitor

    if fault == "no validation":
        record = TrainingRecord(base_seed=1, seeds=(1,), restart_logliks=(0.0,), best=0, info={})
    elif fault == "window":
        record = _record(-1.0, window=20)
    else:
        record = _record("low")
    with pytest.raises(ModelError):
        LiveDriftMonitor(two_state(training=record))


@RED
@pytest.mark.spec("ARCH§7.5")
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_the_live_monitor_refuses_a_loglik_that_isnt_finite(value: float) -> None:
    monitor = _monitor(-1.0)
    for _ in range(35):
        monitor.update(0.0)
    with pytest.raises(ModelError):
        monitor.update(value)
    assert monitor.update(0.0).mean == 0.0  # the window is as it was
