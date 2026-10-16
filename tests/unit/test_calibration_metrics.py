"""B7b: the calibration metrics and isotonic recalibration (spec §9, A4; C13).

Each metric is checked against the formula written out with plain numpy and `math.fsum`
(tests.support.calibration), and the pass rule at its exact edges: ECE ≤ 0.05 on the decimal
values (exactly 0.05 passes, the next float fails) and Brier skill strictly above 0 (a forecast
equal to the base rate has skill exactly 0 and fails). PLAN B7's red tests: a well-calibrated
synthetic forecaster passes; a miscalibrated one fails, and isotonic recalibration fixes it.
The recalibration is judged out of sample: knots fit on the older half, scored on the newer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant.core.calibration import BINS, interpolate, r1_calibration_config
from quant.core.errors import ModelError
from quant.core.regime import Label
from tests.support.calibration import forecaster, ref_bin, ref_isotonic, ref_metrics

CONFIG = r1_calibration_config()


def _metrics(forecast: np.ndarray, outcome: np.ndarray) -> object:
    from quant.eval.calibration import label_metrics

    return label_metrics(forecast, outcome, config=CONFIG)


# --- the config -----------------------------------------------------------------------------


@pytest.mark.spec("SPEC§9", "C13")
def test_r1_calibration_config_holds_the_spec_numbers() -> None:
    config = r1_calibration_config()
    assert (config.min_bars, config.settle_bars, config.outcome_bars) == (1000, 35, 1000)
    assert (config.ece_max, config.outcome_level) == (0.05, 0.90)
    assert (config.coverage_low, config.coverage_high) == (0.876, 0.924)
    assert config.failed_cap == 0.25
    assert BINS == 10
    assert r1_calibration_config() is not config  # a new object on each call


@pytest.mark.spec("SPEC§9", "C13")
@pytest.mark.parametrize(
    "update",
    [
        {"min_bars": 1},
        {"settle_bars": -1},
        {"ece_max": 0.0},
        {"outcome_bars": 0},
        {"outcome_level": 1.0},
        {"coverage_low": 0.95},
        {"coverage_high": 0.85},
        {"failed_cap": 0.3},
    ],
)
def test_a_calibration_config_out_of_order_is_refused(update: dict[str, object]) -> None:
    from pydantic import ValidationError

    from quant.core.calibration import CalibrationConfig

    with pytest.raises(ValidationError):
        CalibrationConfig(**{**r1_calibration_config().model_dump(), **update})


# --- the metrics ----------------------------------------------------------------------------


@pytest.mark.spec("SPEC§9", "A4")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_each_metric_is_the_formula(seed: int) -> None:
    from quant.eval.calibration import brier_score, expected_calibration_error, reliability

    forecast, outcome = forecaster(3000, seed, distort=lambda p: p**1.5)
    ref = ref_metrics(forecast, outcome)
    metrics = _metrics(forecast, outcome)
    assert metrics.n == 3000
    assert metrics.base_rate == ref["base_rate"]
    assert metrics.brier == ref["brier"] == brier_score(forecast, outcome)
    assert metrics.reference == ref["reference"]
    assert metrics.skill == ref["skill"]
    assert metrics.ece == ref["ece"] == expected_calibration_error(forecast, outcome)
    curve = reliability(forecast, outcome)
    assert metrics.reliability == curve
    assert [b.count for b in curve] == ref["counts"]
    assert [(b.lower, b.upper) for b in curve] == [(b / 10, (b + 1) / 10) for b in range(10)]
    y = outcome.astype(np.float64)
    for b, row in enumerate(curve):
        inside = np.array([ref_bin(p) == b for p in forecast])
        if row.count:
            assert row.forecast == math.fsum(forecast[inside]) / row.count
            assert row.observed == math.fsum(y[inside]) / row.count
    assert metrics.passed == ref["passed"]


@pytest.mark.spec("SPEC§9", "A4")
@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0, 0),
        (0.09999999999999999, 0),
        (0.1, 1),
        (0.29999999999999993, 2),
        (0.3, 3),  # 0.1 + 0.2's neighbour: repr "0.3"; floor(0.3 * 10) happens to agree
        (0.7, 7),
        (0.8999999999999999, 8),
        (0.9, 9),
        (1.0, 9),
    ],
)
def test_reliability_bins_compare_the_decimal_value_of_each_forecast(
    p: float, expected: int
) -> None:
    from quant.eval.calibration import bin_of

    assert bin_of(p) == expected


@pytest.mark.spec("SPEC§9")
@pytest.mark.parametrize("p", [-0.1, 1.0000001, math.nan, math.inf])
def test_a_forecast_outside_zero_to_one_has_no_bin(p: float) -> None:
    from quant.eval.calibration import bin_of

    with pytest.raises(ModelError):
        bin_of(p)


@pytest.mark.spec("SPEC§9")
def test_empty_bins_are_reported_empty_not_dropped() -> None:
    from quant.eval.calibration import reliability

    forecast = np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float64)
    outcome = np.array([False, False, True, True])
    curve = reliability(forecast, outcome)
    assert len(curve) == BINS
    assert [b.count for b in curve] == [2, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    assert all(b.forecast is None and b.observed is None for b in curve[1:9])
    assert (curve[0].forecast, curve[0].observed) == (0.05, 0.0)
    assert (curve[9].forecast, curve[9].observed) == (0.95, 1.0)


@pytest.mark.spec("SPEC§9", "A4")
def test_ece_exactly_at_its_threshold_passes_and_the_next_float_fails() -> None:
    # Ten bars forecast 0.1 that never hold, and ten forecast 1.0 that always do:
    # ECE = (|fsum(ten 0.1s) - 0| + 0) / 20 = 1.0 / 20, which is the float 0.05 exactly.
    outcome = np.array([False] * 10 + [True] * 10)
    at = np.array([0.1] * 10 + [1.0] * 10, dtype=np.float64)
    metrics = _metrics(at, outcome)
    assert metrics.ece == 0.05
    assert metrics.skill is not None
    assert metrics.skill > 0
    assert metrics.passed, "ECE equal to 0.05 is within ECE ≤ 0.05"

    above = at.copy()
    above[:10] = np.nextafter(0.1, 1.0)
    past = _metrics(above, outcome)
    assert past.ece > 0.05
    assert repr(past.ece) != "0.05"
    assert not past.passed, "ECE past 0.05 must fail"


@pytest.mark.spec("SPEC§9", "A4")
def test_a_forecast_of_the_base_rate_has_skill_exactly_zero_and_fails() -> None:
    _, outcome = forecaster(1000, 3)
    base = math.fsum(outcome.astype(np.float64)) / len(outcome)
    metrics = _metrics(np.full(len(outcome), base), outcome)
    assert metrics.skill == 0.0
    assert metrics.brier == metrics.reference
    assert not metrics.passed, "Brier skill must be strictly above 0"


@pytest.mark.spec("SPEC§9", "A4")
@pytest.mark.parametrize("held", [False, True], ids=["never", "always"])
def test_a_label_whose_target_never_varies_has_no_skill_and_fails(held: bool) -> None:
    # A label the model lacks (forecast 0, target never) or one always in force: the base rate
    # forecasts perfectly, the reference Brier score is 0 and skill can't be shown.
    n = 500
    forecast = np.full(n, 1.0 if held else 0.0)
    metrics = _metrics(forecast, np.full(n, held))
    assert metrics.reference == 0.0
    assert metrics.skill is None
    assert metrics.brier == 0.0
    assert metrics.ece == 0.0
    assert not metrics.passed


@pytest.mark.spec("SPEC§9", "C13")
def test_no_bars_give_empty_metrics_that_fail() -> None:
    metrics = _metrics(np.empty(0, dtype=np.float64), np.empty(0, dtype=np.bool_))
    assert metrics.n == 0
    assert (metrics.base_rate, metrics.brier, metrics.reference, metrics.skill, metrics.ece) == (
        None,
        None,
        None,
        None,
        None,
    )
    assert [b.count for b in metrics.reliability] == [0] * BINS
    assert not metrics.passed


@pytest.mark.spec("SPEC§9")
@pytest.mark.parametrize(
    "fault", ["lengths", "nan", "above one", "below zero", "float outcome", "2-D", "float32"]
)
def test_malformed_forecasts_are_refused(fault: str) -> None:
    forecast, outcome = forecaster(50, 4)
    outcome_arg: np.ndarray = outcome
    if fault == "lengths":
        outcome_arg = outcome[:-1]
    elif fault == "nan":
        forecast[3] = math.nan
    elif fault == "above one":
        forecast[3] = 1.5
    elif fault == "below zero":
        forecast[3] = -0.5
    elif fault == "float outcome":
        outcome_arg = outcome.astype(np.float64)
    elif fault == "2-D":
        forecast = forecast.reshape(10, 5)
        outcome_arg = outcome.reshape(10, 5)
    else:
        forecast = forecast.astype(np.float32)
    with pytest.raises(ModelError):
        _metrics(forecast, outcome_arg)


# --- PLAN B7's red tests --------------------------------------------------------------------


@pytest.mark.spec("SPEC§9", "A4", "C13")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_well_calibrated_synthetic_forecaster_passes(seed: int) -> None:
    from quant.eval.calibration import calibrate_label

    forecast, outcome = forecaster(4000, seed)
    verdict, knots = calibrate_label(
        Label.CALM_UP, forecast, outcome, config=CONFIG, sufficient=True
    )
    assert verdict.raw.passed, verdict.raw
    assert verdict.raw.ece is not None
    assert verdict.raw.ece < 0.03
    assert verdict.passed
    # A label that passes as it is isn't recalibrated.
    assert (verdict.recalibrated, verdict.knots, knots) == (None, 0, None)


@pytest.mark.spec("SPEC§9", "A4", "C13")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_miscalibrated_forecaster_fails_and_isotonic_recalibration_fixes_it(seed: int) -> None:
    from quant.eval.calibration import calibrate_label, fit_isotonic

    n = 8000
    forecast, outcome = forecaster(n, 100 + seed, distort=lambda p: p**3)
    verdict, knots = calibrate_label(
        Label.CALM_UP, forecast, outcome, config=CONFIG, sufficient=True
    )
    assert not verdict.raw.passed
    assert verdict.raw.ece is not None
    assert verdict.raw.ece > 0.2
    assert verdict.recalibrated is not None
    assert verdict.recalibrated.passed, verdict.recalibrated
    assert verdict.passed

    # The verdict is out of sample: knots fit on the older half short of its last settle_bars
    # (B7b review, finding D), scored on the newer half.
    half = n // 2
    fit = half - CONFIG.settle_bars
    early_x, early_y = ref_isotonic(forecast[:fit], outcome[:fit])
    mapped = np.interp(forecast[half:], early_x, early_y)
    assert verdict.recalibrated == _metrics(mapped, outcome[half:])
    assert verdict.recalibrated.n == n - half
    # The artifact's knots are fit on every scored bar.
    assert knots is not None
    ref_x, ref_y = ref_isotonic(forecast, outcome)
    assert np.array_equal(knots[0], ref_x)
    assert np.array_equal(knots[1], ref_y)
    assert verdict.knots == len(ref_x)
    fitted = fit_isotonic(forecast, outcome)
    assert np.array_equal(fitted[0], ref_x)
    assert np.array_equal(fitted[1], ref_y)


@pytest.mark.spec("SPEC§9", "A4")
@pytest.mark.parametrize("seed", [0, 1])
def test_a_recalibration_that_doesnt_hold_on_newer_bars_still_fails(seed: int) -> None:
    # The forecaster turns from too low to too high halfway: the map fit on the older half
    # makes the newer half worse. Scoring the map on the bars it was fit on would pass it.
    from quant.eval.calibration import calibrate_label

    n = 8000
    forecast, outcome = forecaster(n, 200 + seed)
    forecast = np.where(np.arange(n) < n // 2, forecast**3, forecast ** (1 / 3))
    verdict, knots = calibrate_label(
        Label.STRESS, forecast, outcome, config=CONFIG, sufficient=True
    )
    assert not verdict.raw.passed
    assert verdict.recalibrated is not None
    assert not verdict.recalibrated.passed
    assert not verdict.passed
    assert knots is not None
    assert verdict.knots == len(knots[0])


@pytest.mark.spec("SPEC§9", "A4")
def test_a_forecast_without_information_still_fails_after_recalibration() -> None:
    # A constant forecast: isotonic maps it to one constant, which can't beat the base rate.
    from quant.eval.calibration import calibrate_label

    _, outcome = forecaster(4000, 7)
    verdict, knots = calibrate_label(
        Label.CHOP, np.full(4000, 0.3), outcome, config=CONFIG, sufficient=True
    )
    assert not verdict.raw.passed
    assert verdict.recalibrated is not None
    assert verdict.recalibrated.skill is not None
    assert verdict.recalibrated.skill <= 0
    assert not verdict.passed
    assert knots is not None
    assert len(knots[0]) == 1 == verdict.knots


@pytest.mark.spec("SPEC§9", "C13")
def test_without_enough_bars_a_label_fails_and_isnt_recalibrated() -> None:
    from quant.eval.calibration import calibrate_label

    forecast, outcome = forecaster(4000, 0)
    verdict, knots = calibrate_label(
        Label.CALM_UP, forecast, outcome, config=CONFIG, sufficient=False
    )
    assert verdict.raw.passed  # a good forecast...
    assert not verdict.passed  # ...that can't count yet (C13)
    assert (verdict.recalibrated, verdict.knots, knots) == (None, 0, None)


@pytest.mark.spec("SPEC§9", "ARCH§7.4")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_knots_are_monotone_within_zero_and_one(seed: int) -> None:
    from quant.eval.calibration import fit_isotonic

    forecast, outcome = forecaster(2000, 300 + seed, distort=lambda p: 1 - (1 - p) ** 2)
    x, y = fit_isotonic(forecast, outcome)
    assert x.dtype == y.dtype == np.float64
    assert len(x) == len(y) >= 1
    assert (np.diff(x) > 0).all(), "knot positions strictly increase"
    assert (np.diff(y) >= 0).all(), "knot values never decrease"
    assert ((y >= 0) & (y <= 1)).all()
    assert x[0] >= forecast.min()
    assert x[-1] <= forecast.max()


@pytest.mark.spec("SPEC§9", "ARCH§7.4", "K3")
def test_sizings_interpolation_is_scikit_learns_isotonic_prediction() -> None:
    # Sizing maps P through the knots with quant.core.calibration.interpolate (no scikit-learn
    # on the decision path, K3): it must give scikit-learn's own predictions, clipping included.
    from sklearn.isotonic import IsotonicRegression

    forecast, outcome = forecaster(3000, 9, distort=lambda p: 0.2 + 0.6 * p)
    fit = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    fit.fit(forecast, outcome.astype(np.float64))
    x, y = ref_isotonic(forecast, outcome)
    probe = np.concatenate([np.linspace(0.0, 1.0, 1001), forecast[:200]])
    ours = interpolate(x, y, probe)
    np.testing.assert_allclose(ours, fit.predict(probe), rtol=0, atol=1e-12)
    assert ours[0] == y[0]
    assert ours[1000] == y[-1]
    assert np.array_equal(interpolate(x[:1], y[:1], probe), np.full(len(probe), y[0]))


@pytest.mark.spec("SPEC§9", "ARCH§7.4")
@pytest.mark.parametrize(
    "fault",
    ["x not increasing", "y decreasing", "y above one", "nan", "lengths", "empty", "value outside"],
)
def test_interpolation_refuses_knots_that_arent_an_isotonic_map(fault: str) -> None:
    x = np.array([0.1, 0.5, 0.9])
    y = np.array([0.0, 0.4, 1.0])
    values = np.array([0.2, 0.7])
    if fault == "x not increasing":
        x = np.array([0.1, 0.5, 0.5])
    elif fault == "y decreasing":
        y = np.array([0.0, 0.6, 0.4])
    elif fault == "y above one":
        y = np.array([0.0, 0.4, 1.2])
    elif fault == "nan":
        x = np.array([0.1, math.nan, 0.9])
    elif fault == "lengths":
        y = y[:2]
    elif fault == "empty":
        x, y = np.empty(0), np.empty(0)
    else:
        values = np.array([0.2, 1.5])
    with pytest.raises(ModelError):
        interpolate(x, y, values)
