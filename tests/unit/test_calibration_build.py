"""B7b: building a calibration from a forecast series: C13's sample rules and the outcome check
(spec §9, A4; ARCHITECTURE §7.4).

PLAN B7's red tests: fewer than 1,000 out-of-sample bars count as failing, and the newest 35
bars are excluded. Pinned here: the 1,000 are counted after the 35 are left out; the outcome
check keeps the newest bars, uses the latest 1,000 pairs, and passes on exactly [87.6%, 92.4%]
(876 and 924 of 1,000 pass, 875 and 925 fail), both interval ends inclusive; a failed outcome
check caps every label at 25%; and a series holding a bar at or after the as-of time is refused.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from quant.core.calibration import CalibrationSource, r1_calibration_config
from quant.core.errors import ClockError, ModelError
from quant.core.regime import Label
from tests.support.calibration import (
    LABELS,
    MODEL_HASH,
    OTHER_HASH,
    SERIES_START,
    after,
    covered_share,
    exact,
    series,
)

RED = pytest.mark.xfail(strict=True, reason="red:B7", raises=NotImplementedError)
CONFIG = r1_calibration_config()
CALM, CHOP = LABELS.index(Label.CALM_UP), LABELS.index(Label.CHOP)


def _exact_series(n: int, *, covered: np.ndarray | None = None) -> object:
    forecast, target = exact(n)
    return series(
        n,
        label_forecast=forecast,
        target=target,
        covered=covered_share(n) if covered is None else covered,
    )


def _build(s: object, as_of: datetime, **kwargs: object) -> object:
    from quant.eval.calibration import build_calibration

    return build_calibration(
        s,
        as_of=as_of,
        source=kwargs.pop("source", CalibrationSource.BACKTEST),
        config=kwargs.pop("config", CONFIG),
    )


# --- C13 ------------------------------------------------------------------------------------


@RED
@pytest.mark.spec("C13", "SPEC§9", "A4")
def test_fewer_than_a_thousand_out_of_sample_bars_fail_every_label() -> None:
    # 1,034 bars leave 999 once the newest 35 are out: every label fails, however good.
    short = _build(_exact_series(1034), after(1034))
    document = short.document
    assert (document.bars, document.scored, document.sufficient) == (1034, 999, False)
    assert document.labels[CALM].raw.passed  # the forecast itself is fine...
    for label in Label:
        assert not short.passed(label), label  # ...but no verdict yet (C13)
        assert short.cap_fraction(label) == 0.25
        assert short.verdict(label).recalibrated is None
        assert short.verdict(label).knots == 0
    assert short.knots == {}

    # 1,035 bars leave 1,000: the verdicts count.
    enough = _build(_exact_series(1035), after(1035))
    assert (enough.document.bars, enough.document.scored, enough.document.sufficient) == (
        1035,
        1000,
        True,
    )
    assert enough.passed(Label.CALM_UP)
    assert enough.passed(Label.CHOP)
    assert enough.document.outcome.passed
    assert enough.cap_fraction(Label.CALM_UP) == enough.cap_fraction(Label.CHOP) == 1.0
    # CRASH and STRESS never hold: no skill can be shown, so they fail and are capped.
    for label in (Label.CRASH, Label.STRESS):
        assert not enough.passed(label)
        assert enough.cap_fraction(label) == 0.25


@RED
@pytest.mark.spec("C13", "SPEC§9", "A4")
def test_the_newest_35_bars_are_left_out_of_the_label_metrics() -> None:
    n = 1035
    base = _exact_series(n)
    as_of = after(n)
    reference = _build(base, as_of)

    # Rewriting the newest 35 bars' forecasts and targets changes no label verdict.
    forecast, target = exact(n)
    forecast[-35:] = 0.99
    target[-35:] = CHOP
    noisy = _build(
        series(n, label_forecast=forecast, target=target, covered=covered_share(n)), as_of
    )
    assert noisy.document.labels == reference.document.labels
    assert noisy.document.outcome == reference.document.outcome
    # The series differs, so the calibration's data hash, and with it its artifact hash, do.
    assert noisy.document.data_hash != reference.document.data_hash
    assert noisy.artifact_hash() != reference.artifact_hash()

    # The 36th newest bar is scored.
    forecast, target = exact(n)
    forecast[-36] = 0.99
    target[-36] = CHOP
    scored = _build(
        series(n, label_forecast=forecast, target=target, covered=covered_share(n)), as_of
    )
    assert scored.document.labels[CALM].raw != reference.document.labels[CALM].raw
    assert scored.document.labels[CALM].raw.n == reference.document.labels[CALM].raw.n == 1000


@RED
@pytest.mark.spec("C13", "SPEC§9")
def test_fewer_bars_than_the_settling_window_score_nothing() -> None:
    calibration = _build(_exact_series(20), after(20))
    document = calibration.document
    assert (document.bars, document.scored, document.sufficient) == (20, 0, False)
    assert all(v.raw.n == 0 and not v.passed for v in document.labels)


@RED
@pytest.mark.spec("C13", "SPEC§9", "ARCH§7.4")
def test_an_empty_series_is_a_calibration_in_which_every_label_fails() -> None:
    from quant.eval.calibration import ForecastSeries

    as_of = SERIES_START
    calibration = _build(ForecastSeries.empty(), as_of)
    document = calibration.document
    assert (document.bars, document.scored, document.sufficient, document.models) == (
        0,
        0,
        False,
        (),
    )
    assert document.outcome.n == 0
    assert not document.outcome.passed
    assert [v.label for v in document.labels] == list(Label)
    for label in Label:
        assert not calibration.passed(label)
        assert calibration.cap_fraction(label) == 0.25
        assert calibration.calibrated(label, 0.4) == 0.4  # no knots: P̃ = P


@RED
@pytest.mark.spec("C13", "SPEC§9")
def test_a_failing_label_with_enough_bars_is_recalibrated_on_the_scored_bars_only() -> None:
    n = 4035
    forecast, target = exact(n)
    forecast = np.where(forecast < 0.5, 0.35, 0.65)  # right order, far too timid
    calibration = _build(
        series(n, label_forecast=forecast, target=target, covered=covered_share(n)), after(n)
    )
    verdict = calibration.verdict(Label.CALM_UP)
    assert not verdict.raw.passed
    assert verdict.recalibrated is not None
    assert verdict.recalibrated.n == 2000
    assert verdict.passed
    x, y = calibration.knots[Label.CALM_UP]
    assert verdict.knots == len(x) == 2
    # Fit on the 4,000 scored bars: 5% of the 0.35s and 95% of the 0.65s were CALM_UP.
    assert (x[0], x[1]) == (0.35, 0.65)
    assert abs(y[0] - 0.05) < 1e-12
    assert abs(y[1] - 0.95) < 1e-12
    assert abs(calibration.calibrated(Label.CALM_UP, 0.65) - 0.95) < 1e-12


# --- the outcome check ----------------------------------------------------------------------


def _covering(k: int, pairs: int = 1000) -> np.ndarray:
    """pairs + 1 bars whose pairs are covered k times (bar 0's own value is never scored)."""
    covered = np.zeros(pairs + 1, dtype=np.bool_)
    covered[1 : k + 1] = True
    return covered


@RED
@pytest.mark.spec("C13", "SPEC§9", "A4")
@pytest.mark.parametrize(
    ("covered", "passes"),
    [(875, False), (876, True), (900, True), (924, True), (925, False)],
)
def test_the_outcome_check_passes_on_exactly_its_binomial_band(covered: int, passes: bool) -> None:
    from quant.eval.calibration import outcome_check

    check = outcome_check(series(1001, covered=_covering(covered)), config=CONFIG)
    assert (check.n, check.covered, check.level) == (1000, covered, 0.90)
    assert check.passed is passes


@RED
@pytest.mark.spec("C13", "SPEC§9")
def test_the_outcome_check_uses_the_latest_thousand_pairs() -> None:
    from quant.eval.calibration import outcome_check

    n = 1501
    covered = np.zeros(n, dtype=np.bool_)  # the oldest 500 pairs all miss...
    covered[501:] = covered_share(1000)  # ...and the latest 1,000 cover 90%
    check = outcome_check(series(n, covered=covered), config=CONFIG)
    assert (check.n, check.covered, check.passed) == (1000, 900, True)

    # 1,000 bars give 999 pairs: too few, whatever the coverage.
    few = outcome_check(series(1000, covered=covered_share(1000)), config=CONFIG)
    assert few.n == 999
    assert not few.passed


@RED
@pytest.mark.spec("C13", "SPEC§9")
def test_the_outcome_check_keeps_the_newest_bars() -> None:
    # The settling window is for smoothed labels; the realized F1 of the newest bars is final.
    from quant.eval.calibration import outcome_check

    n = 1035
    covered = covered_share(n)
    covered[-35:] = False
    check = outcome_check(series(n, covered=covered), config=CONFIG)
    assert check.n == 1000
    assert check.covered == 900 - int(covered_share(n)[-35:].sum())


@RED
@pytest.mark.spec("C13", "SPEC§9")
@pytest.mark.parametrize(
    ("realized", "covered"),
    [
        (1.0, True),
        (-1.0, True),
        (float(np.nextafter(1.0, 2.0)), False),
        (-1.0000000000000002, False),
    ],
    ids=["high end", "low end", "just above", "just below"],
)
def test_the_interval_includes_both_ends(realized: float, covered: bool) -> None:
    from quant.eval.calibration import ForecastSeries, outcome_check

    base = series(3)
    values = base.realized.copy()
    values[1] = realized
    s = ForecastSeries(
        base.models,
        base.available_at,
        base.forecast,
        base.target,
        base.smoothed,
        base.low,
        base.high,
        values,
    )
    check = outcome_check(s, config=CONFIG)
    assert check.n == 2
    assert check.covered == 1 + int(covered)  # pair (1, 2) is covered in any case


@RED
@pytest.mark.spec("C13", "SPEC§9", "A4")
def test_a_failed_outcome_check_caps_every_label() -> None:
    n = 1035
    calibration = _build(_exact_series(n, covered=np.ones(n, dtype=np.bool_)), after(n))
    assert calibration.document.outcome.covered == 1000  # 100% coverage: the interval is too wide
    assert not calibration.document.outcome.passed
    assert calibration.passed(Label.CALM_UP)  # the label itself passes...
    assert calibration.cap_fraction(Label.CALM_UP) == 0.25  # ...and is capped all the same


# --- the series and the as-of time ----------------------------------------------------------


@RED
@pytest.mark.spec("T-NL2", "ARCH§7.4", "C13")
def test_a_series_holding_a_bar_at_or_after_the_as_of_time_is_refused() -> None:
    n = 50
    s = series(n)
    last = SERIES_START + timedelta(hours=n - 1)
    _build(s, last + timedelta(microseconds=1))  # strictly after the last bar: fine
    for as_of in (last, last - timedelta(minutes=1)):
        with pytest.raises(ModelError):
            _build(s, as_of)
    with pytest.raises(ClockError):
        _build(s, last.replace(tzinfo=None) + timedelta(hours=1))


@RED
@pytest.mark.spec("ARCH§7.4", "SPEC§9")
def test_the_document_records_its_source_models_data_and_as_of() -> None:
    from zoneinfo import ZoneInfo

    s = series(60, models=(MODEL_HASH, OTHER_HASH))
    as_of = after(60).astimezone(ZoneInfo("America/New_York"))
    calibration = _build(s, as_of, source=CalibrationSource.FILLS)
    document = calibration.document
    assert document.source is CalibrationSource.FILLS == calibration.source
    assert document.models == (MODEL_HASH, OTHER_HASH)
    assert document.data_hash == s.data_hash()
    assert calibration.as_of == as_of
    assert calibration.as_of.utcoffset() == timedelta(0)
    assert document.config == CONFIG
    assert [v.label for v in document.labels] == list(Label)


@RED
@pytest.mark.spec("ARCH§7.4", "SPEC§9")
@pytest.mark.parametrize(
    "fault",
    [
        "times not increasing",
        "row sum",
        "forecast above one",
        "target out of range",
        "smoothed row sum",
        "low above high",
        "nan realized",
        "bad model hash",
        "lengths",
        "int forecast",
    ],
)
def test_a_malformed_series_is_refused(fault: str) -> None:
    from quant.eval.calibration import ForecastSeries

    base = series(10)
    parts = {
        "models": base.models,
        "available_at": base.available_at.copy(),
        "forecast": base.forecast.copy(),
        "target": base.target.copy(),
        "smoothed": base.smoothed.copy(),
        "low": base.low.copy(),
        "high": base.high.copy(),
        "realized": base.realized.copy(),
    }
    if fault == "times not increasing":
        parts["available_at"][4] = parts["available_at"][3]
    elif fault == "row sum":
        parts["forecast"][2, CALM] += 0.01
    elif fault == "forecast above one":
        parts["forecast"][2] = 0.0
        parts["forecast"][2, CALM] = 1.5
        parts["forecast"][2, CHOP] = -0.5
    elif fault == "target out of range":
        parts["target"][2] = len(LABELS)
    elif fault == "smoothed row sum":
        parts["smoothed"][2, CALM] += 0.01
    elif fault == "low above high":
        parts["low"][2] = 2.0
    elif fault == "nan realized":
        parts["realized"][2] = float("nan")
    elif fault == "bad model hash":
        parts["models"] = ("not a hash",)
    elif fault == "lengths":
        parts["low"] = parts["low"][:-1]
    else:
        parts["forecast"] = parts["forecast"].astype(np.int64)
    with pytest.raises(ModelError):
        ForecastSeries(**parts)


@RED
@pytest.mark.spec("ARCH§7.4", "SPEC§9")
def test_series_concatenate_in_time_order_only() -> None:
    from quant.eval.calibration import ForecastSeries

    first = series(10, models=(MODEL_HASH,))
    second = series(10, start=SERIES_START + timedelta(hours=10), models=(OTHER_HASH,), seed=1)
    joined = ForecastSeries.concat([first, second])
    assert len(joined) == 20
    assert joined.models == (MODEL_HASH, OTHER_HASH)
    assert np.array_equal(joined.available_at[:10], first.available_at)
    assert np.array_equal(joined.forecast[10:], second.forecast)
    assert ForecastSeries.concat([first, ForecastSeries.empty()]).data_hash() == first.data_hash()
    with pytest.raises(ModelError):
        ForecastSeries.concat([second, first])
    with pytest.raises(ValueError, match="read-only"):
        joined.forecast[0, 0] = 0.5
