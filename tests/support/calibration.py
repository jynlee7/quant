"""Synthetic forecasters, forecast series and a naive reference for calibration (B7b).

The references work with plain numpy, `math.fsum` and `decimal`, and none of the code under
test, so a test can check a metric against the formula in `quant.eval.calibration`'s docstring
rather than against itself. scikit-learn's IsotonicRegression is the reference for the knots
(tests may import it; K3 keeps it off the decision path).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import numpy.typing as npt

from quant.core.regime import Label

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]
Bools = npt.NDArray[np.bool_]

LABELS = tuple(Label)
# The series' bars: one an hour from here, all inside FIT.
SERIES_START = datetime(2017, 1, 3, 15, 30, tzinfo=UTC)
# The as-of time just after a series of n bars: the hour after its last bar.
MODEL_HASH = "ab" * 32
OTHER_HASH = "cd" * 32
# The level the series' intervals are stated at: r1's outcome level.
SERIES_LEVEL = 0.90


def forecaster(
    n: int, seed: int, *, distort: Callable[[Floats], Floats] | None = None
) -> tuple[Floats, Bools]:
    """n bars: true probabilities drawn uniformly from [0, 1], outcomes drawn from them, and the
    forecast: the true probability (well calibrated), or `distort` of it."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.0, 1.0, n)
    outcome = rng.uniform(0.0, 1.0, n) < truth
    forecast = truth if distort is None else np.clip(distort(truth), 0.0, 1.0)
    return forecast.astype(np.float64), outcome


def ref_bin(p: float) -> int:
    return min(9, int(Decimal(repr(float(p))) * 10))


def ref_metrics(forecast: Floats, outcome: Bools, ece_max: float = 0.05) -> dict[str, Any]:
    """The module docstring's formulas, written out."""
    n = len(forecast)
    y = outcome.astype(np.float64)
    base = math.fsum(y) / n
    brier = math.fsum((forecast - y) ** 2) / n
    reference = math.fsum((np.full(n, base) - y) ** 2) / n
    skill = None if reference == 0 else 1 - brier / reference
    bins = np.array([ref_bin(p) for p in forecast], dtype=np.int64)
    ece = (
        math.fsum(
            abs(math.fsum(forecast[bins == b]) - math.fsum(y[bins == b]))
            for b in range(10)
            if (bins == b).any()
        )
        / n
    )
    counts = [int((bins == b).sum()) for b in range(10)]
    return {
        "base_rate": base,
        "brier": brier,
        "reference": reference,
        "skill": skill,
        "ece": ece,
        "counts": counts,
        "passed": skill is not None and skill > 0 and Decimal(repr(ece)) <= Decimal(repr(ece_max)),
    }


def ref_isotonic(forecast: Floats, outcome: Bools) -> tuple[Floats, Floats]:
    from sklearn.isotonic import IsotonicRegression

    fit = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    fit.fit(forecast, outcome.astype(np.float64))
    return (
        np.asarray(fit.X_thresholds_, dtype=np.float64),
        np.asarray(fit.y_thresholds_, dtype=np.float64),
    )


def series(
    n: int,
    *,
    label_forecast: Floats | None = None,
    target: Ints | None = None,
    covered: Bools | None = None,
    start: datetime = SERIES_START,
    models: tuple[str, ...] = (MODEL_HASH,),
    seed: int = 0,
    level: float = SERIES_LEVEL,
) -> Any:
    """A ForecastSeries of n hourly bars from `start`.

    - forecast: column CALM_UP is `label_forecast` (default: well calibrated draws) and CHOP the
      rest, every other label 0;
    - target: `target` (default: CALM_UP where the default draws' outcome holds, else CHOP),
      and the smoothed probabilities one-hot on it;
    - the outcome check: interval [-1, 1] at every bar, realized F1 0.5 (inside) where
      `covered` holds for that bar and 2.0 (outside) elsewhere; the first bar's realized value
      is never scored. Default: every bar covered;
    - the intervals' level: `level` (r1's 0.90), and one span of every bar.
    """
    from quant.eval.calibration import ForecastSeries

    forecast_calm, outcome = forecaster(n, seed)
    if label_forecast is not None:
        forecast_calm = label_forecast
    if target is None:
        target = np.where(outcome, LABELS.index(Label.CALM_UP), LABELS.index(Label.CHOP))
    forecast = np.zeros((n, len(LABELS)), dtype=np.float64)
    forecast[:, LABELS.index(Label.CALM_UP)] = forecast_calm
    forecast[:, LABELS.index(Label.CHOP)] = 1.0 - forecast_calm
    inside = np.ones(n, dtype=np.bool_) if covered is None else covered
    available = np.array(
        [int((start + timedelta(hours=i)).timestamp() * 1_000_000) for i in range(n)],
        dtype=np.int64,
    )
    return ForecastSeries(
        models=models if n else (),
        available_at=available,
        forecast=forecast,
        target=np.asarray(target, dtype=np.int64),
        smoothed=np.eye(len(LABELS), dtype=np.float64)[np.asarray(target, dtype=np.int64)],
        low=np.full(n, -1.0),
        high=np.full(n, 1.0),
        realized=np.where(inside, 0.5, 2.0).astype(np.float64),
        level=level,
    )


def after(n: int, start: datetime = SERIES_START) -> datetime:
    """An as-of time after every bar of `series(n, start=start)`: one hour past its last."""
    return start + timedelta(hours=n)


def exact(n: int) -> tuple[Floats, Ints]:
    """A CALM_UP forecast calibrated exactly in every block of 40 bars: 20 bars forecast 0.05,
    one of which is CALM_UP, and 20 forecast 0.95, nineteen of which are; every other bar's
    target is CHOP. Repeated (and cut) to n bars."""
    block_forecast = np.array([0.05] * 20 + [0.95] * 20, dtype=np.float64)
    calm = np.zeros(40, dtype=np.bool_)
    calm[7] = True  # one of the twenty 0.05 bars
    calm[20:40] = True
    calm[33] = False  # one of the twenty 0.95 bars isn't
    reps = -(-n // 40)
    forecast = np.tile(block_forecast, reps)[:n]
    target = np.where(
        np.tile(calm, reps)[:n], LABELS.index(Label.CALM_UP), LABELS.index(Label.CHOP)
    ).astype(np.int64)
    return forecast, target


def covered_share(n: int, *, misses_every: int = 10) -> Bools:
    """Bar t's realized F1 is inside the interval forecast at t - 1 unless t % misses_every is
    0: over any run of whole tens of pairs, 90% coverage."""
    return np.arange(n) % misses_every != 0
