"""Calibration: scoring the filter's label probabilities (spec §9, A4; ARCHITECTURE §7.4; C13).
Built in B7b.

Hidden states are never observed, so a label's forecast is scored against the ex-post smoothed
label (A4). That needs the smoother, which may run only in eval (I2); the result is a plain-data
artifact, `quant.core.calibration.Calibration`, which sizing reads without importing this
module (K3).

**The series** (`ForecastSeries`, built per model by `model_series`). Each out-of-sample bar t
of a model's active span gives:

- the forecast: the filtered label probabilities after row t, `quant.hmm.labels.label_probs`
  of p_t under the model's label table, exactly as the engine's filter reports them (a label
  the model lacks is 0), each clipped to [0, 1] (a sum of state probabilities can leave it by
  rounding);
- the target: the ex-post smoothed label, `top_label(label_probs(s_t))` of the smoothed
  posterior s_t = P(state_t | every row of the model's own feature rows up to the end of its
  span), a tie going to the most cautious label as everywhere else. The smoothed label
  probabilities `label_probs(s_t)` are kept beside it (clipped as the forecast is), so the
  series' hash, and the calibration's, change with anything the smoother read, even where the
  label doesn't;
- the outcome check's pair: the central `level` interval for F1 of the next bar under
  q_t = p_t A (`f1_interval`), and the realized F1 of bar t itself.

Each model's rows start at its own fit_start: the filter the engine runs for it is warmed over
its fit window and stepped on to the swap (spec §6), so p_t is that filter's. The smoother runs
over the same rows and stops at the span's end, which is at or before the as-of time: nothing
the model's span didn't see reaches its targets.

**Label metrics** (`label_metrics`), for a forecast p (n,) and the outcome y_t = 1 when the
target is the label, else 0, each sum `math.fsum` (exactly rounded):

- base rate b = Σ y / n; Brier score BS = Σ (p - y)² / n; reference = the Brier score of the
  constant forecast b (computed by the same function, so a forecast equal to b has skill
  exactly 0); skill = 1 - BS / reference, None when the reference is 0;
- the reliability curve: BINS bins by `bin_of(p)`; per bin its count, mean forecast
  Σ p / count and observed frequency Σ y / count (None for an empty bin);
- ECE = Σ_bins |Σ_bin p - Σ_bin y| / n, the count-weighted mean of |mean forecast - observed|
  written so each bin's term is one exactly rounded difference;
- pass: skill is not None, skill > 0, and ECE ≤ ece_max on the decimal values of the floats
  (`Decimal(repr(x))`, as B4's sanity edges and B7a's drift edges compare).

**C13.** The series holds every out-of-sample bar before the as-of time. The label metrics use
all but the newest `settle_bars` (35) of them, whose smoothed labels haven't settled; a verdict
needs at least `min_bars` (1,000) bars after that exclusion, and until then every label fails
(and isn't recalibrated). The outcome check keeps the newest bars, whose realized F1 is final:
it uses the latest `outcome_bars` pairs (bar t-1's interval, bar t's F1) of the whole series.

**Recalibration** ("fit on past out-of-sample data only"). A label whose raw forecast fails,
with enough bars, is recalibrated with isotonic regression (scikit-learn's
`IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")`):

- the artifact's knots are fit on every scored bar, all of them before the as-of time;
- the verdict judges the map out of sample: knots fit on the older half of the scored bars
  (the first ⌊n/2⌋) map the newer half through `quant.core.calibration.interpolate`, and those
  mapped forecasts are scored on the newer half. Scoring the map on the bars it was fit on
  would always pass (an isotonic fit is calibrated on its own bars), so "if it still fails"
  could never happen;
- if the recalibrated forecast still fails, the label fails, and sizing caps it
  (`Calibration.cap_fraction`).

**Outcome check** (`outcome_check`): the pair (t-1, t) is covered when low_{t-1} ≤ F1_t ≤
high_{t-1}, both ends inclusive. Over the latest min(outcome_bars, n - 1) pairs it passes when
there are outcome_bars of them and covered / n lies in [coverage_low, coverage_high], compared
exactly (as fractions of the decimal values).

Nothing here reads a clock or a store: the job (`quant.jobs.calibrate`) reads the bars and
models and hands them in. hmmlearn is not imported (I2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
import numpy.typing as npt

from quant.core.calibration import (
    Calibration,
    CalibrationConfig,
    CalibrationSource,
    Knots,
    LabelMetrics,
    LabelVerdict,
    OutcomeCheck,
    ReliabilityBin,
)
from quant.core.regime import Label
from quant.data.frames import Floats, Ints
from quant.features.stream import FeatureMatrix
from quant.hmm.model import HmmModel

# A boolean outcome array: y_t, whether bar t's smoothed label is the label scored.
Bools = npt.NDArray[np.bool_]
# The labels, in the order every per-label array and table follows.
LABELS: Final = tuple(Label)
# The feature the outcome check forecasts: F1, the bar's normalized return (spec §3, C13).
OUTCOME_FEATURE: Final = "ret"
# f1_interval's root finding: brentq to this absolute tolerance, inside a bracket this many
# state deviations wide on each side.
INTERVAL_XTOL: Final = 1e-12
INTERVAL_SPAN: Final = 12.0


@dataclass(frozen=True, eq=False)
class ForecastSeries:
    """Per out-of-sample bar, oldest first (n bars): what a calibration scores.

    - `models`: the artifact hashes of the models whose spans the bars come from, in order;
    - `available_at` (n,) int64: each bar's availability, epoch microseconds UTC, strictly
      increasing;
    - `forecast` (n, len(LABELS)) float64: the filtered label probabilities, columns in LABELS
      order, each in [0, 1] and each row summing to 1 within 1e-9;
    - `target` (n,) int64: the index in LABELS of the bar's smoothed label;
    - `smoothed` (n, len(LABELS)) float64: the smoothed label probabilities the target was taken
      from, as `forecast` (each in [0, 1], rows summing to 1 within 1e-9);
    - `low`, `high` (n,) float64: the central interval for F1 of the next bar, forecast at this
      one, low ≤ high;
    - `realized` (n,) float64: this bar's F1, raw.

    Immutable. Zero bars is valid (`empty`)."""

    models: tuple[str, ...]
    available_at: Ints
    forecast: Floats
    target: Ints
    smoothed: Floats
    low: Floats
    high: Floats
    realized: Floats

    def __post_init__(self) -> None:
        """Raise ModelError unless the arrays have the dtypes and shapes above, every value is
        finite, the rules above hold and every `models` entry is 64 lowercase hex characters;
        then store the arrays read-only (`freeze_arrays`)."""
        raise NotImplementedError("B7b: ForecastSeries.__post_init__")

    def __len__(self) -> int:
        raise NotImplementedError("B7b: ForecastSeries.__len__")

    @classmethod
    def empty(cls) -> ForecastSeries:
        """No bars and no models."""
        raise NotImplementedError("B7b: ForecastSeries.empty")

    @classmethod
    def concat(cls, parts: Sequence[ForecastSeries]) -> ForecastSeries:
        """The parts' bars in order, and their models in order. ModelError when the bars don't
        strictly increase across parts."""
        raise NotImplementedError("B7b: ForecastSeries.concat")

    def data_hash(self) -> str:
        """`quant.core.canonical.sha256_hex` of {"models": [...], and for each of
        "available_at", "forecast", "target", "smoothed", "low", "high", "realized" {"dtype",
        "shape", "sha256"}}, each array hashed as HmmModel's are (little-endian, C-order bytes)."""
        raise NotImplementedError("B7b: ForecastSeries.data_hash")


def f1_interval(model: HmmModel, next_state_probs: Floats, *, level: float) -> tuple[float, float]:
    """The central `level` predictive interval for the next bar's F1, in raw units (C13).

    The next bar's state is distributed as q = `next_state_probs` (p_t A), so its F1 is the
    mixture Σ_k q_k N(μ_k, sigma_k²) with, for i the index of OUTCOME_FEATURE in the model's scaler
    columns, μ_k = means[k, i] · std_i + mean_i and sigma_k = sqrt(covars[k, i, i]) · std_i: the
    state's standardized marginal taken back to raw units. The interval is [F⁻¹((1 - level) / 2),
    F⁻¹((1 + level) / 2)] of the mixture's CDF F (`scipy.special.ndtr` per state), each found by
    `scipy.optimize.brentq` with xtol INTERVAL_XTOL in the bracket [min_k(μ_k - INTERVAL_SPAN
    sigma_k), max_k(μ_k + INTERVAL_SPAN sigma_k)] over the states with q_k > 0.

    Raises ModelError when the scaler has no OUTCOME_FEATURE column, `next_state_probs` isn't a
    float64 (K,) vector of nonnegative values summing to 1 within 1e-9, or `level` isn't in
    (0, 1)."""
    raise NotImplementedError("B7b: f1_interval")


def model_series(
    model: HmmModel, matrix: FeatureMatrix, *, start: datetime, end: datetime, level: float
) -> ForecastSeries:
    """The forecasts `model` made over its active span [start, end) (module docstring, "The
    series"), with `models` = (its artifact hash,).

    `matrix` is the model's own feature rows from its fit_start up to the end of the span,
    raw: every row's available_at in [model.fit_start, end). `forward_pass` from π over all of
    them gives p_t (`trace.state_probs[t]`) and q_t = `trace.state_probs[t] @
    model.params.transmat`, and `quant.eval.smoother.smooth` over the same rows the smoothed
    posteriors. The realized F1 is the row's raw OUTCOME_FEATURE value. The out-of-sample bars are the rows with start ≤ available_at < end;
    rows before `start` only warm the filter. No such row gives an empty series with no models.

    Raises ModelError when: the matrix's columns aren't the model scaler's; a row lies outside
    [model.fit_start, end); start < model.fit_end (an in-sample span); start >= end; the matrix
    has no rows; or `forward_pass`, `smooth` or `f1_interval` refuses. ClockError for a naive
    time."""
    raise NotImplementedError("B7b: model_series")


def bin_of(p: float) -> int:
    """The reliability bin of forecast `p` in [0, 1]: min(BINS - 1, ⌊10 · Decimal(repr(p))⌋),
    so 0.1 is in bin 1, 0.3 in bin 3 (its repr is "0.3") and 1.0 in bin 9. ModelError for a
    value that isn't a finite float in [0, 1]."""
    raise NotImplementedError("B7b: bin_of")


def brier_score(forecast: Floats, outcome: Bools) -> float:
    """`math.fsum((p - y)²) / n` over the bars, y = 1.0 where `outcome` is True. ModelError for
    n = 0, unequal lengths, or a forecast outside [0, 1] or not finite."""
    raise NotImplementedError("B7b: brier_score")


def reliability(forecast: Floats, outcome: Bools) -> tuple[ReliabilityBin, ...]:
    """The BINS-bin reliability curve (module docstring): bin b has lower b/10, upper
    (b + 1)/10 (as the floats `b / 10`), its count, `fsum(p) / count` and `fsum(y) / count`, or
    None for both when empty. n = 0 gives BINS empty bins. Validates as `brier_score` (n = 0
    allowed)."""
    raise NotImplementedError("B7b: reliability")


def expected_calibration_error(forecast: Floats, outcome: Bools) -> float:
    """ECE = `math.fsum(abs(fsum(p in bin) - count(y in bin)) for each non-empty bin) / n`
    (module docstring). ModelError for n = 0, as `brier_score` otherwise."""
    raise NotImplementedError("B7b: expected_calibration_error")


def label_metrics(forecast: Floats, outcome: Bools, *, config: CalibrationConfig) -> LabelMetrics:
    """Every metric of the module docstring for one forecast, as a LabelMetrics. n = 0 gives
    the empty metrics (every number None, BINS empty bins, passed False). Raises ModelError for
    unequal lengths, `outcome` not a 1-D bool array, or a forecast that isn't a 1-D float64
    array of finite values in [0, 1]."""
    raise NotImplementedError("B7b: label_metrics")


def fit_isotonic(forecast: Floats, outcome: Bools) -> Knots:
    """The isotonic map of `outcome` (as 0.0/1.0) on `forecast`: scikit-learn's
    `IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")` fit on
    them, its `X_thresholds_` and `y_thresholds_` as new float64 arrays (x strictly increasing,
    y nondecreasing in [0, 1]). Deterministic. ModelError for n = 0, as `label_metrics`
    otherwise."""
    raise NotImplementedError("B7b: fit_isotonic")


def calibrate_label(
    label: Label,
    forecast: Floats,
    outcome: Bools,
    *,
    config: CalibrationConfig,
    sufficient: bool,
) -> tuple[LabelVerdict, Knots | None]:
    """One label's verdict and knots (module docstring, "Recalibration"):

    - raw = `label_metrics(forecast, outcome)`;
    - not `sufficient`: (raw, recalibrated None, 0 knots, passed False), no knots;
    - raw passes: (raw, None, 0, passed True), no knots;
    - else: half = n // 2; the knots `fit_isotonic(forecast[:half], outcome[:half])` map
      forecast[half:] through `interpolate`, recalibrated = `label_metrics` of those against
      outcome[half:]; the artifact's knots are `fit_isotonic(forecast, outcome)`; passed =
      recalibrated.passed; the verdict counts the artifact knots.

    Validates as `label_metrics`."""
    raise NotImplementedError("B7b: calibrate_label")


def outcome_check(series: ForecastSeries, *, config: CalibrationConfig) -> OutcomeCheck:
    """The outcome check of the module docstring over `series`: pairs (t - 1, t) for t ≥ 1, the
    latest min(config.outcome_bars, len - 1) of them (none for fewer than two bars), covered
    when low[t - 1] ≤ realized[t] ≤ high[t - 1]; level = config.outcome_level; passed when there
    are config.outcome_bars pairs and Fraction(covered, n) lies in [Fraction(Decimal(repr(
    coverage_low))), Fraction(Decimal(repr(coverage_high)))]."""
    raise NotImplementedError("B7b: outcome_check")


def build_calibration(
    series: ForecastSeries,
    *,
    as_of: datetime,
    source: CalibrationSource,
    config: CalibrationConfig,
) -> Calibration:
    """The calibration of `series` as of `as_of` (module docstring):

    - bars = len(series); scored = max(0, bars - config.settle_bars), the oldest bars;
      sufficient = scored ≥ config.min_bars;
    - per label j of LABELS: `calibrate_label(label, series.forecast[:scored, j],
      series.target[:scored] == j, config=config, sufficient=sufficient)`;
    - outcome = `outcome_check(series, config=config)` over the whole series;
    - the document's models and data_hash are the series', `as_of` in UTC.

    Raises ModelError, before scoring anything, when a bar's available_at is at or after
    `as_of` (the calibration would score what wasn't known at its as-of time). ClockError for a
    naive `as_of`."""
    raise NotImplementedError("B7b: build_calibration")
