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

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.special import ndtr
from sklearn.isotonic import IsotonicRegression

from quant.core.calibration import (
    BINS,
    CALIBRATION_FORMAT,
    Calibration,
    CalibrationConfig,
    CalibrationDocument,
    CalibrationSource,
    Knots,
    LabelMetrics,
    LabelVerdict,
    OutcomeCheck,
    ReliabilityBin,
    interpolate,
)
from quant.core.canonical import sha256_hex
from quant.core.clock import ensure_utc
from quant.core.epoch import to_us
from quant.core.errors import ModelError
from quant.core.regime import Label
from quant.data.frames import Floats, Ints
from quant.eval.smoother import smooth
from quant.features.stream import FeatureMatrix, freeze_arrays
from quant.hmm.filter import forward_pass
from quant.hmm.labels import label_probs, top_label
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
        n = (
            len(self.available_at)
            if isinstance(self.available_at, np.ndarray) and self.available_at.ndim == 1
            else -1
        )
        ok = (
            isinstance(self.available_at, np.ndarray)
            and self.available_at.dtype == np.int64
            and self.available_at.ndim == 1
            and isinstance(self.forecast, np.ndarray)
            and self.forecast.dtype == np.float64
            and self.forecast.shape == (n, len(LABELS))
            and isinstance(self.smoothed, np.ndarray)
            and self.smoothed.dtype == np.float64
            and self.smoothed.shape == (n, len(LABELS))
            and isinstance(self.target, np.ndarray)
            and self.target.dtype == np.int64
            and self.target.shape == (n,)
            and all(
                isinstance(a, np.ndarray) and a.dtype == np.float64 and a.shape == (n,)
                for a in (self.low, self.high, self.realized)
            )
        )
        if not ok:
            raise ModelError("forecast series arrays have invalid dtypes or shapes")
        if n and not (
            (np.diff(self.available_at) > 0).all()
            and np.isfinite(self.forecast).all()
            and (self.forecast >= 0).all()
            and (self.forecast <= 1).all()
            and (np.abs(self.forecast.sum(axis=1) - 1) <= 1e-9).all()
            and ((self.target >= 0) & (self.target < len(LABELS))).all()
            and np.isfinite(self.smoothed).all()
            and (self.smoothed >= 0).all()
            and (self.smoothed <= 1).all()
            and (np.abs(self.smoothed.sum(axis=1) - 1) <= 1e-9).all()
            and np.isfinite(self.low).all()
            and np.isfinite(self.high).all()
            and np.isfinite(self.realized).all()
            and (self.low <= self.high).all()
        ):
            raise ModelError("forecast series values or time order are invalid")
        if not isinstance(self.models, tuple) or any(
            not isinstance(m, str) or re.fullmatch(r"[0-9a-f]{64}", m) is None for m in self.models
        ):
            raise ModelError("forecast series model hashes must be lowercase SHA-256")
        freeze_arrays(
            self, "available_at", "forecast", "target", "smoothed", "low", "high", "realized"
        )

    def __len__(self) -> int:
        return len(self.available_at)

    @classmethod
    def empty(cls) -> ForecastSeries:
        """No bars and no models."""
        return cls(
            (),
            np.empty(0, np.int64),
            np.empty((0, len(LABELS))),
            np.empty(0, np.int64),
            np.empty((0, len(LABELS))),
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )

    @classmethod
    def concat(cls, parts: Sequence[ForecastSeries]) -> ForecastSeries:
        """The parts' bars in order, and their models in order. ModelError when the bars don't
        strictly increase across parts."""
        parts = list(parts)
        if not parts:
            return cls.empty()
        t = np.concatenate([p.available_at for p in parts])
        if len(t) and not (np.diff(t) > 0).all():
            raise ModelError("forecast series parts must increase strictly in time")
        return cls(
            tuple(m for p in parts for m in p.models),
            t,
            np.concatenate([p.forecast for p in parts]) if len(t) else np.empty((0, len(LABELS))),
            np.concatenate([p.target for p in parts]),
            np.concatenate([p.smoothed for p in parts]),
            np.concatenate([p.low for p in parts]),
            np.concatenate([p.high for p in parts]),
            np.concatenate([p.realized for p in parts]),
        )

    def data_hash(self) -> str:
        """`quant.core.canonical.sha256_hex` of {"models": [...], and for each of
        "available_at", "forecast", "target", "smoothed", "low", "high", "realized" {"dtype",
        "shape", "sha256"}}, each array hashed as HmmModel's are (little-endian, C-order bytes)."""
        arrays = {
            n: getattr(self, n)
            for n in ("available_at", "forecast", "target", "smoothed", "low", "high", "realized")
        }
        return sha256_hex(
            {
                "models": list(self.models),
                **{
                    n: {
                        "dtype": a.dtype.str,
                        "shape": list(a.shape),
                        "sha256": hashlib.sha256(
                            np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<")).tobytes()
                        ).hexdigest(),
                    }
                    for n, a in arrays.items()
                },
            }
        )


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
    cols = model.scaler.columns
    if OUTCOME_FEATURE not in cols:
        raise ModelError("predictive interval requires the ret feature")
    q = next_state_probs
    k = model.params.n_states
    if not (
        isinstance(q, np.ndarray)
        and q.dtype == np.float64
        and q.shape == (k,)
        and np.isfinite(q).all()
        and (q >= 0).all()
        and abs(q.sum() - 1) <= 1e-9
    ):
        raise ModelError("next-state probabilities must be a float64 probability vector")
    if not 0 < level < 1:
        raise ModelError("predictive interval level must lie in (0, 1)")
    i = cols.index(OUTCOME_FEATURE)
    mu = model.params.means[:, i] * model.scaler.std[i] + model.scaler.mean[i]
    sd = np.sqrt(model.params.covars[:, i, i]) * model.scaler.std[i]
    live = q > 0
    lo = float(np.min(mu[live] - INTERVAL_SPAN * sd[live]))
    hi = float(np.max(mu[live] + INTERVAL_SPAN * sd[live]))

    def cdf(x: float) -> float:
        return float(np.sum(q[live] * ndtr((x - mu[live]) / sd[live])))

    return (
        brentq(lambda x: cdf(x) - (1 - level) / 2, lo, hi, xtol=INTERVAL_XTOL),
        brentq(lambda x: cdf(x) - (1 + level) / 2, lo, hi, xtol=INTERVAL_XTOL),
    )


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
    start, end = ensure_utc(start), ensure_utc(end)
    if (
        matrix.columns != model.scaler.columns
        or not len(matrix)
        or start < model.fit_end
        or start >= end
    ):
        raise ModelError(
            "model series requires nonempty rows and an out-of-sample span with matching columns"
        )
    if int(matrix.available_at.min()) < to_us(model.fit_start) or int(
        matrix.available_at.max()
    ) >= to_us(end):
        raise ModelError("model series rows must lie within [fit_start, end)")
    trace = forward_pass(model.params, model.scaler, matrix.values, columns=matrix.columns)
    smoothed = smooth(model.params, model.scaler, matrix.values, columns=matrix.columns)
    indices = np.nonzero(matrix.available_at >= to_us(start))[0]
    if not len(indices):
        return ForecastSeries.empty()
    i = matrix.columns.index(OUTCOME_FEATURE)
    forecast = np.empty((len(indices), len(LABELS)))
    smoothed_forecast = np.empty_like(forecast)
    target = np.empty(len(indices), np.int64)
    low = np.empty(len(indices))
    high = np.empty(len(indices))
    for j, t in enumerate(indices):
        filtered_probs = label_probs(trace.state_probs[t], model.labels)
        forecast[j] = [min(1.0, max(0.0, filtered_probs.get(label, 0.0))) for label in LABELS]
        smooth_probs = label_probs(smoothed[t], model.labels)
        smoothed_forecast[j] = [
            min(1.0, max(0.0, smooth_probs.get(label, 0.0))) for label in LABELS
        ]
        target[j] = LABELS.index(top_label(smooth_probs))
        low[j], high[j] = f1_interval(
            model, trace.state_probs[t] @ model.params.transmat, level=level
        )
    return ForecastSeries(
        (model.artifact_hash(),),
        matrix.available_at[indices].copy(),
        forecast,
        target,
        smoothed_forecast,
        low,
        high,
        matrix.values[indices, i].copy(),
    )


def bin_of(p: float) -> int:
    """The reliability bin of forecast `p` in [0, 1]: min(BINS - 1, ⌊10 · Decimal(repr(p))⌋),
    so 0.1 is in bin 1, 0.3 in bin 3 (its repr is "0.3") and 1.0 in bin 9. ModelError for a
    value that isn't a finite float in [0, 1]."""
    if not isinstance(p, float) or not math.isfinite(p) or p < 0 or p > 1:
        raise ModelError("reliability forecast must be a finite float in [0, 1]")
    return min(BINS - 1, int(Decimal(repr(p)) * 10))


def brier_score(forecast: Floats, outcome: Bools) -> float:
    """`math.fsum((p - y)²) / n` over the bars, y = 1.0 where `outcome` is True. ModelError for
    n = 0, unequal lengths, or a forecast outside [0, 1] or not finite."""
    _check(forecast, outcome, False)
    y = outcome.astype(np.float64)
    return math.fsum((forecast - y) ** 2) / len(forecast)


def reliability(forecast: Floats, outcome: Bools) -> tuple[ReliabilityBin, ...]:
    """The BINS-bin reliability curve (module docstring): bin b has lower b/10, upper
    (b + 1)/10 (as the floats `b / 10`), its count, `fsum(p) / count` and `fsum(y) / count`, or
    None for both when empty. n = 0 gives BINS empty bins. Validates as `brier_score` (n = 0
    allowed)."""
    _check(forecast, outcome)
    b = _bins(forecast)
    y = outcome.astype(np.float64)
    out = []
    for k in range(BINS):
        m = b == k
        c = int(m.sum())
        out.append(
            ReliabilityBin(
                lower=k / 10,
                upper=(k + 1) / 10,
                count=c,
                forecast=math.fsum(forecast[m]) / c if c else None,
                observed=math.fsum(y[m]) / c if c else None,
            )
        )
    return tuple(out)


def expected_calibration_error(forecast: Floats, outcome: Bools) -> float:
    """ECE = `math.fsum(abs(fsum(p in bin) - count(y in bin)) for each non-empty bin) / n`
    (module docstring). ModelError for n = 0, as `brier_score` otherwise."""
    _check(forecast, outcome, False)
    b = _bins(forecast)
    y = outcome.astype(np.float64)
    return math.fsum(
        abs(math.fsum(forecast[b == k]) - math.fsum(y[b == k]))
        for k in range(BINS)
        if (b == k).any()
    ) / len(forecast)


def label_metrics(forecast: Floats, outcome: Bools, *, config: CalibrationConfig) -> LabelMetrics:
    """Every metric of the module docstring for one forecast, as a LabelMetrics. n = 0 gives
    the empty metrics (every number None, BINS empty bins, passed False). Raises ModelError for
    unequal lengths, `outcome` not a 1-D bool array, or a forecast that isn't a 1-D float64
    array of finite values in [0, 1]."""
    _check(forecast, outcome)
    n = len(forecast)
    if not n:
        return LabelMetrics(
            n=0,
            base_rate=None,
            brier=None,
            reference=None,
            skill=None,
            ece=None,
            reliability=reliability(forecast, outcome),
            passed=False,
        )
    y = outcome.astype(np.float64)
    base = math.fsum(y) / n
    bs = brier_score(forecast, outcome)
    ref = brier_score(np.full(n, base), outcome)
    skill = None if ref == 0 else 1 - bs / ref
    ece = expected_calibration_error(forecast, outcome)
    passed = skill is not None and skill > 0 and Decimal(repr(ece)) <= Decimal(repr(config.ece_max))
    return LabelMetrics(
        n=n,
        base_rate=base,
        brier=bs,
        reference=ref,
        skill=skill,
        ece=ece,
        reliability=reliability(forecast, outcome),
        passed=passed,
    )


def fit_isotonic(forecast: Floats, outcome: Bools) -> Knots:
    """The isotonic map of `outcome` (as 0.0/1.0) on `forecast`: scikit-learn's
    `IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")` fit on
    them, its `X_thresholds_` and `y_thresholds_` as new float64 arrays (x strictly increasing,
    y nondecreasing in [0, 1]). Deterministic. ModelError for n = 0, as `label_metrics`
    otherwise."""
    _check(forecast, outcome, False)
    fitted = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(
        forecast, outcome.astype(np.float64)
    )
    return np.asarray(fitted.X_thresholds_, dtype=np.float64).copy(), np.asarray(
        fitted.y_thresholds_, dtype=np.float64
    ).copy()


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
    raw = label_metrics(forecast, outcome, config=config)
    if not sufficient:
        return LabelVerdict(label=label, raw=raw, recalibrated=None, knots=0, passed=False), None
    if raw.passed:
        return LabelVerdict(label=label, raw=raw, recalibrated=None, knots=0, passed=True), None
    half = len(forecast) // 2
    early_x, early_y = fit_isotonic(forecast[:half], outcome[:half])
    recalibrated = label_metrics(
        interpolate(early_x, early_y, forecast[half:]), outcome[half:], config=config
    )
    knots = fit_isotonic(forecast, outcome)
    return LabelVerdict(
        label=label,
        raw=raw,
        recalibrated=recalibrated,
        knots=len(knots[0]),
        passed=recalibrated.passed,
    ), knots


def outcome_check(series: ForecastSeries, *, config: CalibrationConfig) -> OutcomeCheck:
    """The outcome check of the module docstring over `series`: pairs (t - 1, t) for t ≥ 1, the
    latest min(config.outcome_bars, len - 1) of them (none for fewer than two bars), covered
    when low[t - 1] ≤ realized[t] ≤ high[t - 1]; level = config.outcome_level; passed when there
    are config.outcome_bars pairs and Fraction(covered, n) lies in [Fraction(Decimal(repr(
    coverage_low))), Fraction(Decimal(repr(coverage_high)))]."""
    n = len(series)
    pairs = max(0, n - 1)
    m = min(config.outcome_bars, pairs)
    covered = 0
    for t in range(n - m, n):
        if series.low[t - 1] <= series.realized[t] <= series.high[t - 1]:
            covered += 1
    passed = (
        m == config.outcome_bars
        and Fraction(Decimal(repr(config.coverage_low)))
        <= Fraction(covered, m)
        <= Fraction(Decimal(repr(config.coverage_high)))
        if m
        else False
    )
    return OutcomeCheck(n=m, covered=covered, level=config.outcome_level, passed=passed)


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
    as_of = ensure_utc(as_of)
    if len(series) and int(series.available_at.max()) >= to_us(as_of):
        raise ModelError("calibration series contains bars at or after its as-of time")
    bars = len(series)
    scored = max(0, bars - config.settle_bars)
    sufficient = scored >= config.min_bars
    verdicts = []
    knots = {}
    for j, label in enumerate(LABELS):
        verdict, label_knots = calibrate_label(
            label,
            np.ascontiguousarray(series.forecast[:scored, j]),
            series.target[:scored] == j,
            config=config,
            sufficient=sufficient,
        )
        verdicts.append(verdict)
        if label_knots is not None:
            knots[label] = label_knots
    document = CalibrationDocument(
        format=CALIBRATION_FORMAT,
        source=source,
        as_of=as_of,
        models=series.models,
        bars=bars,
        scored=scored,
        sufficient=sufficient,
        data_hash=series.data_hash(),
        config=config,
        labels=tuple(verdicts),
        outcome=outcome_check(series, config=config),
    )
    return Calibration(document, knots)


def _check(forecast: Floats, outcome: Bools, allow_empty: bool = True) -> None:
    if not (
        isinstance(forecast, np.ndarray)
        and forecast.dtype == np.float64
        and forecast.ndim == 1
        and isinstance(outcome, np.ndarray)
        and outcome.dtype == np.bool_
        and outcome.shape == forecast.shape
    ):
        raise ModelError(
            "metrics require matching one-dimensional float64 forecasts and bool outcomes"
        )
    if not np.isfinite(forecast).all() or (forecast < 0).any() or (forecast > 1).any():
        raise ModelError("forecasts must be finite probabilities in [0, 1]")
    if not allow_empty and not len(forecast):
        raise ModelError("metrics require at least one forecast")


def _bins(forecast: Floats) -> Ints:
    return np.array([bin_of(float(p)) for p in forecast], dtype=np.int64)
