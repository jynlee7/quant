"""Calibration artifacts: what sizing reads (spec §9, A4; ARCHITECTURE §7.4, §9.1; C13). Built
in B7b.

`quant.eval.calibration` builds a calibration at each refit (ARCHITECTURE §7.3 step 5) and
nightly for the report. Sizing (B8) and the engine (B14) consume it here, as plain data: this
module imports nothing outside `quant.core`, numpy and pydantic, so the decision path never
imports eval, scikit-learn or hmmlearn to read one (K3).

**What a calibration says.** For each label (`tuple(Label)` order, every label, used or not):

- the raw metrics of the filtered label probability against the ex-post smoothed label over the
  scored out-of-sample bars (`LabelMetrics`: Brier score, Brier skill against the base rate, a
  BINS-bin reliability curve, ECE, and whether ECE ≤ ece_max and skill > 0);
- when the raw forecast fails and there are enough bars, the isotonic recalibration: its knots
  (in `knots.npz`) and the recalibrated forecast's metrics on held-out bars
  (`quant.eval.calibration`, "Recalibration");
- the verdict, `LabelVerdict.passed`: enough bars (`sufficient`, C13) and either the raw
  forecast or its recalibration passes.

And once for every label, the outcome check (`OutcomeCheck`): how often the realized F1 fell
inside the central `outcome_level` one-bar-ahead predictive interval over the latest
`outcome_bars` out-of-sample bars, passing when the coverage lies in [coverage_low,
coverage_high] (C13).

**Using one** (spec §9):

- `calibrated(label, p)`: P̃, the filtered probability `p` mapped through the label's knots by
  `interpolate`, or `p` itself when the label has none;
- `cap_fraction(label)`: 1.0 when the label passed and the outcome check passed, else
  `config.failed_cap` (0.25 for r1): "a failing label is sized at no more than 25% of its cap",
  and "a failure [of the outcome check] caps every label at 25% of its cap". Having no
  calibration at all is every label failing; that is the caller's to apply (B8);
- `require_before(tau)`: T-NL2 for calibrations (ARCHITECTURE §7.4): a calibration may be used
  only at a τ strictly after its as-of time.

**The artifact** (ARCHITECTURE §9.1) is the immutable directory `<root>/<artifact_hash>/`,
`root` being `artifacts/calibrations`, with two files:

- `calibration.json`: the canonical JSON (quant.core.canonical) of
  `{"artifact_hash": <hash>, "calibration": <CalibrationDocument>}`;
- `knots.npz`: for each label with knots, in `tuple(Label)` order, the members
  `<LABEL>_x.npy` then `<LABEL>_y.npy` (e.g. `CALM_UP_x.npy`), float64, written exactly as
  `quant.hmm.model.save_model` writes params.npz: an uncompressed ZIP (ZIP_STORED), each member
  with the fixed date 1980-01-01 00:00:00, by `numpy.lib.format.write_array(...,
  allow_pickle=False)`. A calibration with no knots still writes the file, with no members. It
  reads back only with `numpy.load(path, allow_pickle=False)`. No pickles.

**The artifact hash** is `quant.core.canonical.sha256_hex` of

    {"calibration": <CalibrationDocument>,
     "knots": {"<LABEL>_x": {"dtype": "float64", "shape": [...], "sha256": <hex>},
               "<LABEL>_y": {...}, ... for each label with knots}}

with each array hashed as `HmmModel.artifact_hash` hashes its arrays (little-endian, C-order
bytes). It names the calibration everywhere: the directory, ops.db's "calibration" pointer
(`quant.store.ops`) and the decision records.

Times are UTC. "Now" never appears here (K6): a calibration's only time is its as-of.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Final, Self

import numpy as np
import numpy.typing as npt
from pydantic import AwareDatetime, model_validator

from quant.core.canonical import canonical_dumps, canonical_loads, sha256_hex
from quant.core.clock import ensure_utc
from quant.core.errors import ClockError, ModelError
from quant.core.model import StrictModel
from quant.core.regime import Label

Floats = npt.NDArray[np.float64]

# calibration.json's format version. Bump it when the artifact's layout changes.
CALIBRATION_FORMAT: Final = 1
# Spec §9: a 10-bin reliability curve. Bin b holds the forecasts p with b/10 ≤ p < (b+1)/10,
# the last bin [0.9, 1.0] closed, compared on the decimal value of p (`quant.eval.calibration.
# bin_of`).
BINS: Final = 10
# The two files of an artifact directory.
DOCUMENT_FILE: Final = "calibration.json"
KNOTS_FILE: Final = "knots.npz"
# A label probability this far outside [0, 1] is rounding, and is clipped (`calibrated`).
PROBABILITY_SLACK: Final = 1e-9


class CalibrationSource(StrEnum):
    """Where the scored forecasts come from (spec §9, ARCHITECTURE §7.4): the walk-forward
    backtest's bars, or the bars with a paper fill. The fills rule (a label counts once it has
    30 fills; until then the backtest calibration stands in, flagged as such) is B14's."""

    BACKTEST = "backtest"
    FILLS = "fills"


class CalibrationConfig(StrictModel):
    """The calibration rules' numbers (spec §9, C13), fixed in config like the drift
    thresholds (`r1_calibration_config`).

    - `min_bars`: a verdict needs at least this many scored out-of-sample bars; until then every
      label fails (C13: 1,000);
    - `settle_bars`: the newest bars before the as-of time left out of the label metrics,
      because their smoothed labels haven't settled (C13: 35);
    - `ece_max`: pass needs ECE ≤ this (spec §9: 0.05), compared on decimal values;
    - `outcome_bars`, `outcome_level`: the outcome check's window and interval (C13: the latest
      1,000 bars, the central 90%);
    - `coverage_low`, `coverage_high`: its band, inclusive (C13: [0.876, 0.924], the 99%
      binomial band), compared exactly;
    - `failed_cap`: the fraction of its cap a failing label, or every label after a failed
      outcome check, is sized at (spec §9: no more than 25%).
    """

    min_bars: int
    settle_bars: int
    ece_max: float
    outcome_bars: int
    outcome_level: float
    coverage_low: float
    coverage_high: float
    failed_cap: float

    @model_validator(mode="after")
    def _valid(self) -> Self:
        """Return self when min_bars ≥ 2 (the recalibration splits the scored bars in two),
        settle_bars ≥ 0, 0 < ece_max < 1, outcome_bars ≥ 1, 0 < outcome_level < 1,
        0 ≤ coverage_low ≤ outcome_level ≤ coverage_high ≤ 1 and 0 ≤ failed_cap ≤ 0.25.
        Otherwise raise ValueError naming the values (a pydantic ValidationError)."""
        if not (
            self.min_bars >= 2
            and self.settle_bars >= 0
            and 0 < self.ece_max < 1
            and self.outcome_bars >= 1
            and 0 < self.outcome_level < 1
            and 0 <= self.coverage_low <= self.outcome_level <= self.coverage_high <= 1
            and 0 <= self.failed_cap <= 0.25
        ):
            raise ValueError(f"invalid calibration config: {self.model_dump()}")
        return self


def r1_calibration_config() -> CalibrationConfig:
    """Spec §9's and C13's numbers, which r1 keeps: min_bars 1000, settle_bars 35, ece_max
    0.05, outcome_bars 1000, outcome_level 0.90, coverage [0.876, 0.924], failed_cap 0.25. A
    new, validated object on each call."""
    return CalibrationConfig(
        min_bars=1000,
        settle_bars=35,
        ece_max=0.05,
        outcome_bars=1000,
        outcome_level=0.90,
        coverage_low=0.876,
        coverage_high=0.924,
        failed_cap=0.25,
    )


class ReliabilityBin(StrictModel):
    """One bin of a reliability curve: [lower, upper) (the last bin [0.9, 1.0]), how many
    forecasts fell in it, their mean, and the fraction of those bars whose target was the
    label. `forecast` and `observed` are None for an empty bin."""

    lower: float
    upper: float
    count: int
    forecast: float | None
    observed: float | None


class LabelMetrics(StrictModel):
    """One forecast's metrics against the smoothed label over `n` bars
    (`quant.eval.calibration.label_metrics`, which defines each).

    - `base_rate`: the fraction of bars whose target is the label;
    - `brier`: the Brier score; `reference`: the Brier score of the constant base rate;
    - `skill`: 1 - brier / reference, None when reference is 0 (a target that never or always
      holds: no skill can be shown);
    - `ece`: the expected calibration error over the BINS bins;
    - `reliability`: exactly BINS bins, in order;
    - `passed`: skill is not None, skill > 0 and ECE ≤ ece_max (decimal values).

    With n = 0 every number is None, every bin is empty and `passed` is False."""

    n: int
    base_rate: float | None
    brier: float | None
    reference: float | None
    skill: float | None
    ece: float | None
    reliability: tuple[ReliabilityBin, ...]
    passed: bool


class LabelVerdict(StrictModel):
    """One label's calibration.

    - `raw`: the filtered probability's metrics on every scored bar;
    - `recalibrated`: the isotonic map's metrics on the held-out later half of the scored bars,
      None when the label wasn't recalibrated (raw passed, or too few bars);
    - `knots`: how many knots the label has in knots.npz, 0 when it has none; > 0 exactly when
      `recalibrated` is not None;
    - `passed`: `sufficient` and (raw passed or the recalibration passed)."""

    label: Label
    raw: LabelMetrics
    recalibrated: LabelMetrics | None
    knots: int
    passed: bool


class OutcomeCheck(StrictModel):
    """The outcome check (spec §9, A4, C13): over the latest `n` one-bar-ahead pairs (at most
    config.outcome_bars), how many realized F1 values fell inside the central `level` interval
    forecast the bar before. `passed`: n == config.outcome_bars and covered / n lies in
    [coverage_low, coverage_high], compared exactly."""

    n: int
    covered: int
    level: float
    passed: bool


class CalibrationDocument(StrictModel):
    """calibration.json's `calibration` object: everything but the knots arrays.

    - `format`: CALIBRATION_FORMAT;
    - `source`: backtest or fills;
    - `as_of`: every bar scored became available strictly before it, and the calibration may
      be used only strictly after it (T-NL2);
    - `models`: the artifact hashes of the models whose forecasts were scored, in the order
      their spans governed (empty when none had governed before `as_of`);
    - `spans`: how many out-of-sample bars each span of the series holds, oldest first: one span
      per model span, a span the lockbox cuts counting as two (`quant.jobs.calibrate`);
    - `bars`: the out-of-sample bars before `as_of`, the sum of `spans`; `scored`: the sum over
      the spans of max(0, span - settle_bars), each span's oldest bars, which the label metrics
      use (each span's newest settle_bars targets were smoothed up to the span's end, so they
      haven't settled); `sufficient`: scored ≥ min_bars (C13);
    - `data_hash`: `ForecastSeries.data_hash()` of the series scored
      (`quant.eval.calibration`), so two calibrations of different data never share a hash;
    - `config`: the rules it was judged by;
    - `labels`: one LabelVerdict per label, in `tuple(Label)` order;
    - `outcome`: the outcome check."""

    format: int
    source: CalibrationSource
    as_of: AwareDatetime
    models: tuple[str, ...]
    spans: tuple[int, ...]
    bars: int
    scored: int
    sufficient: bool
    data_hash: str
    config: CalibrationConfig
    labels: tuple[LabelVerdict, ...]
    outcome: OutcomeCheck


Knots = tuple[Floats, Floats]


def scored_bars(spans: tuple[int, ...], config: CalibrationConfig) -> int:
    """How many bars the label metrics score (C13): the sum over the spans of max(0, span -
    config.settle_bars). Each span leaves out its newest settle_bars bars."""
    return sum(max(0, n - config.settle_bars) for n in spans)


def metrics_pass(skill: float | None, ece: float | None, config: CalibrationConfig) -> bool:
    """Spec §9's pass: skill is not None, skill > 0, and ECE ≤ config.ece_max, compared on the
    decimal values of the floats (`Decimal(repr(x))`). The one rule `LabelMetrics.passed`
    holds to, wherever it is computed or checked."""
    return (
        skill is not None
        and ece is not None
        and skill > 0
        and Decimal(repr(ece)) <= Decimal(repr(config.ece_max))
    )


def outcome_pass(n: int, covered: int, config: CalibrationConfig) -> bool:
    """The outcome check's pass (C13): n == config.outcome_bars and Fraction(covered, n) lies in
    [coverage_low, coverage_high], each bound taken exactly from its decimal value."""
    return (
        n > 0
        and n == config.outcome_bars
        and Fraction(Decimal(repr(config.coverage_low)))
        <= Fraction(covered, n)
        <= Fraction(Decimal(repr(config.coverage_high)))
    )


def interpolate(knots_x: Floats, knots_y: Floats, values: Floats) -> Floats:
    """The isotonic map at `values`: piecewise linear through the knots (x_i, y_i), constant
    y_0 below x_0 and y_m above x_m, exactly `numpy.interp(values, knots_x, knots_y)`, as a new
    float64 array of `values`' shape. This is scikit-learn's `IsotonicRegression(...,
    out_of_bounds="clip").predict` from its fitted thresholds, written without scikit-learn
    so sizing can apply it (K3). A single knot maps everything to y_0.

    Raises ModelError unless the knots are 1-D float64 arrays of one length m ≥ 1, x strictly
    increasing, y nondecreasing within [0, 1], all finite, and `values` is a float64 array of
    finite values in [0, 1]."""
    _check_knots(knots_x, knots_y)
    if not (
        isinstance(values, np.ndarray)
        and values.dtype == np.float64
        and np.isfinite(values).all()
        and (values >= 0).all()
        and (values <= 1).all()
    ):
        raise ModelError("values must be probabilities")
    return np.asarray(np.interp(values, knots_x, knots_y), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class Calibration:
    """A calibration: its document and each recalibrated label's knots (module docstring).
    Immutable: equality is by `artifact_hash`, not `==`."""

    document: CalibrationDocument
    knots: Mapping[Label, Knots]

    def __post_init__(self) -> None:
        """Raise ModelError unless the parts agree:

        - `document.format` is CALIBRATION_FORMAT and `document.labels` holds exactly one
          verdict per label, in `tuple(Label)` order, each with BINS reliability bins in its
          metrics;
        - every `spans` entry is positive, `bars == sum(spans)`, `scored ==
          scored_bars(spans, config)` and `sufficient == (scored ≥ config.min_bars)`;
        - each LabelMetrics' `passed` (raw and recalibrated) is `metrics_pass(skill, ece,
          config)`: a verdict flag its own numbers contradict is refused;
        - `recalibrated` is None when not `sufficient` (too few bars recalibrate nothing); each
          verdict's `passed` is `sufficient and (raw.passed or (recalibrated is not None and
          recalibrated.passed))`, and its `knots` count is > 0 exactly when `recalibrated` is
          not None;
        - `knots` has an entry for exactly the labels whose verdict counts knots, each a pair
          of 1-D float64 arrays of that many points, x strictly increasing, y nondecreasing
          within [0, 1], all finite;
        - `outcome.level == config.outcome_level`, 0 ≤ covered ≤ n ≤ config.outcome_bars, and
          `outcome.passed == outcome_pass(n, covered, config)`;
        - every `models` entry and `data_hash` are 64 lowercase hex characters.

        Store `knots` as a read-only mapping of read-only array copies (as
        `quant.features.stream.freeze_arrays` makes them, without importing it), so nothing
        reachable from a Calibration can change its hash."""
        document = self.document
        config = document.config
        if document.format != CALIBRATION_FORMAT or tuple(
            verdict.label for verdict in document.labels
        ) != tuple(Label):
            raise ModelError("calibration format or label order is invalid")
        if any(
            len(verdict.raw.reliability) != BINS
            or (verdict.recalibrated is not None and len(verdict.recalibrated.reliability) != BINS)
            for verdict in document.labels
        ):
            raise ModelError("calibration metrics must have ten reliability bins")
        if (
            any(n <= 0 for n in document.spans)
            or document.bars != sum(document.spans)
            or document.scored != scored_bars(document.spans, config)
            or document.sufficient != (document.scored >= config.min_bars)
        ):
            raise ModelError("calibration scored count or sufficiency disagrees with its config")
        for verdict in document.labels:
            if any(
                metrics.passed != metrics_pass(metrics.skill, metrics.ece, config)
                for metrics in (verdict.raw, verdict.recalibrated)
                if metrics is not None
            ):
                raise ModelError("label metrics' pass disagrees with their skill and ECE")
            if verdict.recalibrated is not None and not document.sufficient:
                raise ModelError("a calibration with too few bars recalibrates nothing")
            if verdict.passed != (
                document.sufficient
                and (
                    verdict.raw.passed
                    or (verdict.recalibrated is not None and verdict.recalibrated.passed)
                )
            ):
                raise ModelError("label verdict disagrees with its metrics or sufficiency")
            if (verdict.knots > 0) != (verdict.recalibrated is not None) or verdict.knots < 0:
                raise ModelError("knot count disagrees with recalibration metrics")
        expected = {
            verdict.label: verdict.knots for verdict in document.labels if verdict.knots > 0
        }
        if set(self.knots) != set(expected):
            raise ModelError("knot labels disagree with the document")
        frozen = {}
        for label in Label:
            if label in expected:
                x, y = self.knots[label]
                _check_knots(x, y)
                if len(x) != expected[label]:
                    raise ModelError("knot length disagrees with the document")
                frozen[label] = (_ro(x), _ro(y))
        outcome = document.outcome
        if (
            outcome.level != config.outcome_level
            or not 0 <= outcome.covered <= outcome.n <= config.outcome_bars
            or outcome.passed != outcome_pass(outcome.n, outcome.covered, config)
        ):
            raise ModelError("outcome level, counts or pass disagree with the config")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", m) is None for m in (*document.models, document.data_hash)
        ):
            raise ModelError("calibration artifact hash is invalid or does not match")
        object.__setattr__(self, "knots", MappingProxyType(frozen))

    @property
    def as_of(self) -> datetime:
        """`document.as_of`, in UTC."""
        return ensure_utc(self.document.as_of)

    @property
    def source(self) -> CalibrationSource:
        return self.document.source

    def verdict(self, label: Label) -> LabelVerdict:
        """The label's LabelVerdict."""
        return self.document.labels[tuple(Label).index(label)]

    def passed(self, label: Label) -> bool:
        """The label's verdict: `verdict(label).passed`."""
        return self.verdict(label).passed

    def cap_fraction(self, label: Label) -> float:
        """1.0 when `passed(label)` and the outcome check passed, else `config.failed_cap`
        (module docstring, "Using one")."""
        return (
            1.0
            if self.passed(label) and self.document.outcome.passed
            else self.document.config.failed_cap
        )

    def calibrated(self, label: Label, p: float) -> float:
        """P̃ for a filtered probability `p` of `label`: `p` clipped to [0, 1], then mapped by
        `interpolate` through the label's knots when it has any, else returned as clipped. A
        label's probability is a sum of state probabilities, which can leave [0, 1] by
        rounding, so `p` within PROBABILITY_SLACK outside it is clipped; ModelError for
        anything else (not finite, or further outside)."""
        if (
            not isinstance(p, float)
            or not np.isfinite(p)
            or p < -PROBABILITY_SLACK
            or p > 1 + PROBABILITY_SLACK
        ):
            raise ModelError(
                "filtered probability must be finite and within probability slack of [0, 1]"
            )
        p = min(1.0, max(0.0, p))
        if label in self.knots:
            x, y = self.knots[label]
            # Construction already validated and froze these knots; only p needs checking here.
            return float(np.interp(p, x, y))
        return p

    def require_before(self, tau: datetime) -> None:
        """T-NL2 (ARCHITECTURE §7.4): return when `as_of < tau`; raise ModelError, naming both
        times, when `tau <= as_of`. ClockError for a naive `tau`."""
        tau = ensure_utc(tau)
        if not self.as_of < tau:
            raise ModelError(f"calibration as of {self.as_of} used at {tau}")

    def artifact_hash(self) -> str:
        """The artifact hash of the module docstring, 64 lowercase hex characters.
        Deterministic: it depends only on `document` and the knots' values."""
        return sha256_hex(
            {
                "calibration": self.document,
                "knots": {
                    n: {
                        "dtype": "float64",
                        "shape": list(a.shape),
                        "sha256": hashlib.sha256(
                            np.ascontiguousarray(a, dtype="<f8").tobytes()
                        ).hexdigest(),
                    }
                    for n, a in _arrays(self).items()
                },
            }
        )


def save_calibration(calibration: Calibration, root: Path) -> Path:
    """Write the artifact to `root/<artifact_hash>/` and return that directory, exactly as
    `quant.hmm.model.save_model` saves a model: staged in a temporary `root/.calibration-*/`
    and renamed into place, each file, the staged directory and `root` fsynced, writers holding
    an exclusive flock on `root` (a stale `.calibration-*` entry found under it was left by a
    crash and is removed), `root` created if missing. Saving a calibration whose directory
    exists checks it with `load_calibration` and returns it unchanged (idempotent, and never an
    overwrite); an existing directory that doesn't load as this calibration raises ModelError
    and is left alone. Any OSError is a ModelError."""
    digest = calibration.artifact_hash()
    directory = root / digest
    try:
        root.mkdir(parents=True, exist_ok=True)
        lock = os.open(root, os.O_RDONLY)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if directory.exists():
                load_calibration(directory)
                return directory
            for stale in root.glob(".calibration-*"):
                if stale.is_dir() and not stale.is_symlink():
                    shutil.rmtree(stale)
                else:
                    stale.unlink()
            with tempfile.TemporaryDirectory(prefix=".calibration-", dir=root) as temporary:
                staged = Path(temporary) / digest
                staged.mkdir()
                with zipfile.ZipFile(
                    staged / KNOTS_FILE, "w", compression=zipfile.ZIP_STORED
                ) as archive:
                    for name, array in _arrays(calibration).items():
                        info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                        with archive.open(info, "w") as member:
                            np.lib.format.write_array(member, array, allow_pickle=False)
                (staged / DOCUMENT_FILE).write_bytes(
                    canonical_dumps({"artifact_hash": digest, "calibration": calibration.document})
                )
                for path in (staged / KNOTS_FILE, staged / DOCUMENT_FILE, staged):
                    _fsync(path)
                staged.rename(directory)
                _fsync(root)
        finally:
            os.close(lock)
        return directory
    except (OSError, ValueError) as exc:
        raise ModelError(f"cannot save calibration artifact at {directory}") from exc


def load_calibration(directory: Path) -> Calibration:
    """The calibration saved in `directory`, every knot bit for bit as saved.

    Reads knots.npz only with `numpy.load(path, allow_pickle=False)`, so an object array is
    refused, never unpickled. Raises ModelError when a file is missing or unreadable;
    calibration.json isn't the canonical JSON of exactly {"artifact_hash", "calibration"}; the
    document doesn't parse (`CalibrationDocument.model_validate_json`, strict); knots.npz holds a
    member other than the `<LABEL>_x.npy` / `<LABEL>_y.npy` pairs the document counts, or an
    array that isn't float64; the Calibration fails its own checks; or the hash recomputed from
    what was read differs from calibration.json's `artifact_hash` or from the directory's
    name."""
    try:
        raw = (directory / DOCUMENT_FILE).read_bytes()
        payload = canonical_loads(raw)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"artifact_hash", "calibration"}
            or canonical_dumps(payload) != raw
        ):
            raise ModelError("calibration document must be canonical JSON with the expected keys")
        document = CalibrationDocument.model_validate_json(canonical_dumps(payload["calibration"]))
        expected = [
            f"{verdict.label.value}_{a}.npy"
            for verdict in document.labels
            if verdict.knots > 0
            for a in "xy"
        ]
        with zipfile.ZipFile(directory / KNOTS_FILE) as archive:
            if sorted(archive.namelist()) != sorted(expected):
                raise ModelError("knot archive members disagree with the document")
        knots = {}
        with np.load(directory / KNOTS_FILE, allow_pickle=False) as archive:
            for verdict in document.labels:
                if verdict.knots > 0:
                    x, y = archive[f"{verdict.label.value}_x"], archive[f"{verdict.label.value}_y"]
                    if x.dtype != np.float64 or y.dtype != np.float64:
                        raise ModelError("knot arrays must be float64")
                    knots[verdict.label] = (x, y)
        calibration = Calibration(document, knots)
        if (
            calibration.artifact_hash() != payload["artifact_hash"]
            or directory.name != payload["artifact_hash"]
        ):
            raise ModelError("calibration artifact hash is invalid or does not match")
        return calibration
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile, ClockError) as exc:
        raise ModelError("cannot load calibration artifact") from exc


def _ro(a: Floats) -> Floats:
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _check_knots(x: Floats, y: Floats) -> None:
    if not (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and x.dtype == np.float64
        and y.dtype == np.float64
        and x.ndim == 1
        and y.shape == x.shape
        and len(x) >= 1
        and np.isfinite(x).all()
        and np.isfinite(y).all()
        and (np.diff(x) > 0).all()
        and (np.diff(y) >= 0).all()
        and (y >= 0).all()
        and (y <= 1).all()
    ):
        raise ModelError("knots must be an isotonic map")


def _arrays(self: Calibration) -> dict[str, Floats]:
    out = {}
    for label in Label:
        if label in self.knots:
            x, y = self.knots[label]
            out[f"{label.value}_x"] = x
            out[f"{label.value}_y"] = y
    return out


def _fsync(path: Path) -> None:
    """Flush a file's or a directory's contents and entries to the device."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
