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

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

import numpy as np
import numpy.typing as npt
from pydantic import AwareDatetime, model_validator

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
    - `bars`: the out-of-sample bars before `as_of`; `scored`: max(0, bars - settle_bars), the
      oldest bars, which the label metrics use; `sufficient`: scored ≥ min_bars (C13);
    - `data_hash`: `ForecastSeries.data_hash()` of the series scored
      (`quant.eval.calibration`), so two calibrations of different data never share a hash;
    - `config`: the rules it was judged by;
    - `labels`: one LabelVerdict per label, in `tuple(Label)` order;
    - `outcome`: the outcome check."""

    format: int
    source: CalibrationSource
    as_of: AwareDatetime
    models: tuple[str, ...]
    bars: int
    scored: int
    sufficient: bool
    data_hash: str
    config: CalibrationConfig
    labels: tuple[LabelVerdict, ...]
    outcome: OutcomeCheck


Knots = tuple[Floats, Floats]


def interpolate(knots_x: Floats, knots_y: Floats, values: Floats) -> Floats:
    """The isotonic map at `values`: piecewise linear through the knots (x_i, y_i), constant
    y_0 below x_0 and y_m above x_m, exactly `numpy.interp(values, knots_x, knots_y)`, as a new
    float64 array of `values`' shape. This is scikit-learn's `IsotonicRegression(...,
    out_of_bounds="clip").predict` from its fitted thresholds, written without scikit-learn
    so sizing can apply it (K3). A single knot maps everything to y_0.

    Raises ModelError unless the knots are 1-D float64 arrays of one length m ≥ 1, x strictly
    increasing, y nondecreasing within [0, 1], all finite, and `values` is a float64 array of
    finite values in [0, 1]."""
    raise NotImplementedError("B7b: interpolate")


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
        - `scored == max(0, bars - config.settle_bars)` and `sufficient == (scored ≥
          config.min_bars)`; each verdict's `passed` is `sufficient and (raw.passed or
          (recalibrated is not None and recalibrated.passed))`, and its `knots` count is > 0
          exactly when `recalibrated` is not None;
        - `knots` has an entry for exactly the labels whose verdict counts knots, each a pair
          of 1-D float64 arrays of that many points, x strictly increasing, y nondecreasing
          within [0, 1], all finite;
        - `outcome.level == config.outcome_level` and 0 ≤ covered ≤ n ≤ config.outcome_bars;
        - every `models` entry and `data_hash` are 64 lowercase hex characters.

        Store `knots` as a read-only mapping of read-only array copies (as
        `quant.features.stream.freeze_arrays` makes them, without importing it), so nothing
        reachable from a Calibration can change its hash."""
        raise NotImplementedError("B7b: Calibration.__post_init__")

    @property
    def as_of(self) -> datetime:
        """`document.as_of`, in UTC."""
        raise NotImplementedError("B7b: Calibration.as_of")

    @property
    def source(self) -> CalibrationSource:
        raise NotImplementedError("B7b: Calibration.source")

    def verdict(self, label: Label) -> LabelVerdict:
        """The label's LabelVerdict."""
        raise NotImplementedError("B7b: Calibration.verdict")

    def passed(self, label: Label) -> bool:
        """The label's verdict: `verdict(label).passed`."""
        raise NotImplementedError("B7b: Calibration.passed")

    def cap_fraction(self, label: Label) -> float:
        """1.0 when `passed(label)` and the outcome check passed, else `config.failed_cap`
        (module docstring, "Using one")."""
        raise NotImplementedError("B7b: Calibration.cap_fraction")

    def calibrated(self, label: Label, p: float) -> float:
        """P̃ for a filtered probability `p` of `label`: `p` clipped to [0, 1], then mapped by
        `interpolate` through the label's knots when it has any, else returned as clipped. A
        label's probability is a sum of state probabilities, which can leave [0, 1] by
        rounding, so `p` within PROBABILITY_SLACK outside it is clipped; ModelError for
        anything else (not finite, or further outside)."""
        raise NotImplementedError("B7b: Calibration.calibrated")

    def require_before(self, tau: datetime) -> None:
        """T-NL2 (ARCHITECTURE §7.4): return when `as_of < tau`; raise ModelError, naming both
        times, when `tau <= as_of`. ClockError for a naive `tau`."""
        raise NotImplementedError("B7b: Calibration.require_before")

    def artifact_hash(self) -> str:
        """The artifact hash of the module docstring, 64 lowercase hex characters.
        Deterministic: it depends only on `document` and the knots' values."""
        raise NotImplementedError("B7b: Calibration.artifact_hash")


def save_calibration(calibration: Calibration, root: Path) -> Path:
    """Write the artifact to `root/<artifact_hash>/` and return that directory, exactly as
    `quant.hmm.model.save_model` saves a model: staged in a temporary `root/.calibration-*/`
    and renamed into place, each file, the staged directory and `root` fsynced, writers holding
    an exclusive flock on `root` (a stale `.calibration-*` entry found under it was left by a
    crash and is removed), `root` created if missing. Saving a calibration whose directory
    exists checks it with `load_calibration` and returns it unchanged (idempotent, and never an
    overwrite); an existing directory that doesn't load as this calibration raises ModelError
    and is left alone. Any OSError is a ModelError."""
    raise NotImplementedError("B7b: save_calibration")


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
    raise NotImplementedError("B7b: load_calibration")
