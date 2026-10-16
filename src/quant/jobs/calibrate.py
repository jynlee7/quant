"""The calibration job (spec §9, A4; ARCHITECTURE §7.3 step 5, §7.4; C13). Built in B7b.

A calibration as of an instant scores the forecasts every model made while it governed before
that instant. The job reads the history and the bars, and `quant.eval.calibration` does the
scoring; the refit job builds one as of its cutoff on every pass (`quant.jobs.refit`), and the
nightly report builds one for the report (B16), which moves no pointer.

1. **The spans** (`model_spans`): ops.db's model pointer history. Pointer i governs from its
   effective_at until the next one's, and every span is cut at the as-of time; a span that
   would be empty (a pointer superseded before it took effect, or one taking effect at or after
   the as-of time) is left out. So the spans are exactly the instants `OpsStore.current("model",
   t)` names before the as-of time.
2. **The bars** (`calibration_series`): one read, `store.view(as_of, as_of=as_of,
   policy=policy).hourly(symbol, purpose=Purpose.CALIBRATE, since=<FIT_START's New York
   midnight>)`, and `compute_features(bars, features)` once: the feature stream starts where
   every model's does (`quant.jobs.first_model.training_data`), so each model's rows are the
   ones its filter consumed. Nothing known at or after the as-of time reaches the calibration.
3. **Per span**: the model is loaded from `artifacts / "models" / <hash>` (`load_model`, which
   checks its hash), and `model_series(model, matrix.window(model.fit_start, span.end),
   start=span.start, end=span.end, level=config.outcome_level)` gives its bars; the series are
   concatenated in span order.
4. **The calibration**: `build_calibration(series, as_of=as_of, source=source,
   config=config)`. Saving it (`quant.core.calibration.save_calibration` under
   `artifacts / "calibrations"`) and pointing at it are the caller's: the refit job does both
   in the refit's own transaction.

The calibrate purpose reads FIT, DEV with a trial ticket and FORWARD, never the lockbox
(ARCHITECTURE §6.3, the data policy): B7b runs only inside FIT. Skipping the lockbox span for
live calibrations is B14's. `model_series` is imported into this module, so the truncation
harness can reuse its results across runs (it is a pure function of its arguments).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path

from quant.core.calendar import NEW_YORK
from quant.core.calibration import Calibration, CalibrationConfig, CalibrationSource
from quant.core.clock import ensure_utc
from quant.core.errors import ModelError
from quant.data.market_db import BarStore
from quant.data.policy import FIT_START, DataPolicy, Purpose
from quant.eval.calibration import ForecastSeries, build_calibration, model_series
from quant.features.menu import FeatureConfig
from quant.features.stream import compute_features
from quant.hmm.model import load_model
from quant.store.ops import OpsStore

__all__ = [
    "ModelSpan",
    "build_calibration",
    "calibrate",
    "calibration_series",
    "model_series",
    "model_spans",
]


@dataclass(frozen=True)
class ModelSpan:
    """One model's governing span before an as-of time: [start, end), `artifact` its hash.
    Times are aware UTC."""

    artifact: str
    start: datetime
    end: datetime


def model_spans(ops: OpsStore, *, as_of: datetime) -> tuple[ModelSpan, ...]:
    """The model pointer history before `as_of` as spans (module docstring, step 1), oldest
    first: for consecutive pointers i, i + 1 (in seq order) the span [effective_at_i,
    min(effective_at_{i+1}, as_of)), the last one ending at `as_of`, empty spans left out.
    ClockError for a naive `as_of`."""
    as_of = ensure_utc(as_of)
    rows = ops.pointers("model")
    out = []
    for i, p in enumerate(rows):
        end = rows[i + 1].effective_at if i + 1 < len(rows) else as_of
        end = min(end, as_of)
        if p.effective_at < end:
            out.append(ModelSpan(p.artifact, p.effective_at, end))
    return tuple(out)


def calibration_series(
    store: BarStore,
    ops: OpsStore,
    *,
    as_of: datetime,
    symbol: str,
    features: FeatureConfig,
    artifacts: Path,
    level: float,
    policy: DataPolicy | None = None,
) -> ForecastSeries:
    """The out-of-sample forecasts before `as_of` (module docstring, steps 1–3), through this
    module's `model_series`. `ForecastSeries.empty()` when no model governed before `as_of`.

    Equal for any two stores that agree on what was known before `as_of`. Raises ModelError
    when a span's model artifact is missing or doesn't load, its feature-config hash isn't
    `features.config_hash()` or its columns aren't the config's, or `model_series` refuses (a
    span starting before its model's fit_end is in-sample). PolicyError as the view raises it;
    ClockError for a naive `as_of`."""
    as_of = ensure_utc(as_of)
    spans = model_spans(ops, as_of=as_of)
    if not spans:
        return ForecastSeries.empty()
    origin = datetime.combine(FIT_START, time(), tzinfo=NEW_YORK).astimezone(UTC)
    bars = store.view(as_of, as_of=as_of, policy=policy).hourly(
        symbol, purpose=Purpose.CALIBRATE, since=origin
    )
    matrix = compute_features(bars, features)
    parts = []
    for span in spans:
        model = load_model(artifacts / "models" / span.artifact)
        if (
            model.feature_config_hash != features.config_hash()
            or model.scaler.columns != features.columns
        ):
            raise ModelError(
                "governing model feature config or columns disagree with calibration features"
            )
        parts.append(
            model_series(
                model,
                matrix.window(model.fit_start, span.end),
                start=span.start,
                end=span.end,
                level=level,
            )
        )
    return ForecastSeries.concat(parts)


def calibrate(
    store: BarStore,
    ops: OpsStore,
    *,
    as_of: datetime,
    symbol: str,
    features: FeatureConfig,
    artifacts: Path,
    config: CalibrationConfig,
    source: CalibrationSource = CalibrationSource.BACKTEST,
    policy: DataPolicy | None = None,
) -> Calibration:
    """`build_calibration(calibration_series(store, ops, as_of=as_of, symbol=symbol,
    features=features, artifacts=artifacts, level=config.outcome_level, policy=policy),
    as_of=as_of, source=source, config=config)` (module docstring, step 4). Writes nothing:
    neither the artifact nor a pointer. Raises as those do."""
    return build_calibration(
        calibration_series(
            store,
            ops,
            as_of=as_of,
            symbol=symbol,
            features=features,
            artifacts=artifacts,
            level=config.outcome_level,
            policy=policy,
        ),
        as_of=as_of,
        source=source,
        config=config,
    )
