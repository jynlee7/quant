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
2. **The lockbox is skipped** (ARCHITECTURE §6.3, A2, C1; `scored_spans`): each span is cut
   at the lockbox, [the lockbox's first New York midnight, the midnight after the seal), or
   to the end of time while it is unsealed. A span the lockbox cuts becomes two pieces, the
   one before it and the one after it (FORWARD), and each piece is a span of its own: its
   smoother stops at the piece's end, and it leaves out its own newest bars (C13). No bar
   whose available_at falls in the lockbox is scored, and no target before it is smoothed
   with the lockbox's rows.
3. **The policy** (`calibration_series`): every piece's New York days, [the piece's start,
   its end - 1 µs], are checked with purpose CALIBRATE and the caller's trial `ticket`
   (`DataPolicy.check`) before anything is read: FIT freely, DEV only with a ticket the
   policy's ledger issued that covers the piece's DEV days (C11), FORWARD freely. A piece in
   DEV without a ticket raises PolicyError, recorded in the ledger as every refusal is.
4. **The bars**: one read, `store.view(as_of, as_of=as_of, policy=policy).hourly(symbol,
   purpose=Purpose.OPERATE, since=<FIT_START's New York midnight>)`, and
   `compute_features(bars, features)` once: the feature stream starts where every model's
   does (`quant.jobs.first_model.training_data`), so each model's rows are the ones its filter
   consumed. The rows feed the filter, which runs through the lockbox as the engine's does
   (operate, C1); only the pieces' bars are scored, and step 3 checked those. Nothing known
   at or after the as-of time reaches the calibration.
5. **Per piece**: the model is loaded from `artifacts / "models" / <hash>` (`load_model`, which
   checks its hash), and `model_series(model, matrix.window(model.fit_start, piece.end),
   start=piece.start, end=piece.end, level=config.outcome_level)` gives its bars; the series
   are concatenated in order, one span each.
6. **The calibration**: `build_calibration(series, as_of=as_of, source=source,
   config=config)`. Saving it (`quant.core.calibration.save_calibration` under
   `artifacts / "calibrations"`) and pointing at it are the caller's: the refit job does both
   in the refit's own transaction.

The calibrate purpose reads FIT, DEV with a trial ticket and FORWARD, never the lockbox
(ARCHITECTURE §6.3, the data policy). `model_series` is imported into this module, so tests
can plant a leak in it by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from quant.core.calendar import NEW_YORK
from quant.core.calibration import Calibration, CalibrationConfig, CalibrationSource
from quant.core.clock import ensure_utc
from quant.core.errors import ModelError
from quant.data.market_db import BarStore
from quant.data.policy import FIT_START, DataPolicy, Purpose, ScoringTicket, Segments
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
    "scored_spans",
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


def _midnight(day: date) -> datetime:
    """New York midnight starting `day`, in UTC."""
    return datetime.combine(day, time(), tzinfo=NEW_YORK).astimezone(UTC)


def scored_spans(spans: tuple[ModelSpan, ...], segments: Segments) -> tuple[ModelSpan, ...]:
    """`spans` with the lockbox cut out (module docstring, step 2), oldest first: each span's
    parts before [the lockbox's first New York midnight, the New York midnight after
    `segments.lockbox_end`) and after it (none while the lockbox is unsealed), empty parts left
    out."""
    low = _midnight(segments.lockbox_start)
    high = None if segments.lockbox_end is None else _midnight(segments.lockbox_end + timedelta(1))
    out = []
    for span in spans:
        if span.start < low:
            out.append(ModelSpan(span.artifact, span.start, min(span.end, low)))
        if high is not None and span.end > high:
            out.append(ModelSpan(span.artifact, max(span.start, high), span.end))
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
    ticket: ScoringTicket | None = None,
) -> ForecastSeries:
    """The out-of-sample forecasts before `as_of` (module docstring, steps 1–5), through this
    module's `model_series`, one span per piece. `ForecastSeries.empty()` when no model
    governed before `as_of` outside the lockbox. `policy` is `DataPolicy()` when None; `ticket`
    is the trial ticket a piece in DEV needs (C11).

    Equal for any two stores that agree on what was known before `as_of`. Raises ModelError
    when a span's model artifact is missing or doesn't load, its feature-config hash isn't
    `features.config_hash()` or its columns aren't the config's, or `model_series` refuses (a
    span starting before its model's fit_end is in-sample). PolicyError (TicketError) as the
    policy raises it for a piece, before anything is read; ClockError for a naive `as_of`."""
    as_of = ensure_utc(as_of)
    policy = DataPolicy() if policy is None else policy
    pieces = scored_spans(model_spans(ops, as_of=as_of), policy.segments)
    if not pieces:
        return ForecastSeries.empty()
    for piece in pieces:
        policy.check(
            Purpose.CALIBRATE,
            piece.start.astimezone(NEW_YORK).date(),
            (piece.end - timedelta(microseconds=1)).astimezone(NEW_YORK).date(),
            ticket=ticket,
        )
    bars = store.view(as_of, as_of=as_of, policy=policy).hourly(
        symbol, purpose=Purpose.OPERATE, since=_midnight(FIT_START)
    )
    matrix = compute_features(bars, features)
    parts = []
    for span in pieces:
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
    ticket: ScoringTicket | None = None,
) -> Calibration:
    """`build_calibration(calibration_series(store, ops, as_of=as_of, symbol=symbol,
    features=features, artifacts=artifacts, level=config.outcome_level, policy=policy,
    ticket=ticket), as_of=as_of, source=source, config=config)` (module docstring, step 6).
    Writes nothing: neither the artifact nor a pointer. Raises as those do."""
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
            ticket=ticket,
        ),
        as_of=as_of,
        source=source,
        config=config,
    )
