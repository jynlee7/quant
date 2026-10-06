"""Slow reference features F1–F5, straight from spec §3 as plain loops, plus helpers (B3).

The reference is the oracle the streaming features must match within 1e-12. Nothing in it
streams: every bar's features are recomputed from the whole list of bars before it. Sums use
math.fsum and each EMA its closed form, so it is as exact as float64 allows. It reads only the
config's fields, never the code under test. It takes the bars a stream would consume: complete,
valid and in time order.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime

import numpy as np
import numpy.typing as npt

from quant.core.calendar import TradingCalendar
from quant.core.errors import FeatureError
from quant.core.schedule import Timing
from quant.data.bars import build_hourly
from quant.data.frames import HourlyBar
from quant.data.synthetic import synthetic_minutes
from quant.features.menu import FeatureConfig
from quant.features.stream import FeatureRow, FeatureStream

WARMUP = 60  # spec §3: the first 60 sessions are dropped
NAN = math.nan


@dataclass(frozen=True)
class Reference:
    columns: tuple[str, ...]
    values: npt.NDArray[np.float64]  # one row per bar, NaN where a value is undefined
    warm: tuple[bool, ...]


def synthetic_bars(start: date, end: date, *, seed: int = 11) -> list[HourlyBar]:
    """Hourly bars built by the B2 builder from seeded synthetic minutes on the real calendar."""
    calendar = TradingCalendar.xnys()
    return build_hourly(calendar, synthetic_minutes(calendar, start, end, seed=seed), Timing())


def stream_rows(bars: Iterable[HourlyBar], config: FeatureConfig) -> dict[datetime, FeatureRow]:
    """One stream's rows by bar t_open. A bar the stream refuses gets no row, and the stream
    goes on to the next."""
    stream = FeatureStream(config)
    rows = {}
    for bar in bars:
        try:
            rows[bar.t_open] = stream.update(bar)
        except FeatureError:
            continue
    return rows


def bits(row: FeatureRow) -> tuple[bool, bytes]:
    """A row's warm flag and the exact bits of its values, NaN included."""
    return row.warm, np.asarray(row.values, dtype=np.float64).tobytes()


def with_prices(bar: HourlyBar, factor: float) -> HourlyBar:
    """The bar with every price scaled by `factor`: still a valid bar."""
    return replace(
        bar,
        open=bar.open * factor,
        high=bar.high * factor,
        low=bar.low * factor,
        close=bar.close * factor,
    )


def _log(value: float) -> float:
    """ln, or NaN where it has no finite value (zero, negative or NaN)."""
    return math.log(value) if value > 0 else NAN


def _ema(closes: Sequence[float], i: int, span: int) -> float:
    """EMA_i in closed form: (1 - alpha)^i·C_0 + Σ_{j=1..i} alpha(1 - alpha)^(i-j)·C_j, alpha = 2 / (span + 1).
    This is the recursion seeded with the first close, unrolled."""
    alpha = 2.0 / (span + 1)
    keep = 1.0 - alpha
    terms = [keep**i * closes[0]]
    terms.extend(alpha * keep ** (i - j) * closes[j] for j in range(1, i + 1))
    return math.fsum(terms)


def reference_features(bars: Sequence[HourlyBar], config: FeatureConfig) -> Reference:
    """Every bar's features, warm or not, with spec §3's definitions as the stream module states
    them. A bar's slot history is its slot's last N bars before it."""
    n_hist = config.slot_sessions
    n_rvol = config.rvol.bars
    names = ["ret", "rvol"]
    if config.range is not None:
        names.append("range")
    if config.volume is not None:
        names.append("volume")
    if config.trend is not None:
        names.append("trend")

    closes = [bar.close for bar in bars]
    # The stream's first bar has no previous close, so no return and no range ratio.
    r = [NAN] + [math.log(closes[i] / closes[i - 1]) for i in range(1, len(bars))]
    x = [NAN] + [(bars[i].high - bars[i].low) / closes[i - 1] for i in range(1, len(bars))]

    rows: list[list[float]] = []
    warm: list[bool] = []
    sessions: list[date] = []
    for i, bar in enumerate(bars):
        if not sessions or sessions[-1] != bar.session:
            sessions.append(bar.session)
        warm.append(len(sessions) > WARMUP)

        same_slot = [j for j in range(i) if bars[j].slot == bar.slot]
        ret_hist = [r[j] for j in same_slot if j > 0][-n_hist:]
        range_hist = [x[j] for j in same_slot if j > 0][-n_hist:]
        vol_hist = [bars[j].volume for j in same_slot][-n_hist:]

        values: dict[str, float] = {}
        # F1: the return over the RMS of the slot's earlier returns.
        sigma = math.sqrt(math.fsum(v * v for v in ret_hist) / len(ret_hist)) if ret_hist else 0.0
        values["ret"] = r[i] / sigma if i > 0 and sigma > 0 else NAN
        # F2: ln sqrt of the summed squared returns of the last B bars, this one included.
        last = [r[j] for j in range(max(1, i - n_rvol + 1), i + 1)]
        total = math.fsum(v * v for v in last)
        values["rvol"] = math.log(math.sqrt(total)) if total > 0 else NAN
        # F3: the log range ratio minus the log of the slot's median ratio.
        if i > 0 and range_hist:
            values["range"] = _log(x[i]) - _log(statistics.median(range_hist))
        else:
            values["range"] = NAN
        # F4: the log of the volume over the slot's median volume.
        median_volume = statistics.median(vol_hist) if vol_hist else 0.0
        values["volume"] = _log(bar.volume / median_volume) if median_volume > 0 else NAN
        # F5: the log ratio of the fast EMA to the slow one.
        if config.trend is not None:
            fast = _ema(closes, i, config.trend.fast)
            slow = _ema(closes, i, config.trend.slow)
            values["trend"] = math.log(fast / slow)
        rows.append([values[name] for name in names])

    array = np.array(rows, dtype=np.float64).reshape(len(bars), len(names))
    return Reference(tuple(names), array, tuple(warm))
