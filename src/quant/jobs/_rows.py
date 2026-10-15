"""Shared point-in-time training rows for the first model and lifecycle refits."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from itertools import pairwise

import numpy as np

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.epoch import to_us
from quant.core.errors import DataError
from quant.data.frames import HourlyBar
from quant.data.market_db import BarStore
from quant.data.policy import FIT_START, DataPolicy, Purpose
from quant.features.menu import FeatureConfig
from quant.features.stream import compute_features
from quant.hmm.train import TrainingData


def training_rows(
    store: BarStore,
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    first: date,
    last: date,
    features: FeatureConfig,
    policy: DataPolicy | None = None,
    validate: Callable[[list[HourlyBar]], None] | None = None,
) -> TrainingData:
    """Read from FIT start, validate the span, and build aligned features and returns."""
    origin = datetime.combine(FIT_START, time(), tzinfo=NEW_YORK).astimezone(UTC)
    bars = store.view(end, as_of=end, policy=policy).hourly(
        symbol, purpose=Purpose.OPERATE, since=origin
    )
    if validate is not None:
        validate(bars)
    _require_whole_sessions(bars, first, last)
    matrix = compute_features(bars, features).window(start, end)
    by_open = {
        to_us(bar.t_open): (math.log(bar.close / previous.close), bar.session.toordinal())
        for previous, bar in pairwise(bar for bar in bars if bar.complete)
    }
    return TrainingData(
        matrix,
        np.array([by_open[int(t)][0] for t in matrix.t_open], dtype=np.float64),
        np.array([by_open[int(t)][1] for t in matrix.t_open], dtype=np.int64),
        start,
        end,
        features.config_hash(),
    )


def _require_whole_sessions(bars: list[HourlyBar], first: date, last: date) -> None:
    """DataError unless every session of [first, last] has all its calendar hourly bars among
    `bars`, naming the sessions that don't."""
    calendar = TradingCalendar.xnys()
    counts = Counter(bar.session for bar in bars)
    short = [
        day
        for day in calendar.sessions(first, last)
        if counts[day] != len(calendar.session(day).bars)
    ]
    if short:
        shown = ", ".join(day.isoformat() for day in short[:10])
        raise DataError(
            f"the view at fit_end lacks hourly bars of {len(short)} session(s) of the span "
            f"({shown}{', ...' if len(short) > 10 else ''}): known after fit_end, or never "
            "stored; check the store with `quant data quality`"
        )
