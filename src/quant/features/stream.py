"""Streaming features F1–F5 over hourly bars (spec §3, D3, ARCHITECTURE §4.2).

A `FeatureStream` consumes one symbol's hourly bars in time order, one `update` per bar, and
returns that bar's `FeatureRow`. Its state is small and fixed-size: per-slot histories of the
last N = `slot_sessions` sessions, the last B = `rvol.bars` returns, two EMAs and the previous
close. It snapshots to canonical JSON for EngineState (`FeatureState`), and a restored stream
continues bit for bit. A model's window is featurized by the same stream (`compute_features`),
and a decision runs it through the point-in-time view (`features_for`), so live trading, the
backtest and model fits share one implementation of every feature.

**Definitions.** "The bars" are the bars the stream has consumed, in order, and "the previous
bar" is the last one consumed before bar t. For bar t with close C_t, high H_t, low L_t and
volume V_t:

- r_t = ln(C_t / C_{t-1}), the log return from the previous bar's close. The first bar of a
  session takes C_{t-1} from the bar consumed before it, normally the previous session's CLOSE
  bar (a half day's too), so the opening bar's return carries the overnight gap. The stream's
  first bar has no return.
- x_t = (H_t - L_t) / C_{t-1}, the range ratio. The stream's first bar has none.
- Sessions are numbered 1, 2, ... in the order the stream consumes them: a bar whose session
  differs from the previous bar's starts the next number. A half day is a session like any
  other; a session none of whose bars was consumed gets no number.
- A slot's history of a quantity: its values on the bars consumed before t with the same
  `HourlyBar.slot` whose sessions are among the prior N sessions, numbers k - N to k - 1 for
  t's session k (spec §3: "the same time slot over the prior 60 sessions"), oldest first. Each
  session has each slot at most once, so a history holds at most N values. The final 30-minute
  bar of every session, full or half day, is the CLOSE slot, so CLOSE bars are compared only
  with CLOSE bars (ARCHITECTURE §6.1). Half days have no 12:30, 13:30 or 14:30 bar, so for the
  N sessions after one those histories hold N - 1 values. Early on a history holds the values
  of the sessions seen so far.

- F1 ret = r_t / sigma, where sigma = sqrt(mean of r² over the slot's return history): an RMS, with no
  mean subtracted.
- F2 rvol = ln(sqrt(S)), where S = Σ r² over the returns of the last B bars, t's own included
  (all of them while the stream has fewer than B).
- F3 range = ln(x_t) - ln(m), where m = the median of the slot's range history.
- F4 volume = ln(V_t / m), where m = the median of the slot's volume history.
- F5 trend = ln(EMA_fast / EMA_slow), each an EMA of every bar's close with alpha = 2 / (n + 1) for
  its span n: EMA = C at the stream's first bar, then EMA_t = alpha·C_t + (1 - alpha)·EMA_{t-1}.

The median of an even count is the mean of its two middle values (as `numpy.median` and
`statistics.median` give it), and F3 takes the log of the median, not the median of the logs.

F1, F3 and F4 read the histories as they stood before bar t (the prior sessions), and F2 reads
the returns after adding r_t. Then every bar but the stream's first adds r_t to its slot's return
history and to the last-B returns, and x_t to its slot's range history; every bar adds V_t to its
slot's volume history; each history drops its oldest value beyond its bound.

**Undefined values.** A value is NaN in the row when its formula has no finite value: no
previous close (the stream's first bar), an empty history, sigma = 0, S = 0, or the log of zero (a
bar with H == L or V == 0, or a zero median). Computing one must neither raise nor warn: the tests
turn numpy warnings into errors, and Python's float division and `math.log` raise on zero, so
test the operands first. NaN is never stored in the state: x_t = 0 and V_t = 0 are stored as the
numbers they are. Every value from a bar is stored as a float64, so an int volume or price can't
change the state's canonical JSON when it round-trips.

**Overflow.** Valid prices can still be too far apart for float64: C_t / C_{t-1} rounds to
infinity or to 0, x_t to infinity, an EMA leaves the finite positive numbers, or a row value
(V_t / m, r_t / sigma, EMA_fast / EMA_slow) is infinite. That is no feature value, so `update`
refuses the bar like bad data: FeatureError, with the state as it was. Neither NaN nor an
infinity from arithmetic reaches the state, and no row value is ever infinite.

**Warm-up.** Rows of sessions 1 to WARMUP_SESSIONS = 60 are not warm and are dropped (spec §3),
whatever N is. With N > 60, or a slow EMA span of hundreds of bars, the first warm rows still read
short histories, or EMAs that carry their seed (the first close). Fits and live trading agree all
the same, because both stream from the start of history (see `compute_features`).

**Missing and bad bars.** An incomplete bar (its last minute missing: `complete` False) is
missing data, and decisions treat it as missing. A bar with a price that isn't finite and
positive, prices out of order (low above open or close, high below either), a negative or
non-finite volume, or a t_open not after the previous bar's is bad data. `update` refuses both,
and a bar that overflows, with FeatureError and leaves the state as it was. The stream then
continues from its previous bar, so the next bar's return and range span the gap. A missing or
NaN feature at decision time is a model error, so rule 6 applies (spec §3): `FeatureRow.vector`
and `features_for` raise FeatureError, a ModelError. A model fit has no τ: `compute_features`
drops a warm row holding a NaN, and logs it, rather than abort the fit.

**Determinism.** Plain float64 arithmetic on the bars' values, with no clock, randomness or I/O,
so identical bars give bit-identical rows (spec §3, ARCHITECTURE §14). Recompute each window
statistic from the stored values on every update rather than carrying running sums, so the
state holds only values taken from bars and a restored stream continues bit for bit. The tests
compare the stream with a slow reference within 1e-12, so the summation order is free.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final

import numpy as np
import structlog

from quant.core.calendar import BarSpec, Slot
from quant.core.epoch import from_us, to_us
from quant.core.errors import ConfigError, FeatureError
from quant.core.model import StrictModel
from quant.data.frames import Floats, HourlyBar, Ints
from quant.data.policy import Purpose
from quant.data.view import PointInTimeView
from quant.features.menu import FeatureConfig

WARMUP_SESSIONS: Final = 60


@dataclass(frozen=True)
class FeatureRow:
    """The features of one bar, as the stream computed them when it consumed the bar.

    `session`, `slot`, `t_open` and `available_at` are the bar's. `values` follow
    `FeatureConfig.columns`, one float per transform that is on, NaN where a value is undefined.
    `warm` is False for the bars of the stream's first WARMUP_SESSIONS sessions.
    """

    session: date
    slot: Slot
    t_open: datetime
    available_at: datetime
    values: tuple[float, ...]
    warm: bool

    def vector(self) -> Floats:
        """The values as a new 1-D float64 array, for the scaler and the filter.

        Raises FeatureError when the row is in the warm-up or any value is NaN: a missing or NaN
        feature is a model error, so rule 6 applies (spec §3).
        """
        values = np.array(self.values, dtype=np.float64)
        if not self.warm or not np.isfinite(values).all():
            raise FeatureError(f"features at {self.t_open.isoformat()} are not warm and finite")
        return values


class FeatureState(StrictModel):
    """A FeatureStream's state as plain JSON data, for EngineState (ARCHITECTURE §4.3).

    Times are epoch microseconds UTC, sessions are ISO dates, and every float is finite, so the
    state has one canonical JSON form, and `FeatureState.model_validate(canonical_loads(...))`
    reads it back. Every list is oldest first. Each slot map has all seven `Slot` values as keys
    ("09:30" … "14:30", "CLOSE") from the first snapshot on. The histories of a transform that is
    off stay empty, and its EMAs stay None. A stream that has consumed no bar has every optional
    field None, `sessions` 0 and every list empty.

    - `config_hash`: the `FeatureConfig.config_hash()` of the config the stream runs under.
    - `last_t_open`, `last_session`, `prev_close`: the last bar consumed (its t_open, its
      session and its close), all None before the first.
    - `sessions`: distinct sessions consumed, the warm-up count.
    - `returns`: the returns of the last up-to-B bars (F2).
    - `slot_numbers`: per slot, the session numbers of the slot's bars still in its window,
      oldest first: at most N of them, in order, each from `sessions` - N to `sessions`.
    - `slot_returns`, `slot_ranges`, `slot_volumes`: per slot, the histories of r, x and V
      (F1, F3, F4). Each holds the values of the slot's last len(history) bars in
      `slot_numbers` (the stream's first bar has a volume but no r or x).
    - `ema_fast`, `ema_slow`: F5's EMAs after the last bar.
    """

    config_hash: str
    last_t_open: int | None
    last_session: str | None
    sessions: int
    prev_close: float | None
    returns: list[float]
    slot_numbers: dict[str, list[int]]
    slot_returns: dict[str, list[float]]
    slot_ranges: dict[str, list[float]]
    slot_volumes: dict[str, list[float]]
    ema_fast: float | None
    ema_slow: float | None


class FeatureStream:
    """Streaming F1–F5 for one symbol's hourly bars, defined in this module's docstring.

    The stream is mutable: `update` advances it in place. EngineState keeps `snapshot()`, and
    `restore` rebuilds the stream from one.
    """

    def __init__(self, config: FeatureConfig) -> None:
        """A stream under `config` that has consumed no bar."""
        self._config = config
        self._state = FeatureState(
            config_hash=config.config_hash(),
            last_t_open=None,
            last_session=None,
            sessions=0,
            prev_close=None,
            returns=[],
            slot_numbers={slot.value: [] for slot in Slot},
            slot_returns={slot.value: [] for slot in Slot},
            slot_ranges={slot.value: [] for slot in Slot},
            slot_volumes={slot.value: [] for slot in Slot},
            ema_fast=None,
            ema_slow=None,
        )

    @property
    def config(self) -> FeatureConfig:
        return self._config

    @property
    def last_t_open(self) -> datetime | None:
        """The t_open (UTC) of the last bar consumed, or None before the first."""
        value = self._state.last_t_open
        return None if value is None else from_us(value)

    @property
    def sessions(self) -> int:
        """Distinct sessions consumed so far. Rows are warm once this exceeds WARMUP_SESSIONS."""
        return self._state.sessions

    def update(self, bar: HourlyBar) -> FeatureRow:
        """Consume the next bar and return its row, warm or not.

        Raises FeatureError, leaving the state exactly as it was, for:

        - an incomplete bar (`complete` False);
        - a bar whose t_open isn't strictly after the last bar's;
        - open, high, low or close not finite and > 0, low > min(open, close), or
          high < max(open, close);
        - a volume that is negative or not finite;
        - a value that isn't a number float64 can hold (`float()` fails);
        - a bar whose features overflow float64 (see "Overflow" in the module docstring).

        Every check runs before the state changes. Every other bar is consumed, even when some
        of its values are undefined (NaN in its row): it is real data, and later bars need its
        close and its place in the histories.
        """
        state, config = self._state, self._config
        t_open = to_us(bar.t_open)
        try:  # float64 from here on, so an int volume or price never reaches the state
            open_, high, low, close, volume = (
                float(value) for value in (bar.open, bar.high, bar.low, bar.close, bar.volume)
            )
        except (TypeError, ValueError, OverflowError):
            raise FeatureError(f"invalid feature bar at {bar.t_open.isoformat()}") from None
        if (
            not bar.complete
            or (state.last_t_open is not None and t_open <= state.last_t_open)
            or any(not math.isfinite(p) or p <= 0 for p in (open_, high, low, close))
            or low > min(open_, close)
            or high < max(open_, close)
            or not math.isfinite(volume)
            or volume < 0
        ):
            raise FeatureError(f"invalid or incomplete feature bar at {bar.t_open.isoformat()}")

        # Everything is computed before the state changes, so a refused bar leaves it as it was.
        n = config.slot_sessions
        session = bar.session.isoformat()
        sessions = state.sessions + (session != state.last_session)  # this bar's session number
        slot = bar.slot.value
        numbers = state.slot_numbers[slot]
        window = len(numbers) - bisect.bisect_left(numbers, sessions - n)  # bars in k - N .. k - 1
        ret_hist = _last(state.slot_returns[slot], window)
        range_hist = _last(state.slot_ranges[slot], window)
        vol_hist = _last(state.slot_volumes[slot], window)
        ret = math.nan
        ratio = math.nan
        returns = state.returns.copy()
        if state.prev_close is not None:
            quotient = close / state.prev_close
            ratio = (high - low) / state.prev_close
            if not (math.isfinite(quotient) and quotient > 0 and math.isfinite(ratio)):
                raise FeatureError(f"feature bar at {bar.t_open.isoformat()} overflows float64")
            ret = math.log(quotient)
            _append(returns, ret, config.rvol.bars)
        sigma = math.sqrt(math.fsum(r * r for r in ret_hist) / len(ret_hist)) if ret_hist else 0.0
        total = math.fsum(r * r for r in returns)
        values = [ret / sigma if sigma > 0 else math.nan, _log(math.sqrt(total))]
        if config.range is not None:
            median_range = statistics.median(range_hist) if range_hist else 0.0
            values.append(_log(ratio) - _log(median_range))
        if config.volume is not None:
            median_volume = statistics.median(vol_hist) if vol_hist else 0.0
            values.append(_log(volume / median_volume) if median_volume > 0 else math.nan)
        fast, slow = state.ema_fast, state.ema_slow
        if config.trend is not None:
            fast = _ema(close, fast, config.trend.fast)
            slow = _ema(close, slow, config.trend.slow)
            if not all(math.isfinite(ema) and ema > 0 for ema in (fast, slow)):
                raise FeatureError(f"feature bar at {bar.t_open.isoformat()} overflows float64")
            values.append(_log(fast / slow))
        if any(math.isinf(value) for value in values):
            raise FeatureError(f"features at {bar.t_open.isoformat()} overflow float64")

        # Histories are updated only after the current row has read the prior slot values. A new
        # session moves every slot's window on. The slot's next bar comes in a later session,
        # whose window starts after session k - N, so this slot keeps at most its last N sessions.
        if sessions != state.sessions:
            for name, slot_numbers in state.slot_numbers.items():
                _trim(
                    slot_numbers,
                    (state.slot_returns[name], state.slot_ranges[name], state.slot_volumes[name]),
                    bisect.bisect_left(slot_numbers, sessions - n),
                )
        histories = (state.slot_returns[slot], state.slot_ranges[slot], state.slot_volumes[slot])
        numbers.append(sessions)
        if state.prev_close is not None:
            histories[0].append(ret)
            if config.range is not None:
                histories[1].append(ratio)
        if config.volume is not None:
            histories[2].append(volume)
        _trim(
            numbers,
            histories,
            max(bisect.bisect_right(numbers, sessions - n), len(numbers) - n),
        )
        self._state = state.model_copy(
            update={
                "last_t_open": t_open,
                "last_session": session,
                "sessions": sessions,
                "prev_close": close,
                "returns": returns,
                "ema_fast": fast,
                "ema_slow": slow,
            }
        )
        return FeatureRow(
            bar.session,
            bar.slot,
            bar.t_open,
            bar.available_at,
            tuple(values),
            sessions > WARMUP_SESSIONS,
        )

    def snapshot(self) -> FeatureState:
        """The state now. Two streams under one config that consumed the same bars give equal
        snapshots with identical canonical JSON."""
        return self._state.model_copy(deep=True)

    @classmethod
    def restore(cls, config: FeatureConfig, state: FeatureState) -> FeatureStream:
        """The stream `state` describes. It continues exactly as the stream that took the
        snapshot would have: the same rows, bit for bit, and the same later snapshots.

        Raises ConfigError when `state.config_hash` isn't `config.config_hash()`, or the state
        doesn't fit the config: a slot map whose keys aren't exactly the seven Slot values, a
        history longer than its bound (N for slot histories, B for `returns`), a non-empty
        history or a non-None EMA for a transform that is off, or some but not all of
        `last_t_open`, `last_session` and `prev_close` set; values out of range: a range or
        volume below 0, or a `last_session` that isn't an ISO date (YYYY-MM-DD); or session
        numbers that don't fit: more than N for a slot, out of order, outside `sessions` - N to
        `sessions`, or none equal to `sessions` (the last bar's session) when a bar has been
        consumed; or histories out of step with them. A slot's volumes, when F4 is on, have one
        value per session number; its returns, and its ranges when F3 is on, have one fewer
        only when its oldest number is 1 and only in one slot (the stream's first bar has no
        return or range), else one per number.
        """
        if state.config_hash != config.config_hash():
            raise ConfigError("feature state belongs to a different config")
        slots = {slot.value for slot in Slot}
        for history, enabled in (
            (state.slot_returns, True),
            (state.slot_ranges, config.range is not None),
            (state.slot_volumes, config.volume is not None),
        ):
            if set(history) != slots or any(
                len(values) > config.slot_sessions
                or (not enabled and values)
                or any(not math.isfinite(value) for value in values)
                for values in history.values()
            ):
                raise ConfigError("slot histories do not fit the feature config")
        if any(
            value < 0
            for history in (state.slot_ranges, state.slot_volumes)
            for values in history.values()
            for value in values
        ):
            raise ConfigError("slot ranges and volumes can't be negative")
        oldest = max(1, state.sessions - config.slot_sessions)
        if set(state.slot_numbers) != slots or any(
            len(numbers) > config.slot_sessions
            or numbers != sorted(numbers)
            or any(not oldest <= number <= state.sessions for number in numbers)
            for numbers in state.slot_numbers.values()
        ):
            raise ConfigError("slot session numbers do not fit the feature config")
        if state.sessions > 0 and not any(
            numbers and numbers[-1] == state.sessions for numbers in state.slot_numbers.values()
        ):
            raise ConfigError("no slot holds the last bar's session")
        lacking = 0  # slots whose returns lack a value: only the stream's first bar's slot
        for slot, numbers in state.slot_numbers.items():
            returns = len(state.slot_returns[slot])
            lacks = len(numbers) - returns
            lacking += lacks
            if (
                lacks not in (0, 1)
                or (lacks and numbers[0] != 1)
                or (config.range is not None and len(state.slot_ranges[slot]) != returns)
                or (config.volume is not None and len(state.slot_volumes[slot]) != len(numbers))
            ):
                raise ConfigError("slot histories are out of step with their session numbers")
        if lacking > 1:
            raise ConfigError("only the stream's first bar lacks a return")
        if len(state.returns) > config.rvol.bars or any(
            not math.isfinite(value) for value in state.returns
        ):
            raise ConfigError("return history does not fit the feature config")
        present = [
            value is not None for value in (state.last_t_open, state.last_session, state.prev_close)
        ]
        if any(present) and not all(present):
            raise ConfigError("last bar time, session and close must be set together")
        if state.last_session is not None and not _iso_date(state.last_session):
            raise ConfigError("the last session must be an ISO date")
        emas = (state.ema_fast, state.ema_slow)
        if config.trend is None and any(value is not None for value in emas):
            raise ConfigError("a disabled trend must have no EMAs")
        if config.trend is not None and any((value is not None) != all(present) for value in emas):
            raise ConfigError("trend EMAs must be set exactly when a bar has been consumed")
        if any(
            value is not None and (not math.isfinite(value) or value <= 0)
            for value in (state.prev_close, *emas)
        ):
            raise ConfigError("previous close and EMAs must be finite and positive")
        if state.sessions < 0 or (state.sessions > 0) != all(present):
            raise ConfigError("session count does not match the last bar")
        if not any(present) and (
            state.returns
            or any(
                values
                for history in (state.slot_returns, state.slot_ranges, state.slot_volumes)
                for values in history.values()
            )
        ):
            raise ConfigError("an empty stream must have empty histories")
        stream = cls(config)
        stream._state = state.model_copy(deep=True)
        return stream


@dataclass(frozen=True)
class FeatureMatrix:
    """Warm feature rows in time order as one float64 array: what the scaler and a model fit
    read (ARCHITECTURE §4.2, §7.3).

    One row per bar: `t_open` and `available_at` are its bar's times in epoch microseconds UTC
    (int64, shape (n,)), and `values` has shape (n, len(columns)).
    """

    columns: tuple[str, ...]
    t_open: Ints
    available_at: Ints
    values: Floats

    def __post_init__(self) -> None:
        """Raise FeatureError unless `columns` is a non-empty tuple of distinct names, `t_open`
        and `available_at` are 1-D int64 arrays of one length n, `t_open` strictly increases,
        and `values` is a float64 array of shape (n, len(columns)) whose every value is finite.
        Zero rows is valid. Then replace the three arrays with read-only copies
        (`object.__setattr__`), so a matrix can't change after it is built."""
        if (
            not isinstance(self.columns, tuple)
            or not self.columns
            or any(not isinstance(name, str) or not name for name in self.columns)
            or len(set(self.columns)) != len(self.columns)
        ):
            raise FeatureError("feature columns must be a non-empty tuple of distinct names")
        for name in ("t_open", "available_at"):
            times = getattr(self, name)
            if not isinstance(times, np.ndarray) or times.dtype != np.int64 or times.ndim != 1:
                raise FeatureError(f"feature {name} must be a 1-D int64 array")
        n = len(self.t_open)
        if len(self.available_at) != n or (self.t_open[1:] <= self.t_open[:-1]).any():
            raise FeatureError("feature times must align and bar opens must strictly increase")
        if (
            not isinstance(self.values, np.ndarray)
            or self.values.dtype != np.float64
            or self.values.shape != (n, len(self.columns))
            or not np.isfinite(self.values).all()
        ):
            raise FeatureError(
                "feature values must be a finite float64 matrix with matching columns"
            )
        freeze_arrays(self, "t_open", "available_at", "values")

    def __len__(self) -> int:
        """The number of rows."""
        return len(self.t_open)

    def window(self, start: datetime, end: datetime) -> FeatureMatrix:
        """A new matrix of the rows whose bars became available in [start, end):
        start <= available_at < end. Naive datetimes raise ClockError."""
        start_us, end_us = to_us(start), to_us(end)
        mask = (self.available_at >= start_us) & (self.available_at < end_us)
        return FeatureMatrix(
            self.columns, self.t_open[mask], self.available_at[mask], self.values[mask]
        )


def compute_features(bars: Sequence[HourlyBar], config: FeatureConfig) -> FeatureMatrix:
    """Run one new FeatureStream over `bars`, in the order given, and keep the warm rows.

    This is how a model's window is featurized (ARCHITECTURE §4.2): the decision path's streaming
    code over the bars a view returned. An incomplete bar is missing data and is skipped: it gets
    no row and leaves the stream untouched, exactly as if it weren't in the list. Any other bar the
    stream refuses raises its FeatureError. No bars, or none warm, gives a matrix with zero rows
    and the config's columns.

    A warm row holding a NaN (a flat bar's F3, a zero volume's F4) is dropped from the matrix and
    logged as a warning, "feature row dropped", with its bar's t_open (ISO) and the names of its
    NaN columns. A model can't be fit on the row, and one such bar must not abort a fit of years
    of bars, or every refit whose window holds it. Its bar was still consumed, so its close and
    its place in the histories count for the rows after it, as they do live. At decision time
    the same row is a model error (spec §3: "a missing or NaN feature at τ"): `vector` and
    `features_for` raise. A window with too few rows left fails later, in `Scaler.fit`.

    A row depends on every bar before it, not only on the last N sessions: F5's EMAs are seeded
    with the first close, and the warm-up counts sessions from the first bar. So a fit passes the
    bars from the start of history (2016-01), the same bars the live stream started from, and
    cuts its training window with `FeatureMatrix.window`. Then the fit and live trading share
    one seed, and a fit's rows equal the live stream's. Rows of `bars[:k]` equal the same rows
    of `bars`: no row reads a later bar (I1).
    """
    stream = FeatureStream(config)
    opens, available, values = [], [], []
    for bar in bars:
        if not bar.complete:
            continue
        row = stream.update(bar)
        if not row.warm:
            continue
        missing = [
            name
            for name, value in zip(config.columns, row.values, strict=True)
            if math.isnan(value)
        ]
        if missing:
            structlog.get_logger().warning(
                "feature row dropped", t_open=row.t_open.isoformat(), columns=missing
            )
            continue
        values.append(row.vector())
        opens.append(to_us(row.t_open))
        available.append(to_us(row.available_at))
    return FeatureMatrix(
        config.columns,
        np.array(opens, dtype=np.int64),
        np.array(available, dtype=np.int64),
        np.array(values, dtype=np.float64).reshape(len(values), len(config.columns)),
    )


def features_for(stream: FeatureStream, view: PointInTimeView, symbol: str, bar: BarSpec) -> Floats:
    """The feature vector for the decision on `bar`, read through the point-in-time view (I1).

    Reads `symbol`'s hourly bars from `view` with purpose OPERATE: all of them for a stream that
    has consumed no bar, else those opening after `stream.last_t_open`. Feeds `stream`, in place
    and oldest first, each of them that opens at or before `bar.t_open`; later ones are left for
    later decisions. Incomplete bars are skipped as missing data, as in `compute_features`.
    Returns the `vector()` of `bar`'s row.

    Raises FeatureError, a model error (rule 6 applies), when `bar` isn't among the bars fed:
    it isn't visible in the view (missing, or not yet available at the view's tau), it is
    incomplete, or the stream had already consumed it or a later bar. Also when its row is in the
    warm-up or holds a NaN, and when the stream refuses a bar for bad data (that FeatureError
    propagates, and feeding stops there). Bars consumed before an error stay consumed, so a
    caller that keeps the stream's snapshot keeps them.
    """
    last = stream.last_t_open
    since = None if last is None else last + timedelta(microseconds=1)
    for candidate in view.hourly(symbol, purpose=Purpose.OPERATE, since=since):
        if candidate.t_open > bar.t_open:
            break
        if not candidate.complete:
            continue
        row = stream.update(candidate)
        if candidate.t_open == bar.t_open:
            return row.vector()
    raise FeatureError(f"decision bar at {bar.t_open.isoformat()} is missing or already consumed")


def freeze_arrays(instance: object, *names: str) -> None:
    """Replace each named array attribute of a frozen dataclass with a read-only copy, from its
    `__post_init__`, so writing to one raises ValueError and the caller's array stays its own.

    The copy is a view of an immutable `bytes` object, so its WRITEABLE flag can't be set back
    to True either (a copy that owned its memory could be)."""
    for name in names:
        values = np.ascontiguousarray(getattr(instance, name))
        frozen = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
        object.__setattr__(instance, name, frozen)


def _log(value: float) -> float:
    """Undefined logarithms (of NaN, zero or less) become missing features without a warning.
    The log of infinity stays infinite, so `update` can refuse an overflow."""
    return math.log(value) if value > 0 else math.nan


def _iso_date(text: str) -> bool:
    """Whether `text` is a date in the form `date.isoformat` writes."""
    try:
        return date.fromisoformat(text).isoformat() == text
    except ValueError:
        return False


def _last(history: list[float], count: int) -> list[float]:
    """The newest `count` values of a history, or all of it when it holds fewer."""
    return history[max(0, len(history) - count) :]


def _ema(close: float, previous: float | None, span: int) -> float:
    if previous is None:
        return close
    alpha = 2.0 / (span + 1)
    return alpha * close + (1.0 - alpha) * previous


def _append(history: list[float], value: float, bound: int) -> None:
    history.append(value)
    if len(history) > bound:
        del history[0]


def _trim(numbers: list[int], histories: Sequence[list[float]], drop: int) -> None:
    """Drop a slot's `drop` oldest bars: their session numbers and their values. A history
    with fewer values than `numbers` lacks them for the oldest bars, so it loses fewer."""
    if drop <= 0:
        return
    for history in histories:
        del history[: max(0, drop - (len(numbers) - len(history)))]
    del numbers[:drop]
