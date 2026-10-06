"""Streaming features F1–F5 over hourly bars (spec §3, D3, ARCHITECTURE §4.2).

A `FeatureStream` consumes one symbol's hourly bars in time order, one `update` per bar, and
returns that bar's `FeatureRow`. Its state is small and fixed-size: per-slot histories of at most
N = `slot_sessions` values, the last B = `rvol.bars` returns, two EMAs and the previous close. It
snapshots to canonical JSON for EngineState (`FeatureState`), and a restored stream continues bit
for bit. A model's window is featurized by the same stream (`compute_features`), and a decision
runs it through the point-in-time view (`features_for`), so live trading, the backtest and model
fits share one implementation of every feature.

**Definitions.** "The bars" are the bars the stream has consumed, in order, and "the previous
bar" is the last one consumed before bar t. For bar t with close C_t, high H_t, low L_t and
volume V_t:

- r_t = ln(C_t / C_{t-1}), the log return from the previous bar's close. The first bar of a
  session takes C_{t-1} from the bar consumed before it, normally the previous session's CLOSE
  bar (a half day's too), so the opening bar's return carries the overnight gap. The stream's
  first bar has no return.
- x_t = (H_t - L_t) / C_{t-1}, the range ratio. The stream's first bar has none.
- A slot's history of a quantity: its values on the most recent bars consumed before t with the
  same `HourlyBar.slot`, at most N of them, oldest first. Each session has each slot at most
  once, so on full days this is the slot's previous N sessions. The final 30-minute bar of every
  session, full or half day, is the CLOSE slot, so CLOSE bars are compared only with CLOSE bars
  (ARCHITECTURE §6.1). Half days have no 12:30, 13:30 or 14:30 bar, so those histories reach
  back past them. Until N values exist, a history holds the k < N values seen so far.

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
numbers they are.

**Warm-up.** The stream counts the distinct sessions it has consumed (a bar whose session differs
from the previous bar's starts a new one). Rows of the first WARMUP_SESSIONS = 60 sessions are not
warm and are dropped (spec §3), whatever N is.

**Missing and bad bars.** An incomplete bar (its last minute missing: `complete` False) is
missing data, and decisions treat it as missing. A bar with a price that isn't finite and
positive, prices out of order (low above open or close, high below either), a negative or
non-finite volume, or a t_open not after the previous bar's is bad data. `update` refuses both
with FeatureError and leaves the state as it was. The stream then continues from its previous
bar, so the next bar's return and range span the gap. A missing or NaN feature at decision time
is a model error, so rule 6 applies (spec §3): `FeatureRow.vector` and `features_for` raise
FeatureError, a ModelError.

**Determinism.** Plain float64 arithmetic on the bars' values, with no clock, randomness or I/O,
so identical bars give bit-identical rows (spec §3, ARCHITECTURE §14). Recompute each window
statistic from the stored values on every update rather than carrying running sums, so the
state holds only values taken from bars and a restored stream continues bit for bit. The tests
compare the stream with a slow reference within 1e-12, so the summation order is free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

from quant.core.calendar import BarSpec, Slot
from quant.core.model import StrictModel
from quant.data.frames import Floats, HourlyBar, Ints
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
        raise NotImplementedError("red:B3")


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
    - `slot_returns`, `slot_ranges`, `slot_volumes`: per slot, the up-to-N histories of r, x and
      V (F1, F3, F4).
    - `ema_fast`, `ema_slow`: F5's EMAs after the last bar.
    """

    config_hash: str
    last_t_open: int | None
    last_session: str | None
    sessions: int
    prev_close: float | None
    returns: list[float]
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
        raise NotImplementedError("red:B3")

    @property
    def config(self) -> FeatureConfig:
        raise NotImplementedError("red:B3")

    @property
    def last_t_open(self) -> datetime | None:
        """The t_open (UTC) of the last bar consumed, or None before the first."""
        raise NotImplementedError("red:B3")

    @property
    def sessions(self) -> int:
        """Distinct sessions consumed so far. Rows are warm once this exceeds WARMUP_SESSIONS."""
        raise NotImplementedError("red:B3")

    def update(self, bar: HourlyBar) -> FeatureRow:
        """Consume the next bar and return its row, warm or not.

        Raises FeatureError, leaving the state exactly as it was, for:

        - an incomplete bar (`complete` False);
        - a bar whose t_open isn't strictly after the last bar's;
        - open, high, low or close not finite and > 0, low > min(open, close), or
          high < max(open, close);
        - a volume that is negative or not finite.

        Every other bar is consumed, even when some of its values are undefined (NaN in its
        row): it is real data, and later bars need its close and its place in the histories.
        """
        raise NotImplementedError("red:B3")

    def snapshot(self) -> FeatureState:
        """The state now. Two streams under one config that consumed the same bars give equal
        snapshots with identical canonical JSON."""
        raise NotImplementedError("red:B3")

    @classmethod
    def restore(cls, config: FeatureConfig, state: FeatureState) -> FeatureStream:
        """The stream `state` describes. It continues exactly as the stream that took the
        snapshot would have: the same rows, bit for bit, and the same later snapshots.

        Raises ConfigError when `state.config_hash` isn't `config.config_hash()`, or the state
        doesn't fit the config: a slot map whose keys aren't exactly the seven Slot values, a
        history longer than its bound (N for slot histories, B for `returns`), a non-empty
        history or a non-None EMA for a transform that is off, or some but not all of
        `last_t_open`, `last_session` and `prev_close` set.
        """
        raise NotImplementedError("red:B3")


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
        raise NotImplementedError("red:B3")

    def __len__(self) -> int:
        """The number of rows."""
        raise NotImplementedError("red:B3")

    def window(self, start: datetime, end: datetime) -> FeatureMatrix:
        """A new matrix of the rows whose bars became available in [start, end):
        start <= available_at < end. Naive datetimes raise ClockError."""
        raise NotImplementedError("red:B3")


def compute_features(bars: Sequence[HourlyBar], config: FeatureConfig) -> FeatureMatrix:
    """Run one new FeatureStream over `bars`, in the order given, and keep the warm rows.

    This is how a model's window is featurized (ARCHITECTURE §4.2): the decision path's streaming
    code over the bars a view returned. An incomplete bar is missing data and is skipped: it gets
    no row and leaves the stream untouched, exactly as if it weren't in the list. Any other bar the
    stream refuses raises its FeatureError. A warm row holding a NaN raises FeatureError naming
    its bar's t_open, because a model can't be fit on it. No bars, or none warm, gives a matrix
    with zero rows and the config's columns.
    """
    raise NotImplementedError("red:B3")


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
    raise NotImplementedError("red:B3")
