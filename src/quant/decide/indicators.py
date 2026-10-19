"""Streaming indicators for the primitives: EMAs, ATR(n), and the N-bar mean, s.d. and z of the
close (spec §7, PLAN §5).

An `IndicatorStream` consumes one symbol's hourly bars in time order, one `update` per bar, like
`quant.features.stream.FeatureStream`, and returns that bar's `IndicatorValues`. Its state is
small and fixed-size (the last N closes, the last n true ranges, one value per EMA, the previous
close), snapshots to canonical JSON for EngineState (`IndicatorState`), and a restored stream
continues bit for bit. Each playbook gets the stream its params and invalidations need
(`IndicatorConfig.for_playbook`).

**Definitions**, for bar t with close C_t, high H_t and low L_t, "the previous bar" being the
last one consumed before it:

- **EMA(span)**: the same recursion as F5's (`quant.features.stream`, made public there as
  `ema` and imported here: ARCHITECTURE §4.2 keeps one implementation of every feature): EMA =
  C at the stream's first bar, then EMA_t = alpha·C_t + (1 - alpha)·EMA_{t-1}, alpha = 2 /
  (span + 1). An indicator stream and a feature stream over the same bars hold bit-identical
  EMAs of the same span.
- **TR_t** = max(H_t, C_{t-1}) - min(L_t, C_{t-1}), the true range. A session's opening bar takes
  C_{t-1} from the bar consumed before it, normally the previous session's CLOSE bar, so its true
  range includes the overnight gap (PLAN §5). The stream's first bar has TR = H - L.
- **ATR(n)**: `math.fsum` of the last n true ranges, t's own included, divided by n. Undefined
  (None) until n true ranges exist.
- **mean_N, sd_N**: over the last N closes, t's own included: mean = fsum(C) / N and the
  population s.d. (ddof = 0) sd = sqrt(fsum((C - mean)²) / N). Undefined until N closes exist.
- **z** = (C_t - mean_N) / sd_N. Undefined until N closes exist, and when every close in the
  window is equal (sd = 0: no signal).

Undefined values are None, never NaN, and computing them neither raises nor warns.

**Determinism.** Plain float64 arithmetic on the bars' values, no clock, randomness or I/O.
Window statistics are recomputed from the stored values on every update, never carried as
running sums, so the state holds only values taken from bars (and the EMAs) and a restored
stream continues bit for bit. The tests compare with a slow reference within 1e-12 relative.

**Missing and bad bars**, as the feature stream treats them: `update` refuses an incomplete bar
(`complete` False), a bar whose t_open isn't strictly after the last bar's, a price that isn't
finite and > 0, low above min(open, close) or high below max(open, close), with FeatureError
(a ModelError, so rule 6 applies) and the state exactly as it was. The stream then continues
from its previous bar, so the next bar's true range spans the gap. `indicators_for` skips
incomplete bars as `features_for` does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from pydantic import Field

from quant.core.calendar import BarSpec
from quant.core.model import StrictModel
from quant.data.frames import HourlyBar
from quant.data.view import PointInTimeView
from quant.decide.playbook import Playbook


class IndicatorConfig(StrictModel):
    """What one stream computes.

    - `ema_spans`: the EMA spans, ascending and distinct (each >= 1); may be empty;
    - `atr_bars`: n for ATR(n), or None for no ATR;
    - `window`: N for the mean, s.d. and z, or None for none (N >= 2 when set).

    The constructor of `IndicatorStream` refuses (ConfigError) spans that aren't ascending and
    distinct, a span, `atr_bars` or `window` below its minimum.
    """

    ema_spans: tuple[int, ...]
    atr_bars: int | None = Field(ge=1)
    window: int | None = Field(ge=2)

    @classmethod
    def for_playbook(cls, playbook: Playbook) -> IndicatorConfig:
        """The indicators `playbook` needs:

        - `trend_following`: EMAs `fast` and `slow`, `atr_bars`, no window;
        - `mean_reversion`: no trend EMAs, `atr_bars`, window `n`;
        - `flat`: nothing (no spans, no ATR, no window);

        plus the span of each `close_below_ema` invalidation, the union sorted ascending
        without duplicates. r1's CALM_UP gives spans (35, 175), atr_bars 35 and no window;
        r1's CHOP gives no spans, atr_bars 35 and window 35."""
        raise NotImplementedError

    def config_hash(self) -> str:
        """`quant.core.canonical.sha256_hex(self)`."""
        raise NotImplementedError


class IndicatorState(StrictModel):
    """An IndicatorStream's state as plain JSON data, for EngineState (ARCHITECTURE §4.3).
    `IndicatorState.model_validate(canonical_loads(canonical_dumps(state)))` reads it back.

    - `config_hash`: the `IndicatorConfig.config_hash()` the stream runs under;
    - `last_t_open`: epoch microseconds UTC of the last bar consumed, None before the first;
    - `prev_close`: the last bar's close, None before the first;
    - `ema`: per span, as a decimal string key ("35"), the EMA after the last bar; empty
      before the first bar;
    - `closes`: the last up-to-N closes, oldest first (empty without a window);
    - `true_ranges`: the last up-to-n true ranges, oldest first (empty without an ATR).

    Every float is finite.
    """

    config_hash: str
    last_t_open: int | None
    prev_close: float | None
    ema: dict[str, float]
    closes: list[float]
    true_ranges: list[float]


@dataclass(frozen=True)
class IndicatorValues:
    """One bar's indicators, as the stream computed them when it consumed the bar.

    `t_open` is the bar's. `ema` maps each configured span to its EMA (every span is present
    from the first bar). `atr`, `mean`, `sd` and `z` are None where undefined or not configured.
    """

    t_open: datetime
    ema: Mapping[int, float]
    atr: float | None
    mean: float | None
    sd: float | None
    z: float | None


class IndicatorStream:
    """Streaming indicators for one symbol's hourly bars, defined in the module docstring.
    Mutable: `update` advances it in place; `snapshot` and `restore` round-trip its state."""

    def __init__(self, config: IndicatorConfig) -> None:
        """A stream under `config` that has consumed no bar. ConfigError for a config the
        IndicatorConfig docstring refuses."""
        raise NotImplementedError

    @property
    def config(self) -> IndicatorConfig:
        raise NotImplementedError

    @property
    def last_t_open(self) -> datetime | None:
        """The t_open (UTC) of the last bar consumed, or None before the first."""
        raise NotImplementedError

    def update(self, bar: HourlyBar) -> IndicatorValues:
        """Consume the next bar and return its values. FeatureError, leaving the state exactly
        as it was, for the bars the module docstring refuses. Every value is a float64, so an
        int price can't change the state's canonical JSON."""
        raise NotImplementedError

    def snapshot(self) -> IndicatorState:
        """The state now, a deep copy. Two streams under one config that consumed the same bars
        give equal snapshots with identical canonical JSON."""
        raise NotImplementedError

    @classmethod
    def restore(cls, config: IndicatorConfig, state: IndicatorState) -> IndicatorStream:
        """The stream `state` describes; it continues exactly as the stream that took the
        snapshot would have. ConfigError when `state.config_hash` isn't `config.config_hash()`,
        or the state doesn't fit the config: `ema` keys that aren't exactly the configured spans
        (or not empty before the first bar), `closes` longer than N (or non-empty without a
        window), `true_ranges` longer than n (or non-empty without an ATR), a true range below
        0, a close or prev_close not > 0, or some but not all of `last_t_open` and `prev_close`
        set."""
        raise NotImplementedError


def indicators_for(
    stream: IndicatorStream, view: PointInTimeView, symbol: str, bar: BarSpec
) -> IndicatorValues:
    """The indicators for the decision on `bar`, read through the point-in-time view (I1),
    exactly as `quant.features.stream.features_for` reads features: `symbol`'s hourly bars with
    purpose OPERATE, all of them for a stream that has consumed none, else those opening after
    `stream.last_t_open`; each opening at or before `bar.t_open` is fed to `stream` in order,
    incomplete ones skipped. Returns `bar`'s values.

    FeatureError (rule 6 applies) when `bar` isn't among the bars fed: not visible in the view,
    incomplete, or already consumed; and when the stream refuses a bar (feeding stops there).
    Bars consumed before an error stay consumed.
    """
    raise NotImplementedError
