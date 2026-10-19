"""The edge behind Kelly sizing: shadow returns and their accumulators (spec §9, C12).

**Shadow returns.** Each trading playbook (trend_following, mean_reversion) runs its primitive on
every decision bar as a *shadow*, whether or not the engine took a position: the same
`quant.decide.primitives.step` the live trade runs, with its own stop, take-profit and re-arm,
always allowed to enter (`entries_allowed=True`), and without invalidation (`label_prob=None`).
Its position is 0 or 1 (1.0x equity). A flat playbook has no shadow.

For bar t, with C_{t-1} the close of the last bar the shadow consumed before t and pos_{t-1} its
position before bar t (held across bar t), the shadow return is

    r_t = pos_{t-1} * (X_t / C_{t-1} - 1) - changes_t * shadow_cost_bp * 1e-4

- X_t, the exit price: `min(open_t, stop)` when the shadow was held and bar t's low <= its stop
  (a stop hit inside the bar: a gap through the stop fills at the open), else the close C_t;
- changes_t, the changes of shadow position charged to bar t: 1 for the entry when the
  position held across bar t was entered on the bar before (its first bar held), plus 1 when the
  shadow is out after bar t (an exit: signal, stop or take-profit). Both are charged to bars
  the shadow held, so a round trip pays twice on bars that count (design call 10; B11 adds the
  fees from costs.yaml). A bar not held earns 0 and is charged nothing.
- The shadow's first bar has no C_{t-1}: no return.

**Counting** (C12): bar t's return is added to the playbook's accumulator only when the label
active *before* bar t's switching step (the label the position was held under across bar t)
maps to that playbook, and pos_{t-1} = 1. Labels map one to one to playbooks (the release's
map), so the accumulators are keyed by label.

**Accumulators.** Per label, n, the mean and M2 by Welford's update: n += 1; d = x - mean;
mean += d / n; M2 += d * (x - mean). The sample variance is M2 / (n - 1) (ddof = 1).

**f*** = max(0, mean - s / sqrt(n)) / s^2, with s the sample standard deviation: mean minus one
standard error, over the variance (spec §9). f* = 0 when n < `min_edge_bars` (100) or s = 0.
Sizing uses `kelly_fraction` (0.25) x f*.

**Seeding** (spec §9): the first model's accumulators come from the fit-window backtest (B11),
so an `EdgeBook` can be built from a sequence of past returns per label (`EdgeBook.seeded`),
added in order exactly as `add` would. B8 never computes P&L on DEV: tests seed synthetic
returns.

Everything here is pure float64 arithmetic; the states are StrictModels, plain JSON data for
EngineState (ARCHITECTURE §4.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import Field

from quant.core.model import StrictModel
from quant.core.regime import Label
from quant.data.frames import HourlyBar
from quant.decide.indicators import IndicatorValues
from quant.decide.playbook import Playbook
from quant.decide.primitives import Reason, TradeState


class EdgeStats(StrictModel):
    """One label's accumulator: `n` counted returns, their `mean` and `m2` (Welford's sum of
    squared deviations). `EdgeStats()` is never built directly by decide: use `empty()`."""

    n: int = Field(ge=0)
    mean: float
    m2: float = Field(ge=0.0)

    @classmethod
    def empty(cls) -> EdgeStats:
        """n 0, mean 0.0, m2 0.0."""
        raise NotImplementedError

    def add(self, x: float) -> EdgeStats:
        """The accumulator after one more return `x`, by Welford's update (module docstring).
        Pure: `self` is unchanged. ValueError, its message saying "finite", for a non-finite
        `x`."""
        raise NotImplementedError

    @property
    def variance(self) -> float | None:
        """The sample variance m2 / (n - 1); None when n < 2."""
        raise NotImplementedError

    @property
    def sd(self) -> float | None:
        """The sample standard deviation, sqrt(variance); None when n < 2."""
        raise NotImplementedError

    def f_star(self, *, min_bars: int) -> float:
        """`kelly(n, mean, sd, min_bars=min_bars)`: 0.0 when n < min_bars or n < 2."""
        raise NotImplementedError


def kelly(n: int, mean: float, sd: float | None, *, min_bars: int) -> float:
    """f* = max(0, mean - sd / sqrt(n)) / sd^2, the full Kelly fraction on the lower confidence
    bound of the mean (spec §9). 0.0 when n < min_bars, when sd is None or 0, and when the bound
    is not positive. Never negative, never NaN."""
    raise NotImplementedError


class EdgeBook(StrictModel):
    """Every label's accumulator (`stats`, exactly one per Label, flat labels included: they
    never count a return and stay empty)."""

    stats: dict[Label, EdgeStats]

    @classmethod
    def empty(cls) -> EdgeBook:
        """Every label empty."""
        raise NotImplementedError

    @classmethod
    def seeded(cls, returns: Mapping[Label, Sequence[float]]) -> EdgeBook:
        """The book after adding each label's `returns`, in order, to an empty accumulator
        (labels not named stay empty): equal, field for field, to `empty()` followed by `add`
        for each return. ValueError, its message saying "finite", for a non-finite return."""
        raise NotImplementedError

    def add(self, label: Label, x: float) -> EdgeBook:
        """The book with `x` added to `label`'s accumulator. Pure."""
        raise NotImplementedError

    def f_star(self, label: Label, *, min_bars: int) -> float:
        """`self.stats[label].f_star(min_bars=min_bars)`."""
        raise NotImplementedError


class ShadowState(StrictModel):
    """One playbook's shadow (module docstring):

    - `trade`: the shadow's trade lifecycle (`quant.decide.primitives.TradeState`), flat and
      armed when cold;
    - `prev_close`: the close of the last bar the shadow consumed, None before its first;
    - `entered`: the shadow entered on that last bar, so its entry cost is still to be charged
      (to the next bar, its first held).
    """

    trade: TradeState
    prev_close: float | None
    entered: bool

    @classmethod
    def cold(cls) -> ShadowState:
        """`TradeState.flat()`, no previous close, not entered."""
        raise NotImplementedError


@dataclass(frozen=True)
class ShadowStep:
    """One bar of a shadow.

    - `state`: the shadow after the bar;
    - `held`: pos_{t-1}, the shadow was in a position across the bar;
    - `ret`: the bar's shadow return r_t, None on the shadow's first bar (no previous close);
    - `reason`: the primitive's reason on the bar.
    """

    state: ShadowState
    held: bool
    ret: float | None
    reason: Reason


def shadow_return(
    prev_close: float,
    bar: HourlyBar,
    *,
    held: bool,
    stop: float | None,
    changes: int,
    cost_bp: float,
) -> float:
    """r_t of the module docstring: `(X / prev_close - 1 if held else 0.0) - changes * cost_bp *
    1e-4`, X = `min(bar.open, stop)` when held, `stop` is not None and `bar.low <= stop`, else
    `bar.close`. ValueError, its message naming `changes` or `prev_close`, when `changes` is
    negative or `prev_close` isn't finite and > 0."""
    raise NotImplementedError


def shadow_step(
    playbook: Playbook,
    state: ShadowState,
    values: IndicatorValues,
    bar: HourlyBar,
    *,
    cost_bp: float,
) -> ShadowStep:
    """One bar of `playbook`'s shadow (module docstring): `held` is `state.trade.in_position`;
    the trade advances by `quant.decide.primitives.step(playbook, state.trade, values, bar,
    entries_allowed=True, label_prob=None)`; `ret` is `shadow_return(state.prev_close, bar,
    held=held, stop=state.trade.stop, changes=..., cost_bp=cost_bp)` with changes = (1 if held
    and state.entered) + (1 if held and the step left it flat), or None when `state.prev_close`
    is None. The new state has the stepped trade, `prev_close` = bar.close and `entered` = (not
    held and in position after the step). Pure. ValueError (from `step`) when `values` isn't
    `bar`'s; a flat playbook's shadow stays `ShadowState.cold()` apart from `prev_close`, held
    never, its returns 0.0."""
    raise NotImplementedError
