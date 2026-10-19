"""The strategy primitives, their trade lifecycle (ATR stop, take-profit, re-arm) and invalidation
(spec §7, PLAN §5, D9).

**Primitives** are long or flat only (D9). Each reads one bar's indicators and close and gives a
`Signal`, its entry and exit conditions:

- `trend_following`: entry when EMA(fast) > EMA(slow) and close > EMA(fast); exit when
  EMA(fast) < EMA(slow). Both EMAs are defined from the stream's first bar.
- `mean_reversion`: entry when z <= -entry_z; exit when z >= -exit_z. Undefined (no entry, no
  exit) while z is None.
- `flat`: never an entry; never in a position.

The comparisons are plain float comparisons, strict or not exactly as written. Because exit_z <
entry_z and fast/slow are compared strictly both ways, a bar never has both an entry and an exit.

**The lifecycle** (`step`), a pure function of (the playbook, its `TradeState`, the bar's
`IndicatorValues`, the bar) -> `Step(state, reason)`:

In a trade (the state's `in_position`), in this order:

1. `bars_in_trade` += 1 and `peak_close` = max(peak_close, close).
2. **Invalidation**, when label_prob is given: check every condition on the updated trade;
   a hit closes it with reason INVALIDATED and Step.invalidation naming the hit.
3. **Stop**: the bar's low <= `stop` -> out, reason STOP. The stop is fixed at entry,
   `entry_price - stop_loss x ATR(at entry)`, and never trails.
4. **Take-profit** (when the playbook's `take_profit` isn't null): the close >= `take_profit`,
   fixed at entry as `entry_price + take_profit x ATR(at entry)` -> out, reason TAKE_PROFIT.
5. **Exit signal** -> out, reason EXIT.
6. Otherwise HOLD.

Flat (not in a trade):

1. Not `armed` (after a stop or take-profit): if the bar's exit condition holds, the state
   re-arms (`armed` True; reason REARM), else reason DISARMED. No entry on such a bar (design
   call 7: the primitive must see its own exit condition before it can enter again, so a
   stop-out doesn't re-enter on the next bar while the entry condition still holds).
2. Armed, entry condition holds, and ATR is defined and > 0: if `entries_allowed` is False,
   reason BLOCKED and no entry; else ENTER: `entry_price` = the close, `stop` and `take_profit`
   as above from this bar's ATR, `bars_in_trade` 0, `peak_close` the close.
3. Armed, entry condition holds, but ATR is None or 0: reason UNDEFINED, no entry.
4. Otherwise WAIT (UNDEFINED when the primitive's signal itself is undefined).

A STOP or TAKE_PROFIT leaves the state flat with `armed` False, unless the same bar's exit
condition holds, which re-arms it on that bar (the reason stays STOP or TAKE_PROFIT). An EXIT
leaves it flat and armed. No bar both exits and enters. `flat` playbooks always return
`TradeState.flat()` with reason FLAT.

**Invalidation** (`check_invalidation`, spec §7): evaluated inside `step` on every trade bar,
after updating its age and peak and before any exit, when label_prob is given. Conditions run
in their listed order; the first that holds is the hit:

- `label_prob_below`: `Decimal(repr(label_prob)) < Decimal(repr(threshold))` (exactly the
  threshold doesn't fire);
- `close_below_ema`: close < EMA(span) (the stream keeps that EMA: `IndicatorConfig.for_playbook`);
- `bars_in_trade_exceeds`: bars_in_trade > bars;
- `trade_drawdown_exceeds`: (peak_close - close) / peak_close > fraction.

A hit exits the position; decide sets the target to 0 and blocks entries
under that playbook until a fresh takeover (quant.decide.switching.invalidate, design call 6).
Flat entries are refused while label_prob_below or close_below_ema holds, without a block.
The shadow edge book (B8b) runs `step` with label_prob=None, without invalidation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from pydantic import Field

from quant.core.model import StrictModel
from quant.data.frames import HourlyBar
from quant.decide.indicators import IndicatorValues
from quant.decide.playbook import (
    Invalidation,
    MeanReversionParams,
    Playbook,
    TrendParams,
)


class Reason(StrEnum):
    """Why a step ends where it does (the rule trace records it)."""

    ENTER = "enter"
    HOLD = "hold"
    EXIT = "exit"
    INVALIDATED = "invalidated"
    STOP = "stop"
    TAKE_PROFIT = "take_profit"
    REARM = "rearm"
    DISARMED = "disarmed"
    BLOCKED = "blocked"
    UNDEFINED = "undefined"
    WAIT = "wait"
    FLAT = "flat"


@dataclass(frozen=True)
class Signal:
    """A primitive's reading of one bar: `defined` False when its indicators are undefined (then
    `entry` and `exit` are both False); `entry` and `exit` never both True."""

    entry: bool
    exit: bool
    defined: bool


class TradeState(StrictModel):
    """One playbook's trade, as plain JSON data for EngineState (ARCHITECTURE §4.3).

    - `in_position`: in a trade (long, at the playbook's size) or flat;
    - `armed`: may enter (False after a stop or take-profit until the exit condition is seen);
    - `entry_price`, `stop`, `take_profit`: fixed at entry; None when flat (take_profit also
      None when the playbook has none);
    - `bars_in_trade`: decision bars since the entry bar (0 on it and when flat);
    - `peak_close`: the highest close since entry, the entry close included; None when flat.

    `TradeState.flat()` is the cold state: flat and armed.
    """

    in_position: bool
    armed: bool
    entry_price: float | None
    stop: float | None
    take_profit: float | None
    bars_in_trade: int = Field(ge=0)
    peak_close: float | None

    @classmethod
    def flat(cls) -> TradeState:
        """Flat, armed, every trade field None and `bars_in_trade` 0 (C4: a run starts flat)."""
        return cls(
            in_position=False,
            armed=True,
            entry_price=None,
            stop=None,
            take_profit=None,
            bars_in_trade=0,
            peak_close=None,
        )


@dataclass(frozen=True)
class Step:
    """The state after one bar and why. `in_position` is the state's."""

    state: TradeState
    reason: Reason
    invalidation: InvalidationHit | None = None

    @property
    def in_position(self) -> bool:
        return self.state.in_position


@dataclass(frozen=True)
class InvalidationHit:
    """The first invalidation condition that held, and a reason naming its value."""

    condition: Invalidation
    reason: str


def trend_following(params: TrendParams, values: IndicatorValues, close: float) -> Signal:
    """The trend signal of the module docstring from `values.ema[fast]` and `values.ema[slow]`.
    KeyError when `values` lacks either span (a stream configured for another playbook)."""
    fast, slow = values.ema[params.fast], values.ema[params.slow]
    return Signal(entry=fast > slow and close > fast, exit=fast < slow, defined=True)


def mean_reversion(params: MeanReversionParams, values: IndicatorValues) -> Signal:
    """The mean-reversion signal of the module docstring from `values.z`; undefined while z is
    None."""
    if values.z is None:
        return Signal(False, False, False)
    return Signal(entry=values.z <= -params.entry_z, exit=values.z >= -params.exit_z, defined=True)


def flat() -> Signal:
    """No entry, no exit, defined: the flat primitive holds nothing."""
    return Signal(False, False, True)


def signal(playbook: Playbook, values: IndicatorValues, close: float) -> Signal:
    """The signal of `playbook`'s primitive (dispatch on its strategy)."""
    if playbook.strategy == "trend_following":
        return trend_following(playbook.params, values, close)
    if playbook.strategy == "mean_reversion":
        return mean_reversion(playbook.params, values)
    return flat()


def stop_level(entry_price: float, atr: float, multiple: float) -> float:
    """`entry_price - multiple * atr`."""
    return entry_price - multiple * atr


def take_profit_level(entry_price: float, atr: float, multiple: float) -> float:
    """`entry_price + multiple * atr`."""
    return entry_price + multiple * atr


def step(
    playbook: Playbook,
    state: TradeState,
    values: IndicatorValues,
    bar: HourlyBar,
    *,
    entries_allowed: bool = True,
    label_prob: float | None = None,
) -> Step:
    """One bar of the lifecycle in the module docstring. Pure: `state` is not changed.
    `entries_allowed` False (rule 6, an invalidation block, no active label, a frozen or locked
    account) suppresses an ENTER as BLOCKED; it never forces an exit (decide does that with
    `close_trade`). ValueError, its message naming `t_open`, when `values.t_open` isn't
    `bar.t_open`."""
    if values.t_open != bar.t_open:
        raise ValueError("indicator t_open must match bar t_open")
    if playbook.strategy == "flat":
        return Step(TradeState.flat(), Reason.FLAT)
    reading = signal(playbook, values, bar.close)
    if state.in_position:
        bars = state.bars_in_trade + 1
        peak = max(state.peak_close or bar.close, bar.close)
        state = state.model_copy(update={"bars_in_trade": bars, "peak_close": peak})
        if label_prob is not None:
            hit = check_invalidation(playbook, state, values, bar, label_prob=label_prob)
            if hit is not None:
                return Step(close_trade(state), Reason.INVALIDATED, hit)
        if state.stop is None:
            raise ValueError("an open trade must have a stop")
        if bar.low <= state.stop:
            return Step(close_trade(state).model_copy(update={"armed": reading.exit}), Reason.STOP)
        if state.take_profit is not None and bar.close >= state.take_profit:
            return Step(
                close_trade(state).model_copy(update={"armed": reading.exit}), Reason.TAKE_PROFIT
            )
        if reading.exit:
            return Step(close_trade(state), Reason.EXIT)
        return Step(state, Reason.HOLD)
    if not state.armed:
        if reading.exit:
            return Step(state.model_copy(update={"armed": True}), Reason.REARM)
        return Step(state, Reason.DISARMED)
    if not reading.defined:
        return Step(state, Reason.UNDEFINED)
    if reading.entry:
        if values.atr is None or values.atr <= 0:
            return Step(state, Reason.UNDEFINED)
        if not entries_allowed:
            return Step(state, Reason.BLOCKED)
        if label_prob is not None:
            for condition in playbook.invalidation:
                if (condition.kind == "label_prob_below" and label_prob < condition.threshold) or (
                    condition.kind == "close_below_ema" and bar.close < values.ema[condition.span]
                ):
                    return Step(state, Reason.BLOCKED)
        tp = (
            None
            if playbook.take_profit is None
            else take_profit_level(bar.close, values.atr, playbook.take_profit)
        )
        return Step(
            TradeState(
                in_position=True,
                armed=True,
                entry_price=bar.close,
                stop=stop_level(bar.close, values.atr, playbook.stop_loss),
                take_profit=tp,
                bars_in_trade=0,
                peak_close=bar.close,
            ),
            Reason.ENTER,
        )
    return Step(state, Reason.WAIT)


def close_trade(state: TradeState) -> TradeState:
    """The flat, armed state after a forced exit (an invalidation, rule 6, a switch away):
    every trade field cleared. Re-arming waits only after a stop or take-profit (design call
    7); an invalidated playbook is kept out by its block instead (design call 6)."""
    return state.model_copy(
        update={
            "in_position": False,
            "armed": True,
            "entry_price": None,
            "stop": None,
            "take_profit": None,
            "bars_in_trade": 0,
            "peak_close": None,
        }
    )


def check_invalidation(
    playbook: Playbook,
    state: TradeState,
    values: IndicatorValues,
    bar: HourlyBar,
    *,
    label_prob: float,
) -> InvalidationHit | None:
    """The first of `playbook.invalidation`'s conditions that holds on this bar (module
    docstring), or None. Always None when `state` isn't in a trade. `state` has this bar's
    updated age and peak, before exits; `label_prob` is the filtered P(playbook.label) on this
    bar (0 for a label the model lacks). KeyError when a `close_below_ema` span is missing from
    `values.ema`."""
    if not state.in_position:
        return None
    for condition in playbook.invalidation:
        if condition.kind == "label_prob_below" and Decimal(repr(label_prob)) < Decimal(
            repr(condition.threshold)
        ):
            return InvalidationHit(
                condition, f"label probability {label_prob} < {condition.threshold}"
            )
        if condition.kind == "close_below_ema" and bar.close < values.ema[condition.span]:
            return InvalidationHit(
                condition, f"close {bar.close} < EMA({condition.span}) {values.ema[condition.span]}"
            )
        if condition.kind == "bars_in_trade_exceeds" and state.bars_in_trade > condition.bars:
            return InvalidationHit(
                condition, f"bars in trade {state.bars_in_trade} > {condition.bars}"
            )
        if condition.kind == "trade_drawdown_exceeds":
            peak = state.peak_close
            if peak is None or peak <= 0:
                raise ValueError("an open trade must have a positive peak_close")
            if (peak - bar.close) / peak > condition.fraction:
                return InvalidationHit(
                    condition, f"trade drawdown {(peak - bar.close) / peak} > {condition.fraction}"
                )
    return None
