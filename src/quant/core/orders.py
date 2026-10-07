"""Order intents, and the broker and account views the risk engine reads (ARCHITECTURE §4, §8.1).

`decide()` (B8) emits OrderIntents. The risk engine (B9) checks each one against a RiskContext
built from these views and returns a verdict (quant.risk.engine). They live in core because K1
lets `risk` import only `core` and `store`, and `decide` must never import `risk`.

- Quantities are whole shares. A broker-side GTC stop (spec §10) can't cover a fractional
  position, so the system never holds one.
- Prices and equity are dollars, float64, finite and positive (StrictModel refuses NaN).
- Times are timezone-aware UTC. A naive datetime is refused with a pydantic ValidationError.
- Every type is a StrictModel: unknown keys refused, no coercion, immutable once built.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import Field, field_validator, model_validator

from quant.core.clock import ensure_utc
from quant.core.errors import ClockError
from quant.core.ids import decision_id
from quant.core.model import StrictModel

SYMBOL: Final = "SPY"
"""The one instrument (D1)."""


class Label(StrEnum):
    """The regime labels (spec §4, D8). Each has a hard cap in config/risk.yaml."""

    CALM_UP = "CALM_UP"
    CHOP = "CHOP"
    STRESS = "STRESS"
    CRASH = "CRASH"


class Side(StrEnum):
    """Long or flat only (D9): a BUY adds exposure, a SELL removes it and never opens a short."""

    BUY = "buy"
    SELL = "sell"


class OrderKind(StrEnum):
    LIMIT = "limit"  # entries and discretionary exits: marketable limits inside the collar
    STOP = "stop"  # the broker-side protective stop-market order (spec §10)


class TradingMode(StrEnum):
    """Which account the engine trades. The backtest and replays run as PAPER, so D7 flags its
    large orders there exactly as paper trading does."""

    PAPER = "paper"
    LIVE = "live"


class OrderIntent(StrictModel):
    """One rebalance that `decide()` asks for (ARCHITECTURE §4.2, step 10).

    C3 counts an intent once, however many re-price attempts it takes; each attempt is checked
    again and gets its own approval.

    - `decision_id`: `quant.core.ids.decision_id(tau)`.
    - `tau`: the decision time, UTC, one of `quant.core.schedule.decision_schedule`'s for its
      session. The risk engine looks the decision up itself, so it never trusts the intent to say
      whether it is the last fillable one or a closing-bar one.
    - `leg`: 1 for a decision's one intent. `(decision_id, leg)` is the intent's identity: C3
      counts distinct ones, and execution's `client_order_id` names both (ARCHITECTURE §8.2).
    - `symbol`: SYMBOL.
    - `side`, `quantity`: the order, in whole shares (> 0), as decide sized it. The risk engine
      may clamp a SELL to the shares held, and never raises a quantity.
    - `target_exposure`: the position decide aims for after the intent, as a fraction of equity
      in [0, 1] (spec §9). It goes into the record; the checks use the order itself.
    - `label`: the active label, or None when no playbook is active (cold start, C4). Its cap in
      risk.yaml bounds a BUY; None caps it at 0.
    - `uncertain`: rule 5 fired on this bar (spec §8), which caps a BUY at 0.
    - `reduce_only`: decide may only reduce on this decision: the closing-bar decision
      (`Decision.reduce_only`) or a reduction forced by rules 4–6 or a risk limit.
    - `stop_price`: the protective stop level (spec §7's ATR stop). Every BUY must carry one
      below its limit price (check 6). A SELL needs none, and the checks ignore it there.
    """

    decision_id: str
    tau: datetime
    leg: int = Field(default=1, ge=1)
    symbol: str = SYMBOL
    side: Side
    quantity: int = Field(gt=0)
    target_exposure: float = Field(ge=0.0, le=1.0)
    label: Label | None
    uncertain: bool = False
    reduce_only: bool = False
    stop_price: float | None = Field(default=None, gt=0.0)

    @field_validator("tau")
    @classmethod
    def _tau_utc(cls, value: datetime) -> datetime:
        """`value` in UTC (quant.core.clock.ensure_utc); a naive datetime raises ValueError, so
        pydantic reports a ValidationError."""
        return _utc(value)

    @model_validator(mode="after")
    def _consistent(self) -> OrderIntent:
        """ValueError (a ValidationError) unless `decision_id == decision_id(tau)`, `symbol ==
        SYMBOL`, and a BUY is not `reduce_only`. A tau off the whole minute, for which
        `decision_id` raises ClockError, is a ValueError too."""
        try:
            expected = decision_id(self.tau)
        except ClockError as exc:
            raise ValueError(str(exc)) from exc
        if self.decision_id != expected:
            raise ValueError("decision_id must name tau")
        if self.symbol != SYMBOL:
            raise ValueError(f"only {SYMBOL} may be traded")
        if self.side is Side.BUY and self.reduce_only:
            raise ValueError("a BUY cannot be reduce-only")
        return self


class Quote(StrictModel):
    """The freshest quote: Alpaca IEX live, the SIP minute bar in the backtest (C8). It feeds the
    collar and the minute monitor only, never a feature. `at` is the quote's own time, UTC.
    A crossed quote (bid > ask) is a valid Quote that check 6 vetoes."""

    bid: float = Field(gt=0.0)
    ask: float = Field(gt=0.0)
    at: datetime

    @field_validator("at")
    @classmethod
    def _at_utc(cls, value: datetime) -> datetime:
        """`value` in UTC; a naive datetime raises ValueError (a ValidationError)."""
        return _utc(value)


class OpenOrder(StrictModel):
    """One working order, as the broker or the engine's book holds it.

    `quantity` is what remains unfilled, in shares. `price` is the limit price of a LIMIT and
    the trigger of a STOP. Reconciliation compares open orders field by field."""

    client_order_id: str
    side: Side
    kind: OrderKind
    quantity: int = Field(gt=0)
    price: float = Field(gt=0.0)


class BrokerSnapshot(StrictModel):
    """What the broker reported (spec §10, ARCHITECTURE §8.1 check 2).

    - `taken_at`: when the broker answered, UTC. Check 2 needs it no older than 5 s.
    - `position`: SPY shares held. Negative would be a short, which a long-or-flat system must
      never hold, so the type allows it and the risk engine halts on it.
    - `equity`: account equity in dollars; the base of every exposure fraction.
    - `open_orders`: every working order, stops included.
    """

    taken_at: datetime
    position: int
    equity: float = Field(gt=0.0)
    open_orders: tuple[OpenOrder, ...] = ()

    @field_validator("taken_at")
    @classmethod
    def _taken_utc(cls, value: datetime) -> datetime:
        """`value` in UTC; a naive datetime raises ValueError (a ValidationError)."""
        return _utc(value)


class InternalBook(StrictModel):
    """What the engine believes it holds: its position in shares and its working orders. Check 2
    compares it with a BrokerSnapshot and never corrects either."""

    position: int
    open_orders: tuple[OpenOrder, ...] = ()


def _utc(value: datetime) -> datetime:
    try:
        return ensure_utc(value)
    except ClockError as exc:
        raise ValueError(str(exc)) from exc
