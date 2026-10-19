"""Intents: the bar's move as at most one OrderIntent (ARCHITECTURE §4.2 step 10, A3, D9, spec
§10, design call 11).

**The ceiling** (`ceiling`): the most the position may hold after this decision, the smallest of
`Caps.max_position`, the active label's hard cap (0.0 with no active label) and, on the
schedule's last fillable decision (15:46 ET, 12:46 on a half day), `Caps.overnight_cap` (A3:
0.5). `quant.decide.sizing.rebalance` clamps the target to it, and a reduction it forces is
exempt from the deadband.

**The order** (`order`): from the desired exposure (`sizing.Rebalance.desired`), the shares
held, the equity and the decision's last close (the price), with every product and quotient
exact on the decimal values (`Fraction(repr(x))`), then floored:

- target shares = floor(desired x equity / price): whole shares, never above the target, and 0
  for an exit, so an exit to 0 sells every share held;
- the move is target shares - shares held; at most floor(single_order_cap x equity / price)
  shares move in one order, BUY and SELL alike (spec §10's 0.5x, PLAN's property "never over
  0.5x in one order"): a full entry, or a full exit, takes two bars;
- a move of 0 shares is no intent (None). At most one intent per decision, `leg` 1.

The intent is `quant.core.orders.OrderIntent` exactly as B9's RiskEngine reads it:
`decision_id(tau)` and `tau` from the Decision; `side` BUY to add, SELL to reduce (never more
than the shares held: never short, D9); `quantity`; `target_exposure` the desired exposure (the
position this decision aims for; a scale-in reaches it over two bars); `label` the active label;
`uncertain` rule 5's verdict on this bar; `reduce_only` True on a SELL that is forced or on the
closing-bar decision (`Decision.reduce_only`: 16:16 ET, 13:16 on a half day, executing at the
next open), never on a BUY; `stop_price` on a BUY, the trade's ATR stop (spec §7: fixed at entry,
so a scale-in carries the same stop), which must lie strictly between 0 and the price; None on a
SELL.

Pure: no clock, no I/O.
"""

from __future__ import annotations

from quant.core.orders import OrderIntent
from quant.core.regime import Label
from quant.core.schedule import Decision
from quant.decide.sizing import Caps


def ceiling(decision: Decision, caps: Caps, active: Label | None) -> float:
    """The ceiling of the module docstring: min(caps.max_position, caps.cap(active)), and
    caps.overnight_cap too when `decision.last_fillable`."""
    raise NotImplementedError


def target_shares(desired: float, *, equity: float, price: float) -> int:
    """floor(desired x equity / price), exact on the decimal values. ValueError, its message
    naming the bad argument, unless
    `desired` is finite and in [0, 1] and `equity` and `price` are finite and > 0."""
    raise NotImplementedError


def order(
    decision: Decision,
    *,
    desired: float,
    shares: int,
    equity: float,
    price: float,
    caps: Caps,
    label: Label | None,
    uncertain: bool,
    forced: bool,
    stop: float | None,
) -> OrderIntent | None:
    """The decision's one intent (module docstring), or None when no share moves. `shares` is
    the position held (>= 0). ValueError, its message naming `shares`, `reduce_only` or `stop`,
    for `shares` < 0, a BUY on a closing-bar decision (`decision.reduce_only`: decide may only
    reduce there), or a BUY whose `stop` isn't strictly between 0 and `price`; and as
    `target_shares` raises."""
    raise NotImplementedError
