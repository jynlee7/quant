"""A fake broker, a kill store and an alert sink for the B9b tests (the kill executor, the minute
monitor and reconciliation).

FakeBroker implements quant.risk.broker.BrokerPort in memory. Like the real adapter it spends
every approval through the RiskEngine it was built with before it changes anything, so a test
sees exactly which approvals the code under test obtained and used. A kill exit fills at once,
in full, when its limit crosses the quote (a SELL at or below the bid, a BUY at or above the ask)
and is at or through `fill_at` (no further bound when `fill_at` is None); otherwise it stays
working until it is cancelled. `partial` makes the next exit fill only that many shares.
`positions` is the signed position after every change, starting with the first, so a test can
see a short however briefly it lasted. `place_stop` adds a working STOP SELL without asking
whether other SELLs already reserve the shares, so only the engine's own bound keeps a stop from
selling a share twice; `submit` refuses a SELL beyond the shares open SELLs leave free, as
Alpaca does (ARCHITECTURE §8.2). `trigger_stops` and `fill_sells` play a fall through every
stop and the fill of every working SELL. `fail` makes the named methods raise BrokerError, after the spend, a given number of
times. Every change is appended to `events`, a list a test may share with the store and the
alert sink to check the order of the steps.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from quant.core.clock import SimClock
from quant.core.errors import BrokerError
from quant.core.orders import BrokerSnapshot, OpenOrder, OrderKind, Quote, Side
from quant.risk.engine import (
    Approval,
    RiskApprovedCancel,
    RiskApprovedFlatten,
    RiskApprovedOrder,
    RiskApprovedStop,
    RiskEngine,
)
from tests.support.risk import EQUITY, PRICE

Event = tuple[Any, ...]


class FakeBroker:
    """An in-memory BrokerPort (module docstring)."""

    def __init__(
        self,
        engine: RiskEngine,
        clock: SimClock,
        *,
        position: int = 0,
        equity: float = EQUITY,
        orders: Iterable[OpenOrder] = (),
        bid: float = PRICE,
        ask: float = PRICE,
        fill_at: float | None = None,
        partial: int | None = None,
        fail: dict[str, int] | None = None,
        events: list[Event] | None = None,
    ) -> None:
        self.engine = engine
        self.clock = clock
        self.position = position
        self.equity = equity
        self.orders: dict[str, OpenOrder] = {o.client_order_id: o for o in orders}
        self.bid = bid
        self.ask = ask
        self.fill_at = fill_at
        self.partial = partial
        self.fail = dict(fail or {})
        self.events: list[Event] = [] if events is None else events
        self.spent: list[Approval] = []
        self.flatten_quantities: list[int] = []
        self.positions: list[int] = [position]
        self.halted_at_first_change: bool | None = None
        self.snapshots = 0
        self._exits = 0
        self._stops = 0

    # --- reads -------------------------------------------------------------------------------

    def snapshot(self) -> BrokerSnapshot:
        self._maybe_fail("snapshot")
        self.snapshots += 1
        return BrokerSnapshot(
            taken_at=self.clock.now(),
            position=self.position,
            equity=self.equity,
            open_orders=tuple(self.orders.values()),
        )

    def quote(self) -> Quote:
        self._maybe_fail("quote")
        return Quote(bid=self.bid, ask=self.ask, at=self.clock.now())

    # --- changes -----------------------------------------------------------------------------

    def submit(self, approval: RiskApprovedOrder, *, client_order_id: str) -> str:
        self._spend(approval)
        self._maybe_fail("submit")
        if approval.intent.side is Side.SELL:
            reserved = sum(o.quantity for o in self.orders.values() if o.side is Side.SELL)
            if reserved + approval.quantity > self.position:
                raise BrokerError("insufficient qty available")  # Alpaca's rule, ARCH §8.2
        self.orders[client_order_id] = OpenOrder(
            client_order_id=client_order_id,
            side=approval.intent.side,
            kind=OrderKind.LIMIT,
            quantity=approval.quantity,
            price=approval.limit_price,
        )
        self.events.append(("submit", client_order_id))
        return client_order_id

    def cancel(self, approval: RiskApprovedCancel) -> None:
        self._spend(approval)
        self._maybe_fail("cancel")
        self.orders.pop(approval.client_order_id, None)
        self.events.append(("cancel", approval.client_order_id))

    def flatten(self, approval: RiskApprovedFlatten, *, limit_price: float) -> str:
        self._spend(approval)
        self._maybe_fail("flatten")
        if self.position == 0:
            raise BrokerError("no position to close")
        side = Side.SELL if self.position > 0 else Side.BUY
        if any(o.side is side for o in self.orders.values()):
            raise BrokerError("open orders reserve the shares")  # Alpaca's rule, ARCH §8.2
        self._exits += 1
        client_order_id = f"kill-{self._exits}"
        quantity = abs(self.position)
        self.flatten_quantities.append(quantity)
        self.events.append(("flatten", client_order_id, limit_price))
        if side is Side.SELL:
            marketable = limit_price <= self.bid and (
                self.fill_at is None or limit_price <= self.fill_at
            )
        else:
            marketable = limit_price >= self.ask and (
                self.fill_at is None or limit_price >= self.fill_at
            )
        filled = 0
        if marketable:
            filled = quantity if self.partial is None else min(self.partial, quantity)
            self.partial = None
        self.move(-filled if side is Side.SELL else filled)
        if filled < quantity:
            self.orders[client_order_id] = OpenOrder(
                client_order_id=client_order_id,
                side=side,
                kind=OrderKind.LIMIT,
                quantity=quantity - filled,
                price=limit_price,
            )
        return client_order_id

    def place_stop(self, approval: RiskApprovedStop) -> str:
        self._spend(approval)
        self._maybe_fail("place_stop")
        self._stops += 1
        client_order_id = f"kill-stop-{self._stops}"
        self.orders[client_order_id] = OpenOrder(
            client_order_id=client_order_id,
            side=Side.SELL,
            kind=OrderKind.STOP,
            quantity=approval.quantity,
            price=approval.stop_price,
        )
        self.events.append(("place_stop", client_order_id, approval.quantity, approval.stop_price))
        return client_order_id

    def trigger_stops(self) -> None:
        """The price falls through every stop: each working STOP SELL fills in full."""
        for client_order_id, order in list(self.orders.items()):
            if order.kind is OrderKind.STOP and order.side is Side.SELL:
                del self.orders[client_order_id]
                self.move(-order.quantity)

    def fill_sells(self) -> None:
        """Every working SELL LIMIT fills in full."""
        for client_order_id, order in list(self.orders.items()):
            if order.kind is OrderKind.LIMIT and order.side is Side.SELL:
                del self.orders[client_order_id]
                self.move(-order.quantity)

    # --- helpers -----------------------------------------------------------------------------

    def move(self, shares: int) -> None:
        """Change the position by `shares` (a fill), recording the result in `positions`."""
        self.position += shares
        self.positions.append(self.position)

    def _spend(self, approval: Approval) -> None:
        if self.halted_at_first_change is None:
            self.halted_at_first_change = self.engine.state.halted
        self.engine.spend(approval)
        self.spent.append(approval)

    def _maybe_fail(self, method: str) -> None:
        if self.fail.get(method, 0) > 0:
            self.fail[method] -= 1
            self.events.append(("error", method))
            raise BrokerError(f"{method} failed (fake)")


@dataclass
class RecordingKillStore:
    """A KillStore that records each write in `events`. `fail_writes` makes every write raise
    OSError, as a full disk or a locked control.db would."""

    events: list[Event] = field(default_factory=list)
    fail_writes: bool = False

    def write_halted(self, reason: str, at: datetime) -> None:
        if self.fail_writes:
            raise OSError("control.db is locked (fake)")
        self.events.append(("write_halted", reason, at))

    def queue_exit(self, run_at: datetime, reason: str) -> None:
        if self.fail_writes:
            raise OSError("control.db is locked (fake)")
        self.events.append(("queue_exit", run_at, reason))


@dataclass
class RecordingAlerts:
    """KillAlerts that keeps every report. `fail` makes `send` raise after recording."""

    events: list[Event] = field(default_factory=list)
    reports: list[Any] = field(default_factory=list)
    fail: bool = False

    def send(self, report: Any) -> None:
        self.reports.append(report)
        self.events.append(("alert", report.status))
        if self.fail:
            raise RuntimeError("telegram is down (fake)")
