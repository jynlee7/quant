"""The broker port: what the kill executor, reconciliation and execution need from a broker.

ARCHITECTURE §3 lists the port under `execution/`, but K1 lets `risk` import only `core` and
`store`, and the kill executor (quant.risk.kill) lives in `risk`. So the port lives here, beside
the approved types its methods take. `execution` may import `risk`, so B13's Alpaca adapter
(`quant.execution.alpaca_broker`, which only `engine`, `control` and `cli` import, K4) and the
simulated broker implement it there. Tests use `tests.support.broker.FakeBroker`.

The port is a Protocol: an adapter implements it by shape, without inheriting from it.

**Changing anything at the broker takes an approval (I4, ARCHITECTURE §4.4).** `submit`,
`cancel`, `flatten` and `place_stop` are the only methods that change anything at the broker,
and each takes the matching approved type. An adapter is built with the RiskEngine that issued its approvals,
and calls `RiskEngine.spend(approval)` right before it sends; it sends nothing when `spend`
raises, and lets that ApprovalError propagate unchanged. A request the broker fails or refuses
after the spend raises BrokerError, and the approval stays spent: a retry needs a new one.

Every method raises BrokerError when the broker can't be reached or answers with an error. The
read methods change nothing.
"""

from __future__ import annotations

from typing import Protocol

from quant.core.orders import BrokerSnapshot, Quote
from quant.risk.engine import (
    RiskApprovedCancel,
    RiskApprovedFlatten,
    RiskApprovedOrder,
    RiskApprovedStop,
)


class BrokerPort(Protocol):
    """One broker account trading SPY (D1)."""

    def snapshot(self) -> BrokerSnapshot:
        """The account as the broker reports it now: the SPY position in shares, equity, and
        every working order, stops included, each as an OpenOrder whose `quantity` is what
        remains unfilled. `taken_at` is when the broker answered, UTC, from the adapter's Clock.
        Reconciliation and check 2 compare it with the engine's book."""
        ...

    def quote(self) -> Quote:
        """The freshest SPY quote: the IEX real-time quote live, the SIP minute bar in a
        backtest (C8). It feeds the collar, the kill exits and the minute monitor, never a
        feature. `at` is the quote's own time."""
        ...

    def submit(self, approval: RiskApprovedOrder, *, client_order_id: str) -> str:
        """Spend `approval`, then send its order: a day LIMIT order for `approval.quantity`
        shares of SPY on `approval.intent.side` at `approval.limit_price`, under
        `client_order_id` (ARCHITECTURE §8.2: instance id + decision id + leg + attempt, so a
        retry after a timeout can't double-submit). Returns the client order id. Execution
        (B13) calls it; the kill executor never does."""
        ...

    def cancel(self, approval: RiskApprovedCancel) -> None:
        """Spend `approval`, then cancel the working order `approval.client_order_id`, stops
        included. Returns once the broker has accepted the cancel request, which may still
        race a fill: the caller confirms it with a later `snapshot`. Cancelling an order the
        broker no longer holds (filled, cancelled or unknown) is not an error."""
        ...

    def flatten(self, approval: RiskApprovedFlatten, *, limit_price: float) -> str:
        """Spend `approval`, then send one LIMIT order that closes the whole SPY position the
        broker holds now at `limit_price`: a SELL of every share when long, a BUY of every share
        when short (a short is a reconciliation failure, but the kill switch still closes it).
        Returns the new order's client order id. The kill executor (quant.risk.kill) chooses the
        price, widening it after each unfilled `requote_s`; no collar checks it (spec §10).

        BrokerError, after the spend, when the broker holds no position, or refuses the order
        (Alpaca refuses a SELL while open sell orders, the stop included, reserve the shares,
        which is why the executor cancels every order first, ARCHITECTURE §8.2)."""
        ...

    def place_stop(self, approval: RiskApprovedStop) -> str:
        """Spend `approval`, then send one GTC stop-market SELL of `approval.quantity` shares of
        SPY triggered at `approval.stop_price` (spec §10's broker-side stop). Returns the new
        order's client order id. The engine approved it only for shares no other SELL covers,
        below the bid it was shown, so it can never take the position short. The kill executor
        leaves one when it can't finish (quant.risk.kill); execution (B13) places the protective
        stop after a fill with one.

        BrokerError, after the spend, when the broker refuses the order (Alpaca refuses a SELL
        while open sell orders already reserve the shares)."""
        ...
