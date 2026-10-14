"""Risk engines, intents and contexts for the B9 tests.

The desk trades a $100,000 account at $100 a share, so 1,000 shares are exactly 1.0x equity and
every cap lands on a whole number of shares. Times come from the real NYSE calendar: DAY is a
full session in FIT (2018-03-06, before daylight saving time), HALF_DAY the day after
Thanksgiving 2018, which closes at 13:00 ET.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from quant.core.calendar import TradingCalendar
from quant.core.clock import SimClock
from quant.core.ids import decision_id
from quant.core.orders import (
    BrokerSnapshot,
    InternalBook,
    Label,
    OpenOrder,
    OrderIntent,
    OrderKind,
    Quote,
    Side,
    TradingMode,
)
from quant.core.schedule import Decision, decision_schedule
from quant.risk.engine import (
    RiskApprovedStop,
    RiskContext,
    RiskEngine,
    RiskState,
    RiskVerdict,
    SpentApprovals,
)
from quant.risk.limits import RiskConfig, parse_risk_config

REPO = Path(__file__).resolve().parents[2]
RISK_YAML = REPO / "config" / "risk.yaml"
CAL = TradingCalendar.xnys()
DAY = date(2018, 3, 6)
NEXT_DAY = date(2018, 3, 7)
HALF_DAY = date(2018, 11, 23)
EQUITY = 100_000.0
PRICE = 100.0
STOP = 95.0
SHARES_PER_EQUITY = int(EQUITY / PRICE)  # 1,000 shares are 1.0x


def config() -> RiskConfig:
    """The committed config/risk.yaml."""
    return parse_risk_config(RISK_YAML.read_bytes())


def decisions(day: date = DAY) -> tuple[Decision, ...]:
    return decision_schedule(CAL, day)


def last_fillable(day: date = DAY) -> Decision:
    (decision,) = (d for d in decisions(day) if d.last_fillable)
    return decision


def closing(day: date = DAY) -> Decision:
    """The closing-bar decision, which can only reduce and executes at the next open."""
    return decisions(day)[-1]


def intent(
    side: Side = Side.BUY,
    quantity: int = 100,
    *,
    tau: datetime | None = None,
    leg: int = 1,
    label: Label | None = Label.CALM_UP,
    uncertain: bool = False,
    reduce_only: bool = False,
    stop: float | None = STOP,
    target: float = 0.5,
) -> OrderIntent:
    """An intent at `tau` (default: DAY's first decision). A SELL carries no stop."""
    tau = decisions()[0].tau if tau is None else tau
    return OrderIntent(
        decision_id=decision_id(tau),
        tau=tau,
        leg=leg,
        side=side,
        quantity=quantity,
        target_exposure=target,
        label=label,
        uncertain=uncertain,
        reduce_only=reduce_only,
        stop_price=stop if side is Side.BUY else None,
    )


def order(
    client_order_id: str,
    side: Side = Side.SELL,
    kind: OrderKind = OrderKind.STOP,
    quantity: int = 100,
    price: float = STOP,
) -> OpenOrder:
    return OpenOrder(
        client_order_id=client_order_id, side=side, kind=kind, quantity=quantity, price=price
    )


def context(
    now: datetime,
    *,
    position: int = 0,
    equity: float = EQUITY,
    open_orders: tuple[OpenOrder, ...] = (),
    book_position: int | None = None,
    book_orders: tuple[OpenOrder, ...] | None = None,
    snapshot_age: timedelta = timedelta(0),
    quote_age: timedelta = timedelta(0),
    bid: float = PRICE,
    ask: float = PRICE,
    limit: float = PRICE,
    healthy: bool = True,
    drift_frozen: bool = False,
    halted: bool = False,
) -> RiskContext:
    """A context at `now` whose broker and book agree unless told otherwise. The quote is
    locked at PRICE and the limit is PRICE, so exposures are exact."""
    return RiskContext(
        broker=BrokerSnapshot(
            taken_at=now - snapshot_age, position=position, equity=equity, open_orders=open_orders
        ),
        book=InternalBook(
            position=position if book_position is None else book_position,
            open_orders=open_orders if book_orders is None else book_orders,
        ),
        quote=Quote(bid=bid, ask=ask, at=now - quote_age),
        limit_price=limit,
        healthy=healthy,
        drift_frozen=drift_frozen,
        halted=halted,
    )


@dataclass
class Desk:
    """A risk engine on a simulated clock, with its session started."""

    engine: RiskEngine
    clock: SimClock

    def ctx(self, **kwargs: Any) -> RiskContext:
        return context(self.clock.now(), **kwargs)

    def check(self, intent: OrderIntent, **kwargs: Any) -> RiskVerdict:
        return self.engine.check(intent, self.ctx(**kwargs))

    def stop(
        self,
        quantity: int = 300,
        *,
        stop_price: float = STOP,
        position: int = 300,
        open_orders: tuple[OpenOrder, ...] = (),
        snapshot_age: timedelta = timedelta(0),
        quote_age: timedelta = timedelta(0),
        bid: float = PRICE,
        ask: float = PRICE,
        reason: str = "protect",
    ) -> RiskApprovedStop:
        """`RiskEngine.approve_stop` on a snapshot and a quote read now (B9c)."""
        now = self.clock.now()
        return self.engine.approve_stop(
            BrokerSnapshot(
                taken_at=now - snapshot_age,
                position=position,
                equity=EQUITY,
                open_orders=open_orders,
            ),
            Quote(bid=bid, ask=ask, at=now - quote_age),
            quantity=quantity,
            stop_price=stop_price,
            reason=reason,
        )


def desk(
    *,
    mode: TradingMode = TradingMode.PAPER,
    at: datetime | None = None,
    day: date | None = DAY,
    equity: float = EQUITY,
    spent: SpentApprovals | None = None,
    state: RiskState | None = None,
    risk_config: RiskConfig | None = None,
    on_change: Callable[[RiskState], None] | None = None,
) -> Desk:
    """A desk whose clock reads `at` (default: 30 s after DAY's first decision). It starts
    session `day` with `equity` unless `state` is given or `day` is None. `on_change` receives
    every new RiskState, as B14 will persist it."""
    clock = SimClock(decisions()[0].tau + timedelta(seconds=30) if at is None else at)
    engine = RiskEngine(
        config() if risk_config is None else risk_config,
        clock=clock,
        mode=mode,
        calendar=CAL,
        spent=spent,
        state=state,
        on_change=on_change,
    )
    if state is None and day is not None:
        engine.start_session(day, equity=equity)
    return Desk(engine, clock)
