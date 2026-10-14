"""Reconciliation: the broker's positions and orders against the engine's (spec §10, ARCHITECTURE
§8.1 check 2, §16).

Broker and internal positions must match before every order and every minute. A mismatch halts
trading and sends an alert, and fires the kill switch (spec §10); it is never auto-corrected, so
`reconcile` is a pure comparison that changes nothing. A snapshot too old (or from the future)
to compare is STALE, which is not a mismatch: it vetoes an order (check 2) but halts nothing.

`RiskEngine.check`'s check 2 must agree with it: with `instance_prefix=None`, check 2 halts on a
fresh snapshot exactly when `reconcile` returns MISMATCH, and vetoes a snapshot without halting
exactly when it returns STALE. The green step may make check 2 call `reconcile`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite

from quant.core.clock import ensure_utc
from quant.core.orders import BrokerSnapshot, InternalBook, OpenOrder


class ReconcileStatus(StrEnum):
    MATCH = "match"  # fresh, and every comparison agrees
    MISMATCH = "mismatch"  # fresh, and at least one Discrepancy: halt, alert, kill
    STALE = "stale"  # too old or from the future to compare; nothing is compared


class DiscrepancyKind(StrEnum):
    """What one Discrepancy found, in the order `reconcile` reports them."""

    SHORT = "short"  # the broker holds a negative position: never allowed (D9)
    POSITION = "position"  # broker and book hold different share counts
    MISSING_AT_BROKER = "missing_at_broker"  # the book has an order id the broker doesn't
    UNKNOWN_TO_ENGINE = "unknown_to_engine"  # the broker has an order id the book doesn't
    ORDER_DIFFERS = "order_differs"  # same id on both sides, but not the same orders
    FOREIGN = "foreign"  # a broker order whose id isn't this instance's (ARCHITECTURE §8.2)


@dataclass(frozen=True)
class Discrepancy:
    """One difference. `client_order_id` names the order for the order kinds, and is None for
    SHORT and POSITION. `detail` says what differs in plain words, both sides' values included
    (for example "broker 100 shares, engine 0")."""

    kind: DiscrepancyKind
    detail: str
    client_order_id: str | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    """What one comparison found.

    - `status`: MATCH, MISMATCH or STALE;
    - `checked_at`: the `now` it was judged at;
    - `snapshot_age_s`: `(now - snapshot.taken_at)` in seconds, negative from the future;
    - `discrepancies`: empty unless MISMATCH; never empty when MISMATCH.
    """

    status: ReconcileStatus
    checked_at: datetime
    snapshot_age_s: float
    discrepancies: tuple[Discrepancy, ...] = ()

    @property
    def mismatch(self) -> bool:
        """True when the comparison found a discrepancy: halt, alert and fire the kill."""
        return self.status is ReconcileStatus.MISMATCH


def reconcile(
    snapshot: BrokerSnapshot,
    book: InternalBook,
    *,
    now: datetime,
    max_age_s: float,
    instance_prefix: str | None = None,
) -> ReconciliationResult:
    """Compare `snapshot` (the broker) with `book` (the engine) at `now`, changing neither.

    STALE, with no discrepancies, when `now - snapshot.taken_at` is more than `max_age_s`
    seconds (`RiskConfig.snapshot_max_age_s`; exactly the maximum is fresh) or negative.

    Otherwise every discrepancy is listed, in this order:

    1. SHORT when `snapshot.position < 0`;
    2. POSITION when `snapshot.position != book.position`;
    3. the orders, grouped by `client_order_id` and visited in sorted id order:
       - FOREIGN when `instance_prefix` is given and a broker order's id doesn't start with it
         (an order from an unknown instance fails reconciliation, ARCHITECTURE §8.2); a FOREIGN
         id gets no other order discrepancy;
       - MISSING_AT_BROKER when only the book has the id;
       - UNKNOWN_TO_ENGINE when only the broker has it;
       - ORDER_DIFFERS when both have it, but its orders differ as multisets of OpenOrder
         (side, kind, remaining quantity or price; or a different count of orders under it).

    MISMATCH when there is at least one discrepancy, else MATCH. With `instance_prefix=None`,
    the orders produce a discrepancy exactly when `Counter(snapshot.open_orders) !=
    Counter(book.open_orders)`, as check 2 compares them.

    ClockError for a naive `now`. ValueError when `max_age_s` isn't finite and positive. Never raises for any valid snapshot
    and book.
    """
    if (
        isinstance(max_age_s, bool)
        or not isinstance(max_age_s, (int, float))
        or not isfinite(max_age_s)
        or max_age_s <= 0
    ):
        raise ValueError(f"max_age_s must be finite and positive, not {max_age_s!r}")
    now = ensure_utc(now)
    age = (now - snapshot.taken_at).total_seconds()
    if not 0 <= age <= max_age_s:
        return ReconciliationResult(ReconcileStatus.STALE, now, age)

    found: list[Discrepancy] = []
    if snapshot.position < 0:
        found.append(
            Discrepancy(DiscrepancyKind.SHORT, f"broker holds {snapshot.position} shares, a short")
        )
    if snapshot.position != book.position:
        found.append(
            Discrepancy(
                DiscrepancyKind.POSITION,
                f"broker {snapshot.position} shares, engine {book.position}",
            )
        )
    at_broker = _by_id(snapshot.open_orders)
    in_book = _by_id(book.open_orders)
    for order_id in sorted(at_broker.keys() | in_book.keys()):
        broker_orders, book_orders = at_broker.get(order_id, []), in_book.get(order_id, [])
        if (
            broker_orders
            and instance_prefix is not None
            and not order_id.startswith(instance_prefix)
        ):
            detail = f"broker order {_describe(broker_orders)} isn't this instance's"
            found.append(Discrepancy(DiscrepancyKind.FOREIGN, detail, order_id))
        elif not broker_orders:
            detail = f"engine order {_describe(book_orders)}, none at the broker"
            found.append(Discrepancy(DiscrepancyKind.MISSING_AT_BROKER, detail, order_id))
        elif not book_orders:
            detail = f"broker order {_describe(broker_orders)}, none in the engine"
            found.append(Discrepancy(DiscrepancyKind.UNKNOWN_TO_ENGINE, detail, order_id))
        elif Counter(broker_orders) != Counter(book_orders):
            detail = f"broker {_describe(broker_orders)}, engine {_describe(book_orders)}"
            found.append(Discrepancy(DiscrepancyKind.ORDER_DIFFERS, detail, order_id))

    status = ReconcileStatus.MISMATCH if found else ReconcileStatus.MATCH
    return ReconciliationResult(status, now, age, tuple(found))


def _by_id(orders: tuple[OpenOrder, ...]) -> dict[str, list[OpenOrder]]:
    grouped: dict[str, list[OpenOrder]] = {}
    for order in orders:
        grouped.setdefault(order.client_order_id, []).append(order)
    return grouped


def _describe(orders: list[OpenOrder]) -> str:
    """`orders` in plain words, for example "i1-stop: sell stop 100 @ 95.0"."""
    return ", ".join(
        f"{o.client_order_id}: {o.side.value} {o.kind.value} {o.quantity} @ {o.price}"
        for o in orders
    )
