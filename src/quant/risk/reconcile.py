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

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from quant.core.orders import BrokerSnapshot, InternalBook


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

    ValueError when `max_age_s` isn't finite and positive. Never raises for any valid snapshot
    and book.
    """
    raise NotImplementedError("B9b: reconcile")
