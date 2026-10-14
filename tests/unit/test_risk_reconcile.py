"""B9b: reconciliation compares the broker's position and orders with the engine's, and a
mismatch is what check 2 halts on (spec §10, ARCHITECTURE §8.1 check 2, §8.2, §16).
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from quant.core.orders import BrokerSnapshot, InternalBook, OpenOrder, OrderKind, Side
from quant.risk.engine import RiskCheck, RiskContext, Veto
from quant.risk.reconcile import (
    Discrepancy,
    DiscrepancyKind,
    ReconcileStatus,
    reconcile,
)
from tests.support.risk import EQUITY, PRICE, decisions, desk, intent
from tests.support.risk import order as open_order

RED = pytest.mark.xfail(strict=True, reason="red:B9", raises=NotImplementedError)
NOW = decisions()[0].tau + timedelta(seconds=30)  # the desk's default clock
MAX = 5.0
STOP = open_order("i1-stop", Side.SELL, OrderKind.STOP, 100, 95.0)
ENTRY = open_order("i1-o-1", Side.BUY, OrderKind.LIMIT, 50, 100.10)


def snap(
    position: int = 100, orders: tuple[OpenOrder, ...] = (STOP,), age: float = 0.0
) -> BrokerSnapshot:
    return BrokerSnapshot(
        taken_at=NOW - timedelta(seconds=age),
        position=position,
        equity=EQUITY,
        open_orders=orders,
    )


def book(position: int = 100, orders: tuple[OpenOrder, ...] = (STOP,)) -> InternalBook:
    return InternalBook(position=position, open_orders=orders)


def kinds(result: object) -> list[DiscrepancyKind]:
    return [d.kind for d in result.discrepancies]  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§10", "ARCH§8.1")
@RED
def test_a_broker_that_agrees_with_the_book_matches() -> None:
    result = reconcile(
        snap(orders=(STOP, ENTRY)), book(orders=(ENTRY, STOP)), now=NOW, max_age_s=MAX
    )
    assert result.status is ReconcileStatus.MATCH
    assert not result.mismatch
    assert result.discrepancies == ()
    assert result.checked_at == NOW
    assert result.snapshot_age_s == 0.0


@pytest.mark.spec("SPEC§10", "ARCH§8.1")
@RED
@pytest.mark.parametrize(
    ("age", "status"),
    [
        (5.0, ReconcileStatus.MATCH),  # exactly the maximum is fresh
        (5.000001, ReconcileStatus.STALE),
        (-0.000001, ReconcileStatus.STALE),  # from the future
    ],
)
def test_a_snapshot_too_old_or_from_the_future_is_stale_and_compares_nothing(
    age: float, status: ReconcileStatus
) -> None:
    result = reconcile(snap(age=age), book(), now=NOW, max_age_s=MAX)
    assert result.status is status
    stale_and_wrong = reconcile(snap(position=7, age=age), book(), now=NOW, max_age_s=MAX)
    if status is ReconcileStatus.STALE:
        assert stale_and_wrong.status is ReconcileStatus.STALE
        assert stale_and_wrong.discrepancies == ()
        assert not stale_and_wrong.mismatch
    else:
        assert stale_and_wrong.mismatch


@pytest.mark.spec("SPEC§10", "ARCH§8.1", "D9")
@RED
def test_a_short_and_a_position_difference_are_mismatches() -> None:
    result = reconcile(snap(position=-10), book(position=0), now=NOW, max_age_s=MAX)
    assert result.status is ReconcileStatus.MISMATCH
    assert kinds(result) == [DiscrepancyKind.SHORT, DiscrepancyKind.POSITION]
    assert result.discrepancies[0].client_order_id is None

    both_short = reconcile(snap(position=-10), book(position=-10), now=NOW, max_age_s=MAX)
    assert kinds(both_short) == [DiscrepancyKind.SHORT]

    differs = reconcile(snap(position=100), book(position=0), now=NOW, max_age_s=MAX)
    assert kinds(differs) == [DiscrepancyKind.POSITION]
    assert "100" in differs.discrepancies[0].detail
    assert "0" in differs.discrepancies[0].detail


@pytest.mark.spec("SPEC§10", "ARCH§8.1", "ARCH§16")
@RED
def test_order_differences_are_named_by_id_in_sorted_order() -> None:
    changed_stop = open_order("i1-stop", Side.SELL, OrderKind.STOP, 60, 95.0)  # part-filled
    stray = open_order("i1-a", Side.SELL, OrderKind.LIMIT, 10, 99.0)
    lost = open_order("i1-z", Side.BUY, OrderKind.LIMIT, 10, 101.0)
    result = reconcile(
        snap(orders=(changed_stop, stray)),
        book(orders=(STOP, lost)),
        now=NOW,
        max_age_s=MAX,
    )
    assert result.mismatch
    assert result.discrepancies == (
        Discrepancy(DiscrepancyKind.UNKNOWN_TO_ENGINE, result.discrepancies[0].detail, "i1-a"),
        Discrepancy(DiscrepancyKind.ORDER_DIFFERS, result.discrepancies[1].detail, "i1-stop"),
        Discrepancy(DiscrepancyKind.MISSING_AT_BROKER, result.discrepancies[2].detail, "i1-z"),
    )
    assert all(d.detail for d in result.discrepancies)


@pytest.mark.spec("SPEC§10", "ARCH§8.1")
@RED
def test_a_duplicated_order_under_one_id_differs() -> None:
    result = reconcile(snap(orders=(STOP, STOP)), book(orders=(STOP,)), now=NOW, max_age_s=MAX)
    assert kinds(result) == [DiscrepancyKind.ORDER_DIFFERS]


@pytest.mark.spec("SPEC§10", "ARCH§8.2")
@RED
def test_an_order_from_another_instance_fails_reconciliation() -> None:
    foreign = open_order("i2-o-1", Side.BUY, OrderKind.LIMIT, 50, 100.10)
    result = reconcile(
        snap(orders=(STOP, foreign)),
        book(orders=(STOP, foreign)),
        now=NOW,
        max_age_s=MAX,
        instance_prefix="i1-",
    )
    assert result.mismatch
    assert [(d.kind, d.client_order_id) for d in result.discrepancies] == [
        (DiscrepancyKind.FOREIGN, "i2-o-1")
    ]
    assert not reconcile(
        snap(orders=(STOP, foreign)), book(orders=(STOP, foreign)), now=NOW, max_age_s=MAX
    ).mismatch  # no prefix, no instance check


@pytest.mark.spec("ARCH§8.1")
@RED
def test_a_bad_maximum_age_is_refused() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_age_s"):
            reconcile(snap(), book(), now=NOW, max_age_s=bad)


CASES = {
    "agree": (snap(), book()),
    "position": (snap(position=90), book()),
    "short": (snap(position=-5, orders=()), book(position=-5, orders=())),
    "missing order": (snap(orders=()), book()),
    "unknown order": (snap(orders=(STOP, ENTRY)), book()),
    "changed order": (snap(orders=(open_order("i1-stop", Side.SELL, OrderKind.STOP, 99),)), book()),
    "stale": (snap(position=90, age=6.0), book()),
}


@pytest.mark.spec("SPEC§10", "ARCH§8.1")
@RED
@pytest.mark.parametrize("case", list(CASES))
def test_check_2_halts_exactly_when_reconciliation_finds_a_mismatch(case: str) -> None:
    broker, internal = CASES[case]
    d = desk()
    assert d.clock.now() == NOW
    result = reconcile(broker, internal, now=NOW, max_age_s=MAX)
    quote = d.ctx().quote
    verdict = d.engine.check(
        intent(Side.SELL, 10),
        RiskContext(broker=broker, book=internal, quote=quote, limit_price=PRICE),
    )
    at_check_2 = isinstance(verdict, Veto) and verdict.check is RiskCheck.RECONCILIATION
    halted = isinstance(verdict, Veto) and verdict.halts
    assert (at_check_2 and halted) is result.mismatch
    assert (at_check_2 and not halted) is (result.status is ReconcileStatus.STALE)
