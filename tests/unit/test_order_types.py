"""B9: the order intent and the broker views in quant.core.orders (ARCHITECTURE §4.1, §4.2).

`decide()` (B8) emits OrderIntents and the risk engine reads them with a BrokerSnapshot, the
engine's InternalBook and a Quote. They are strict, immutable, UTC and in whole shares.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pydantic
import pytest

from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.ids import decision_id
from quant.core.orders import (
    SYMBOL,
    BrokerSnapshot,
    OrderIntent,
    Quote,
    Side,
)
from quant.core.regime import Label

TAU = datetime(2018, 3, 6, 15, 46, tzinfo=UTC)  # 10:46 ET


def fields(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "decision_id": decision_id(TAU),
        "tau": TAU,
        "side": Side.BUY,
        "quantity": 300,
        "target_exposure": 0.5,
        "label": Label.CALM_UP,
        "stop_price": 95.0,
    }
    return base | changes


@pytest.mark.spec("ARCH§4.1", "ARCH§4.2", "D9")
def test_an_intent_names_its_decision_side_shares_and_target() -> None:
    order = OrderIntent(**fields())
    assert (order.symbol, order.leg, order.reduce_only, order.uncertain) == (
        SYMBOL,
        1,
        False,
        False,
    )
    record = canonical_loads(canonical_dumps(order))
    assert record["tau"] == "2018-03-06T15:46:00Z"
    assert record["side"] == "buy"
    with pytest.raises(pydantic.ValidationError):
        order.quantity = 600  # type: ignore[misc]


@pytest.mark.spec("ARCH§4.1", "ARCH§4.2", "D9", "ARCH§14")
@pytest.mark.parametrize(
    "changes",
    [
        {"decision_id": decision_id(TAU + timedelta(hours=1))},
        {"tau": TAU.replace(tzinfo=None)},
        {"tau": TAU + timedelta(seconds=30)},
        {"symbol": "QQQ"},
        {"quantity": 0},
        {"quantity": 1.5},
        {"target_exposure": 1.01},
        {"target_exposure": -0.1},
        {"leg": 0},
        {"reduce_only": True},  # a BUY can't be reduce-only
        {"stop_price": float("nan")},
        {"label": "CALM_UP"},  # strict: the enum, not its text
        {"extra": 1},
    ],
    ids=[
        "another decision's id",
        "a naive tau",
        "tau off the minute",
        "another symbol",
        "no shares",
        "a fractional share",
        "a target above 1x",
        "a negative target",
        "leg 0",
        "a reduce-only buy",
        "a NaN stop",
        "a label as text",
        "an unknown key",
    ],
)
def test_an_inconsistent_intent_is_refused(changes: dict[str, Any]) -> None:
    OrderIntent(**fields())  # the base is valid
    with pytest.raises(pydantic.ValidationError):
        OrderIntent(**fields(**changes))


@pytest.mark.spec("ARCH§4.2", "ARCH§8.1")
def test_snapshots_and_quotes_refuse_naive_times() -> None:
    naive = TAU.replace(tzinfo=None)
    BrokerSnapshot(taken_at=TAU, position=0, equity=100_000.0)
    Quote(bid=100.0, ask=100.0, at=TAU)
    with pytest.raises(pydantic.ValidationError):
        BrokerSnapshot(taken_at=naive, position=0, equity=100_000.0)
    with pytest.raises(pydantic.ValidationError):
        Quote(bid=100.0, ask=100.0, at=naive)
    with pytest.raises(pydantic.ValidationError):
        BrokerSnapshot(taken_at=TAU, position=0, equity=0.0)
    OrderIntent(**fields(side=Side.SELL, stop_price=None, reduce_only=True))  # a reduce-only sell
