"""B1: ids and the strict base model shared by every schema."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import pydantic
import pytest

from quant.core.calendar import TradingCalendar
from quant.core.ids import decision_id
from quant.core.model import StrictModel
from quant.core.schedule import decision_schedule


class Example(StrictModel):
    name: str
    size: float


@pytest.mark.spec("ARCH§4.3")
def test_decision_ids_are_derived_from_tau_and_unique() -> None:
    tau = datetime(2019, 7, 2, 14, 46, tzinfo=UTC)
    assert decision_id(tau) == "d20190702T1446Z"
    cal = TradingCalendar.xnys()
    ids = [
        decision_id(d.tau)
        for day in cal.sessions(date(2019, 1, 1), date(2019, 12, 31))
        for d in decision_schedule(cal, day)
    ]
    assert len(ids) == len(set(ids))


@pytest.mark.spec("ARCH§3", "ARCH§11")
def test_strict_models_reject_unknown_keys_coercion_nan_and_mutation() -> None:
    assert Example(name="r1", size=0.5).size == 0.5
    with pytest.raises(pydantic.ValidationError):
        Example.model_validate({"name": "r1", "size": 0.5, "extra": 1})
    with pytest.raises(pydantic.ValidationError):
        Example.model_validate({"name": "r1", "size": "0.5"})
    with pytest.raises(pydantic.ValidationError):
        Example(name="r1", size=math.nan)
    model = Example(name="r1", size=0.5)
    with pytest.raises(pydantic.ValidationError):
        model.size = 1.0  # type: ignore[misc]
