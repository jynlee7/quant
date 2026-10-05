"""B1: canonical JSON, the byte form every hash and DecisionRecord comparison relies on
(ARCHITECTURE §4.3): sorted keys, shortest round-trip floats, NaN forbidden."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant.core.canonical import canonical_dumps, canonical_loads, sha256_hex
from quant.core.errors import CanonicalJSONError

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: (
        st.lists(children, max_size=6) | st.dictionaries(st.text(max_size=12), children, max_size=6)
    ),
    max_leaves=40,
)


@pytest.mark.spec("ARCH§4.3", "ARCH§14")
@given(json_values)
def test_any_json_value_round_trips_exactly(value: Any) -> None:
    data = canonical_dumps(value)
    assert canonical_loads(data) == value
    assert canonical_dumps(canonical_loads(data)) == data


@pytest.mark.spec("ARCH§4.3", "ARCH§14")
@given(st.dictionaries(st.text(max_size=8), json_values, min_size=2, max_size=8), st.randoms())
def test_the_hash_ignores_key_order(value: dict[str, Any], rng: Any) -> None:
    keys = list(value)
    rng.shuffle(keys)
    shuffled = {k: value[k] for k in keys}
    assert canonical_dumps(shuffled) == canonical_dumps(value)
    assert sha256_hex(shuffled) == sha256_hex(value)


@pytest.mark.spec("ARCH§4.3", "ARCH§14")
def test_the_canonical_bytes_are_fixed() -> None:
    value = {"b": [1, 2.5, None, True, 0.1], "a": "é", "c": {"z": -0.0, "y": 1e-07}}
    assert canonical_dumps(value) == (
        '{"a":"é","b":[1,2.5,null,true,0.1],"c":{"y":1e-07,"z":-0.0}}'.encode()
    )
    assert sha256_hex(value) == sha256_hex(canonical_loads(canonical_dumps(value)))
    assert len(sha256_hex(value)) == 64


@pytest.mark.spec("ARCH§4.3", "ARCH§14")
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_nan_and_infinity_are_rejected_anywhere(bad: float) -> None:
    for value in (bad, [1.0, bad], {"a": {"b": bad}}, np.float64(bad)):
        with pytest.raises(CanonicalJSONError):
            canonical_dumps(value)
    with pytest.raises(CanonicalJSONError):
        canonical_loads(b'{"a": NaN}')


@pytest.mark.spec("ARCH§4.3", "ARCH§14")
def test_values_without_one_canonical_form_are_rejected() -> None:
    for value in (
        {1: "int key"},
        {"a": {1, 2}},
        b"bytes",
        datetime(2019, 7, 2, 10, 46),  # noqa: DTZ001
        np.float32(0.1),
        np.array([1, 2], dtype=object),
    ):
        with pytest.raises(CanonicalJSONError):
            canonical_dumps(value)
    with pytest.raises(CanonicalJSONError):
        canonical_loads(b'{"a": 1, "a": 2}')


@pytest.mark.spec("ARCH§4.3", "ARCH§14")
def test_times_and_numpy_values_have_one_form() -> None:
    tau = datetime(2019, 7, 2, 14, 46, tzinfo=UTC)
    value = {
        "tau": tau,
        "day": date(2019, 7, 2),
        "p": np.array([0.25, 0.75]),
        "n": np.int64(7),
        "x": np.float64(0.1),
        "ok": np.bool_(True),
    }
    assert canonical_dumps(value) == (
        b'{"day":"2019-07-02","n":7,"ok":true,"p":[0.25,0.75],"tau":"2019-07-02T14:46:00Z","x":0.1}'
    )
