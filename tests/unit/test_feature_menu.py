"""B3: the feature menu and the feature-config hash (spec §3, P8, PLAN §5)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from quant.core.canonical import sha256_hex
from quant.features.menu import MENU, FeatureConfig, Trend, r1_features

RED = pytest.mark.xfail(strict=True, reason="red:B3", raises=NotImplementedError)

R1: dict[str, Any] = {
    "slot_sessions": 60,
    "ret": {},
    "rvol": {"bars": 35},
    "range": {},
    "volume": {},
    "trend": {"fast": 35, "slow": 175},
}


def _with(base: dict[str, Any], path: tuple[str, ...], value: object) -> dict[str, Any]:
    """A copy of `base` with the value at `path` replaced."""
    if len(path) == 1:
        return {**base, path[0]: value}
    return {**base, path[0]: _with(base[path[0]], path[1:], value)}


@pytest.mark.spec("P8", "D3", "PLAN§5")
@RED
def test_r1_turns_on_all_five_transforms_with_the_plan_values() -> None:
    config = r1_features()
    assert config == FeatureConfig.model_validate(R1)
    assert config.columns == MENU == ("ret", "rvol", "range", "volume", "trend")
    assert config.trend is not None
    assert (config.slot_sessions, config.rvol.bars, config.trend.fast, config.trend.slow) == (
        60,
        35,
        35,
        175,
    )
    # Defaults are r1's values, and columns keep the menu order whatever the input order.
    sparse = FeatureConfig.model_validate({"trend": {}, "rvol": {}, "ret": {}})
    assert sparse.columns == ("ret", "rvol", "trend")
    assert sparse.rvol.bars == 35
    assert sparse.slot_sessions == 60


@pytest.mark.spec("P8", "SPEC§3")
@RED
def test_the_menu_rejects_unknown_transforms() -> None:
    assert FeatureConfig.model_validate(R1).columns == MENU
    for unknown in ({"macd": {}}, {"F6": {}}, {"ewma_vol": {"bars": 20}}):
        with pytest.raises(ValidationError):
            FeatureConfig.model_validate({**R1, **unknown})
    with pytest.raises(ValidationError):  # an unknown parameter of a known transform
        FeatureConfig.model_validate(_with(R1, ("rvol", "decay"), 0.9))
    with pytest.raises(ValidationError):  # a transform is its parameters, not a flag
        FeatureConfig.model_validate({**R1, "range": True})


@pytest.mark.spec("P8", "PLAN§5")
@pytest.mark.parametrize(
    ("path", "low", "high"),
    [
        (("slot_sessions",), 20, 120),
        (("rvol", "bars"), 14, 70),
        (("trend", "fast"), 14, 70),
        (("trend", "slow"), 70, 350),
    ],
)
@RED
def test_the_menu_rejects_out_of_bounds_parameters(
    path: tuple[str, ...], low: int, high: int
) -> None:
    for value in (low, high):
        assert FeatureConfig.model_validate(_with(R1, path, value)).columns == MENU
    for value in (low - 1, high + 1, -1, 0):
        with pytest.raises(ValidationError):
            FeatureConfig.model_validate(_with(R1, path, value))
    for value in (float(low), str(low), True):  # strict: no coercion
        with pytest.raises(ValidationError):
            FeatureConfig.model_validate(_with(R1, path, value))


@pytest.mark.spec("P8", "D3", "PLAN§5")
@RED
def test_the_menu_rejects_a_fast_trend_not_below_the_slow_one() -> None:
    assert Trend(fast=69, slow=70).fast == 69
    assert Trend(fast=70, slow=71).slow == 71
    with pytest.raises(ValidationError):
        Trend(fast=70, slow=70)  # within both bounds, the one way fast can reach slow
    with pytest.raises(ValidationError):
        FeatureConfig.model_validate(_with(R1, ("trend",), {"fast": 70, "slow": 70}))


@pytest.mark.spec("P8", "SPEC§3")
@RED
def test_the_menu_rejects_configs_without_f1_or_f2() -> None:
    assert FeatureConfig.model_validate({"ret": {}, "rvol": {}}).columns == ("ret", "rvol")
    for always_on in ("ret", "rvol"):
        without = {key: value for key, value in R1.items() if key != always_on}
        with pytest.raises(ValidationError):
            FeatureConfig.model_validate(without)
        with pytest.raises(ValidationError):
            FeatureConfig.model_validate({**R1, always_on: None})


@pytest.mark.spec("P8", "ARCH§7.2")
@RED
def test_the_config_hash_is_stable_under_key_order_and_changes_with_any_parameter() -> None:
    config = FeatureConfig.model_validate(R1)
    reordered = FeatureConfig.model_validate(
        {
            "trend": {"slow": 175, "fast": 35},
            "volume": {},
            "range": {},
            "rvol": {"bars": 35},
            "ret": {},
            "slot_sessions": 60,
        }
    )
    assert config.config_hash() == reordered.config_hash() == sha256_hex(config)
    assert len(config.config_hash()) == 64
    assert FeatureConfig.model_validate(R1).config_hash() == config.config_hash()

    # A transform that is off hashes the same whether omitted or null, and defaults hash like
    # the same values written out.
    no_volume = {key: value for key, value in R1.items() if key != "volume"}
    assert (
        FeatureConfig.model_validate(no_volume).config_hash()
        == FeatureConfig.model_validate({**no_volume, "volume": None}).config_hash()
    )
    defaults = {"ret": {}, "rvol": {}, "range": {}, "volume": {}, "trend": {}}
    assert FeatureConfig.model_validate(defaults).config_hash() == config.config_hash()

    changed = [
        _with(R1, ("slot_sessions",), 61),
        _with(R1, ("rvol", "bars"), 36),
        _with(R1, ("trend", "fast"), 34),
        _with(R1, ("trend", "slow"), 176),
        no_volume,
        {key: value for key, value in R1.items() if key != "trend"},
    ]
    hashes = {config.config_hash()} | {
        FeatureConfig.model_validate(c).config_hash() for c in changed
    }
    assert len(hashes) == 1 + len(changed)
