"""B8a: config/engine.yaml, the engine's L3 config (ARCHITECTURE §9.2; spec §8, §9; D2, D5, C12).

The committed file holds spec §8's switching thresholds, spec §9's sizing constants and the
decision timing. It parses strictly (unknown and missing keys, wrong types and out-of-bounds
values refused), loads only as committed byte for byte, and hashes its parsed values, so a
comment changes nothing and any value changes the hash.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from quant.core.canonical import sha256_hex
from quant.core.errors import ConfigError
from quant.core.schedule import Timing
from tests.support.decide import ENGINE_YAML


def committed() -> bytes:
    return ENGINE_YAML.read_bytes()


def values() -> dict[str, Any]:
    data = yaml.safe_load(committed())
    assert isinstance(data, dict)
    return copy.deepcopy(data)


def parse(data: dict[str, Any] | bytes) -> Any:
    from quant.decide.config import parse_engine_config

    raw = data if isinstance(data, bytes) else yaml.safe_dump(data).encode()
    return parse_engine_config(raw)


@pytest.mark.spec("ARCH§9.2", "SPEC§8", "D5", "A1")
def test_the_committed_file_holds_spec_8s_switching_values() -> None:
    from quant.decide.config import load_engine_config

    s = load_engine_config(ENGINE_YAML, committed=committed()).switching
    assert (s.takeover_prob, s.persistence_bars, s.cooldown_bars) == (0.70, 3, 7)
    assert (s.derisk_prob, s.derisk_multiplier) == (0.25, 0.5)
    assert s.uncertainty_gap == 0.15
    assert s.health_recovery_bars == 2


@pytest.mark.spec("ARCH§9.2", "SPEC§9", "C12")
def test_the_committed_file_holds_spec_9s_sizing_values() -> None:
    from quant.decide.config import load_engine_config

    s = load_engine_config(ENGINE_YAML, committed=committed()).sizing
    assert (s.kelly_fraction, s.min_edge_bars, s.deadband, s.shadow_cost_bp) == (
        0.25,
        100,
        0.10,
        2.0,
    )


@pytest.mark.spec("ARCH§9.2", "D2")
def test_the_timing_is_the_schedules_default() -> None:
    from quant.decide.config import load_engine_config

    config = load_engine_config(ENGINE_YAML, committed=committed())
    assert (config.timing.feed_delay_min, config.timing.decision_delay_min) == (15, 16)
    assert config.timing.timing() == Timing()


@pytest.mark.spec("ARCH§9.2")
def test_the_config_hash_is_of_the_parsed_values() -> None:
    config = parse(committed())
    assert config.config_hash() == sha256_hex(config)
    # Comments, blank lines and key order don't change it; a value does.
    reordered = yaml.safe_dump(values(), sort_keys=True).encode()
    assert parse(reordered).config_hash() == config.config_hash()
    bumped = values()
    bumped["switching"]["cooldown_bars"] = 8
    assert parse(bumped).config_hash() != config.config_hash()


@pytest.mark.spec("ARCH§9.2")
def test_a_file_that_differs_from_the_committed_bytes_is_refused(tmp_path: Path) -> None:
    from quant.decide.config import load_engine_config

    path = tmp_path / "engine.yaml"
    path.write_bytes(committed() + b"# one more comment\n")
    with pytest.raises(ConfigError):
        load_engine_config(path, committed=committed())
    path.write_bytes(committed())
    assert load_engine_config(path, committed=committed()) == parse(committed())
    with pytest.raises(ConfigError):
        load_engine_config(tmp_path / "missing.yaml", committed=committed())


@pytest.mark.spec("ARCH§9.2")
def test_the_config_is_frozen() -> None:
    from pydantic import ValidationError

    config = parse(committed())
    with pytest.raises(ValidationError):
        config.switching.cooldown_bars = 0
    with pytest.raises(ValidationError):
        config.sizing = config.sizing


def _set(section: str, key: str, value: Any) -> dict[str, Any]:
    data = values()
    data[section][key] = value
    return data


def _drop(section: str, key: str | None) -> dict[str, Any]:
    data = values()
    if key is None:
        del data[section]
    else:
        del data[section][key]
    return data


REFUSED = {
    "unknown top key": {**values(), "collar_bp": 10.0},
    "unknown switching key": _set("switching", "hysteresis", 2),
    "unknown sizing key": _set("sizing", "max_kelly", 1.0),
    "unknown timing key": _set("timing", "jitter_s", 1),
    "no switching": _drop("switching", None),
    "no sizing": _drop("sizing", None),
    "no timing": _drop("timing", None),
    "no takeover_prob": _drop("switching", "takeover_prob"),
    "no health_recovery_bars": _drop("switching", "health_recovery_bars"),
    "no deadband": _drop("sizing", "deadband"),
    "no decision_delay_min": _drop("timing", "decision_delay_min"),
    "bool for int": _set("switching", "persistence_bars", True),
    "float for int": _set("switching", "cooldown_bars", 7.0),
    "string for float": _set("switching", "takeover_prob", "0.70"),
    "null for float": _set("sizing", "kelly_fraction", None),
    "nan": _set("sizing", "shadow_cost_bp", float("nan")),
    "takeover at 0.5": _set("switching", "takeover_prob", 0.5),
    "takeover above 1": _set("switching", "takeover_prob", 1.01),
    "no persistence": _set("switching", "persistence_bars", 0),
    "negative cooldown": _set("switching", "cooldown_bars", -1),
    "derisk_prob 0": _set("switching", "derisk_prob", 0.0),
    "derisk_prob 1": _set("switching", "derisk_prob", 1.0),
    "derisk multiplier above 1": _set("switching", "derisk_multiplier", 1.5),
    "uncertainty gap 1": _set("switching", "uncertainty_gap", 1.0),
    "no recovery bars": _set("switching", "health_recovery_bars", 0),
    "kelly 0": _set("sizing", "kelly_fraction", 0.0),
    "kelly above 1": _set("sizing", "kelly_fraction", 1.5),
    "one edge bar": _set("sizing", "min_edge_bars", 1),
    "deadband 1": _set("sizing", "deadband", 1.0),
    "negative cost": _set("sizing", "shadow_cost_bp", -0.5),
    "decision not after the feed": _set("timing", "decision_delay_min", 15),
    "decision past the next bar": _set("timing", "decision_delay_min", 46),
    "negative feed delay": _set("timing", "feed_delay_min", -1),
}


@pytest.mark.spec("ARCH§9.2", "SPEC§8", "SPEC§9")
@pytest.mark.parametrize("case", sorted(REFUSED))
def test_a_bad_engine_config_is_refused(case: str) -> None:
    with pytest.raises(ConfigError):
        parse(REFUSED[case])


@pytest.mark.spec("ARCH§9.2")
@pytest.mark.parametrize(
    "raw", [b"\xff\xfe", b"switching: [unclosed", b"- a list\n", b"", b"just a string\n"]
)
def test_text_that_isnt_a_yaml_mapping_is_refused(raw: bytes) -> None:
    with pytest.raises(ConfigError):
        parse(raw)


@pytest.mark.spec("ARCH§9.2", "SPEC§8")
@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("switching", "takeover_prob", 1.0),
        ("switching", "takeover_prob", 1),  # a YAML integer for a float field
        ("switching", "cooldown_bars", 0),
        ("switching", "derisk_multiplier", 0.0),
        ("switching", "uncertainty_gap", 0.0),
        ("sizing", "kelly_fraction", 1.0),
        ("sizing", "deadband", 0.0),
        ("sizing", "min_edge_bars", 2),
        ("timing", "decision_delay_min", 45),
    ],
)
def test_a_value_at_its_bound_is_accepted(section: str, key: str, value: Any) -> None:
    config = parse(_set(section, key, value))
    assert getattr(getattr(config, section), key) == value
