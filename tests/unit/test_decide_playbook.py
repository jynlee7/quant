"""B8a: the playbook schema (spec §7, PLAN §5's bounds, D8, D9).

PLAN B8's red test "the schema rejects unknown keys and out-of-bounds values": every bound is
tried at the bound (accepted) and just past it (refused, the next float or the next integer),
unknown keys are refused at the top level, in `params` and in an invalidation, and the label
rules hold: CRASH must be flat, STRESS's `max_size` is at most 0.25. Each case is a playbook
file built from one of r1's own front matters with one change, and goes through
`parse_playbook`, so a refusal is a ConfigError.
"""

from __future__ import annotations

import copy
import math
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import pytest

from quant.core.errors import ConfigError
from quant.core.regime import Label
from tests.support.decide import PLAYBOOKS, front_matter, playbook_text

DOWN, UP = -math.inf, math.inf


def below(x: float) -> float:
    return math.nextafter(x, DOWN)


def above(x: float) -> float:
    return math.nextafter(x, UP)


def base(label: Label) -> dict[str, Any]:
    """r1's front matter for `label`, as a fresh mapping."""
    return copy.deepcopy(front_matter(PLAYBOOKS[label]))


def changed(label: Label, path: str, value: Any) -> dict[str, Any]:
    """r1's `label` front matter with the dotted `path` set to `value` ("params.fast",
    "invalidation.0.threshold")."""
    front = base(label)
    *parents, last = path.split(".")
    node: Any = front
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return front


def parse(front: dict[str, Any]) -> Any:
    from quant.decide.playbook import parse_playbook

    return parse_playbook(playbook_text(front))


# --- r1's files -----------------------------------------------------------------------------


@pytest.mark.spec("SPEC§7", "PLAN§5", "P7")
@pytest.mark.parametrize("label", list(Label))
def test_each_r1_playbook_validates(label: Label) -> None:
    playbook = parse(base(label))
    assert playbook.label == label
    assert playbook.version == 1
    assert playbook.created_at == datetime(2026, 10, 19, tzinfo=UTC)


@pytest.mark.spec("SPEC§7")
def test_the_prose_body_is_ignored() -> None:
    from quant.decide.playbook import parse_playbook

    front = base(Label.CHOP)
    plain = parse_playbook(playbook_text(front, body=""))
    noisy = parse_playbook(
        playbook_text(front, body="\n---\nstop_loss: 99\n: not yaml [\n---\nmore prose\n")
    )
    assert plain == noisy


# --- unknown and missing keys ---------------------------------------------------------------


@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize(
    ("label", "where"),
    [
        (Label.CALM_UP, "top"),
        (Label.CALM_UP, "params"),
        (Label.CALM_UP, "invalidation"),
        (Label.CHOP, "top"),
        (Label.CHOP, "params"),
        (Label.CHOP, "invalidation"),
        (Label.STRESS, "top"),
        (Label.STRESS, "params"),
    ],
)
def test_an_unknown_key_is_refused(label: Label, where: str) -> None:
    front = base(label)
    if where == "top":
        front["entry_rule"] = "close > ema"
    elif where == "params":
        front["params"]["lookback"] = 35
    else:
        front["invalidation"][0]["extra"] = 1
    with pytest.raises(ConfigError):
        parse(front)


@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize(
    "key",
    [
        "label",
        "strategy",
        "params",
        "stop_loss",
        "take_profit",
        "max_size",
        "version",
        "created_at",
        "invalidation",
    ],
)
@pytest.mark.parametrize("label", [Label.CALM_UP, Label.STRESS])
def test_every_key_is_required(label: Label, key: str) -> None:
    # stop_loss is mandatory (spec §7), and take_profit must be written even when it is null.
    front = base(label)
    del front[key]
    with pytest.raises(ConfigError):
        parse(front)


@pytest.mark.spec("SPEC§7", "PLAN§5")
@pytest.mark.parametrize("param", ["fast", "slow", "atr_bars"], ids=lambda p: f"trend-{p}")
def test_a_missing_trend_param_is_refused(param: str) -> None:
    front = base(Label.CALM_UP)
    del front["params"][param]
    with pytest.raises(ConfigError):
        parse(front)


@pytest.mark.spec("SPEC§7", "PLAN§5")
@pytest.mark.parametrize("param", ["n", "entry_z", "exit_z", "atr_bars"])
def test_a_missing_mean_reversion_param_is_refused(param: str) -> None:
    front = base(Label.CHOP)
    del front["params"][param]
    with pytest.raises(ConfigError):
        parse(front)


# --- bounds: accepted at the bound, refused just past it -----------------------------------

ACCEPTED: list[tuple[Label, str, Any]] = [
    # trend_following: fast 14-70, slow 70-350, atr_bars 14-70
    (Label.CALM_UP, "params.fast", 14),
    (Label.CALM_UP, "params.fast", 70),
    (Label.CALM_UP, "params.slow", 70),
    (Label.CALM_UP, "params.slow", 350),
    (Label.CALM_UP, "params.atr_bars", 14),
    (Label.CALM_UP, "params.atr_bars", 70),
    # stop 2-8 x ATR, take-profit null or 2-10 x ATR, max_size 0-1
    (Label.CALM_UP, "stop_loss", 2.0),
    (Label.CALM_UP, "stop_loss", 8.0),
    (Label.CALM_UP, "stop_loss", 4),  # a YAML integer for a float field
    (Label.CALM_UP, "take_profit", 2.0),
    (Label.CALM_UP, "take_profit", 10.0),
    (Label.CALM_UP, "max_size", 0.0),
    (Label.CALM_UP, "max_size", 1.0),
    # mean_reversion: n 14-70, entry_z 0.5-3.0, exit_z -1.0-1.0 below entry_z
    (Label.CHOP, "params.n", 14),
    (Label.CHOP, "params.n", 70),
    (Label.CHOP, "params.entry_z", 0.5),
    (Label.CHOP, "params.entry_z", 3.0),
    (Label.CHOP, "params.exit_z", -1.0),
    (Label.CHOP, "params.exit_z", 1.0),  # entry_z is 1.5
    (Label.CHOP, "params.atr_bars", 14),
    (Label.CHOP, "params.atr_bars", 70),
    (Label.CHOP, "stop_loss", 2.0),
    (Label.CHOP, "stop_loss", 8.0),
    # the invalidation menu: label_prob_below in (0, 1), bars_in_trade_exceeds 7-70
    (Label.CHOP, "invalidation.1.threshold", 5e-324),
    (Label.CHOP, "invalidation.1.threshold", below(1.0)),
    (Label.CHOP, "invalidation.0.bars", 7),
    (Label.CHOP, "invalidation.0.bars", 70),
]

REFUSED: list[tuple[Label, str, Any]] = [
    (Label.CALM_UP, "params.fast", 13),
    (Label.CALM_UP, "params.fast", 71),
    (Label.CALM_UP, "params.slow", 69),
    (Label.CALM_UP, "params.slow", 351),
    (Label.CALM_UP, "params.atr_bars", 13),
    (Label.CALM_UP, "params.atr_bars", 71),
    (Label.CALM_UP, "stop_loss", below(2.0)),
    (Label.CALM_UP, "stop_loss", above(8.0)),
    (Label.CALM_UP, "take_profit", below(2.0)),
    (Label.CALM_UP, "take_profit", above(10.0)),
    (Label.CALM_UP, "max_size", -5e-324),
    (Label.CALM_UP, "max_size", above(1.0)),
    (Label.CHOP, "params.n", 13),
    (Label.CHOP, "params.n", 71),
    (Label.CHOP, "params.entry_z", below(0.5)),
    (Label.CHOP, "params.entry_z", above(3.0)),
    (Label.CHOP, "params.exit_z", below(-1.0)),
    (Label.CHOP, "params.exit_z", above(1.0)),
    (Label.CHOP, "params.exit_z", 1.5),  # equal to entry_z: exit_z must be below it
    (Label.CHOP, "params.atr_bars", 13),
    (Label.CHOP, "params.atr_bars", 71),
    (Label.CHOP, "stop_loss", below(2.0)),
    (Label.CHOP, "stop_loss", above(8.0)),
    (Label.CHOP, "invalidation.1.threshold", 0.0),
    (Label.CHOP, "invalidation.1.threshold", 1.0),
    (Label.CHOP, "invalidation.0.bars", 6),
    (Label.CHOP, "invalidation.0.bars", 71),
]


def _case_id(case: tuple[Label, str, Any]) -> str:
    return f"{case[0].value}-{case[1]}={case[2]!r}"


@pytest.mark.spec("SPEC§7", "PLAN§5")
@pytest.mark.parametrize("case", ACCEPTED, ids=_case_id)
def test_a_value_at_its_bound_is_accepted(case: tuple[Label, str, Any]) -> None:
    label, path, value = case
    playbook = parse(changed(label, path, value))
    node: Any = playbook
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    assert node == value


@pytest.mark.spec("SPEC§7", "PLAN§5")
@pytest.mark.parametrize("case", REFUSED, ids=_case_id)
def test_a_value_just_past_its_bound_is_refused(case: tuple[Label, str, Any]) -> None:
    label, path, value = case
    with pytest.raises(ConfigError):
        parse(changed(label, path, value))


@pytest.mark.spec("SPEC§7", "PLAN§5")
def test_trend_fast_must_be_below_slow() -> None:
    front = base(Label.CALM_UP)
    front["params"].update(fast=70, slow=70)
    with pytest.raises(ConfigError):
        parse(front)
    front["params"].update(fast=69, slow=70)
    assert parse(front).params.fast == 69


@pytest.mark.spec("SPEC§7", "PLAN§5")
def test_mean_reversion_exit_z_must_be_below_entry_z() -> None:
    front = base(Label.CHOP)
    front["params"].update(entry_z=0.75, exit_z=0.75)
    with pytest.raises(ConfigError):
        parse(front)
    front["params"].update(exit_z=below(0.75))
    assert parse(front).params.exit_z == below(0.75)


@pytest.mark.spec("SPEC§7", "PLAN§5")
@pytest.mark.parametrize(
    ("condition", "accepted"),
    [
        ({"kind": "close_below_ema", "span": 14}, True),
        ({"kind": "close_below_ema", "span": 350}, True),
        ({"kind": "close_below_ema", "span": 13}, False),
        ({"kind": "close_below_ema", "span": 351}, False),
        ({"kind": "trade_drawdown_exceeds", "fraction": 5e-324}, True),
        ({"kind": "trade_drawdown_exceeds", "fraction": below(1.0)}, True),
        ({"kind": "trade_drawdown_exceeds", "fraction": 0.0}, False),
        ({"kind": "trade_drawdown_exceeds", "fraction": 1.0}, False),
        ({"kind": "label_prob_below"}, False),
        ({"kind": "close_below_ema", "span": 50, "threshold": 0.3}, False),
        ({"kind": "close_above_ema", "span": 50}, False),  # not on the menu
        ({"span": 50}, False),  # no kind
    ],
)
def test_the_invalidation_menu_is_fixed(condition: dict[str, Any], accepted: bool) -> None:
    front = base(Label.CALM_UP)
    front["invalidation"].append(condition)
    if accepted:
        assert parse(front).invalidation[-1].kind == condition["kind"]
    else:
        with pytest.raises(ConfigError):
            parse(front)


@pytest.mark.spec("SPEC§7")
def test_an_empty_invalidation_list_is_allowed_for_a_trading_playbook() -> None:
    front = base(Label.CALM_UP)
    front["invalidation"] = []
    assert parse(front).invalidation == ()


# --- types: no coercion ---------------------------------------------------------------------


@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("params.fast", "35"),
        ("params.fast", 35.0),
        ("params.fast", True),
        ("stop_loss", "4.0"),
        ("stop_loss", math.nan),
        ("stop_loss", math.inf),
        ("max_size", None),
        ("max_size", "1.0"),
        ("version", 0),
        ("version", True),
        ("version", "1"),
        ("label", "calm_up"),
        ("label", "BULL"),
        ("strategy", "TREND_FOLLOWING"),
        ("created_at", datetime(2026, 10, 19)),  # naive  # noqa: DTZ001
        ("created_at", date(2026, 10, 19)),
        ("created_at", "yesterday"),
        ("invalidation", {"kind": "label_prob_below", "threshold": 0.3}),  # not a list
    ],
    ids=lambda v: repr(v),
)
def test_a_value_of_the_wrong_type_is_refused(path: str, value: Any) -> None:
    with pytest.raises(ConfigError):
        parse(changed(Label.CALM_UP, path, value))


@pytest.mark.spec("SPEC§7")
def test_created_at_with_an_offset_is_stored_in_utc() -> None:
    plus_two = datetime(2026, 10, 19, 2, tzinfo=timezone(timedelta(hours=2)))
    playbook = parse(changed(Label.CALM_UP, "created_at", plus_two))
    assert playbook.created_at == datetime(2026, 10, 19, tzinfo=UTC)
    assert playbook.created_at.utcoffset() == timedelta(0)


# --- strategies and label rules -------------------------------------------------------------


@pytest.mark.spec("SPEC§7", "D9")
@pytest.mark.parametrize("strategy", ["short_trend", "pairs", "momentum", ""])
def test_only_the_three_primitives_exist(strategy: str) -> None:
    # Long or flat only (D9): no other strategy, short or otherwise, can be named.
    with pytest.raises(ConfigError):
        parse(changed(Label.CALM_UP, "strategy", strategy))


@pytest.mark.spec("SPEC§7", "PLAN§5")
def test_params_must_match_the_strategy() -> None:
    # trend params under mean_reversion, and the reverse, are unknown keys of the other.
    with pytest.raises(ConfigError):
        parse(changed(Label.CALM_UP, "strategy", "mean_reversion"))
    with pytest.raises(ConfigError):
        parse(changed(Label.CHOP, "strategy", "trend_following"))


@pytest.mark.spec("SPEC§7", "D8", "PLAN§5")
@pytest.mark.parametrize("source", [Label.CALM_UP, Label.CHOP])
def test_a_crash_playbook_must_be_flat(source: Label) -> None:
    # CRASH's cap is 0 in code; the schema refuses any strategy but flat for it.
    with pytest.raises(ConfigError):
        parse(changed(source, "label", "CRASH"))
    assert parse(base(Label.CRASH)).strategy == "flat"


@pytest.mark.spec("SPEC§7", "D8", "PLAN§5")
@pytest.mark.parametrize("source", [Label.CALM_UP, Label.CHOP])
def test_stress_may_trade_at_most_a_quarter(source: Label) -> None:
    front = changed(source, "label", "STRESS")
    front["max_size"] = 0.25
    assert parse(front).max_size == 0.25
    for too_big in (above(0.25), 0.26, 0.5, 1.0):
        front["max_size"] = too_big
        with pytest.raises(ConfigError):
            parse(front)


@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize("label", [Label.CALM_UP, Label.CHOP])
def test_max_size_above_a_labels_cap_is_not_refused(label: Label) -> None:
    # Spec §7: max_size is clamped to the label's hard cap when used, never rejected for it.
    # CHOP's cap is 0.5; a CHOP playbook may still say 1.0.
    assert parse(changed(label, "max_size", 1.0)).max_size == 1.0


@pytest.mark.spec("SPEC§7", "D8")
@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("stop_loss", 4.0),
        ("take_profit", 3.0),
        ("max_size", 0.1),
        ("invalidation", [{"kind": "label_prob_below", "threshold": 0.3}]),
        ("params", {"fast": 35}),
    ],
)
@pytest.mark.parametrize("label", [Label.STRESS, Label.CRASH])
def test_a_flat_playbook_holds_nothing(label: Label, path: str, value: Any) -> None:
    with pytest.raises(ConfigError):
        parse(changed(label, path, value))


@pytest.mark.spec("SPEC§7")
def test_a_playbook_is_frozen() -> None:
    from pydantic import ValidationError

    playbook = parse(base(Label.CALM_UP))
    with pytest.raises(ValidationError):
        playbook.max_size = 0.5
    with pytest.raises(ValidationError):
        playbook.params.fast = 20


# --- front matter ---------------------------------------------------------------------------


@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize(
    "text",
    [
        "label: CALM_UP\n",  # no front matter at all
        "\n---\nlabel: CALM_UP\n---\n",  # a blank line before it
        "---\nlabel: CALM_UP\n",  # never closed
        "---\n---\nbody\n",  # empty
        "---\n- a\n- b\n---\n",  # a list
        "---\njust text\n---\n",  # a scalar
        "---\nlabel: [CALM_UP\n---\n",  # not YAML
        "--- \nlabel: CALM_UP\n---\n",  # the opening line isn't exactly ---
    ],
)
def test_front_matter_must_be_a_mapping_between_two_dashed_lines(text: str) -> None:
    from quant.decide.playbook import parse_front_matter

    with pytest.raises(ConfigError):
        parse_front_matter(text)


@pytest.mark.spec("SPEC§7")
def test_front_matter_is_the_yaml_between_the_first_two_dashed_lines() -> None:
    from quant.decide.playbook import parse_front_matter

    text = "---\na: 1\nb: [x, y]\n---\n# heading\n---\nc: 3\n---\n"
    assert parse_front_matter(text) == {"a": 1, "b": ["x", "y"]}
