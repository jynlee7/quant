"""B8a review, round 1: a red test for each confirmed finding (A-H from the two reviews).

A and F share one contract, which the green builds into the lifecycle: `primitives.step` takes
`label_prob` (the filtered P(the playbook's label) on this bar; None, the default, means no
invalidation, as the shadow edge book runs it). With it given:

- **In a trade (A):** after bars in trade and the peak close are updated for this bar, the
  playbook's invalidation conditions are checked against the trade as it stands on this bar,
  before the exit signal. A hit closes the trade (`close_trade`) with reason
  `Reason.INVALIDATED`, and `Step.invalidation` names the hit; decide then blocks the playbook
  (`switching.invalidate`). So an invalidation on a bar whose signal also exits is still an
  invalidation, and the block applies.
- **At entry (F):** an entry is refused while an invalidation condition that can hold at entry
  (`label_prob_below`, `close_below_ema`) holds on that bar: no position, no block, the state
  still flat and armed, so a later bar can enter.

B, rules 4 and 5 compare exact decimal values whatever the ambient `decimal` context. C, an
invalid YAML timestamp is a ConfigError, not a ValueError. D, a repeated YAML key, at any level,
is a ConfigError in every L3 file (playbooks, the manifest, engine.yaml, risk.yaml). E, a bar
that overflows float64 in the indicators is refused with FeatureError (rule 6), the state as it
was. G, a restored health streak above `recovery_bars` is clamped, not an error. H,
IndicatorConfig itself refuses bad EMA spans, with ConfigError as the stream did.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import localcontext
from pathlib import Path
from typing import Any

import pytest

from quant.core.canonical import canonical_dumps
from quant.core.errors import ConfigError, FeatureError
from quant.core.regime import Label
from tests.support.decide import (
    CAPS,
    ENGINE_YAML,
    PLAYBOOKS,
    STRATEGY,
    SWITCHING,
    bar,
    copy_release,
    priced,
    probs,
    top,
)
from tests.support.risk import RISK_YAML
from tests.unit.test_decide_primitives import in_trade, playbook, values

RED = "red:B8"
CALM, CHOP, STRESS, CRASH = Label.CALM_UP, Label.CHOP, Label.STRESS, Label.CRASH


# --- A: an exit on the same bar hides an invalidation -------------------------------------------


@pytest.mark.spec("SPEC§7", "SPEC§8")
@pytest.mark.xfail(strict=True, reason=RED, raises=TypeError)
def test_an_invalidation_on_an_exit_bar_is_an_invalidation_and_blocks() -> None:
    from quant.decide.primitives import Reason, TradeState, step
    from quant.decide.switching import SwitchState, invalidate
    from quant.decide.switching import step as switch_step

    chop = playbook(CHOP)  # bars_in_trade_exceeds 21, then label_prob_below 0.30
    # Trade bar 22 (21 bars before it): z = 0.1 is also the exit condition (z >= 0).
    out = step(
        chop, in_trade(98.0, 92.0, bars=21), values(1, z=0.1), priced(1, 100.0), label_prob=0.8
    )
    assert out.reason == Reason.INVALIDATED
    assert out.invalidation is not None
    assert out.invalidation.condition.kind == "bars_in_trade_exceeds"
    assert out.state == TradeState.flat()

    # decide blocks the playbook; z = -1.6 doesn't enter until a fresh 3-bar re-confirmation.
    sw = invalidate(
        SwitchState(
            active=CHOP,
            challenger=None,
            count=0,
            since_switch=10,
            health_streak=SWITCHING.health_recovery_bars,
            blocked=False,
            reconfirm=0,
        )
    )
    trade = out.state
    reasons = []
    for i in (2, 3, 4):
        so = switch_step(sw, top(CHOP), cause=None, config=SWITCHING, caps=CAPS)
        got = step(
            chop,
            trade,
            values(i, z=-1.6),
            priced(i, 98.0),
            entries_allowed=so.entries_allowed,
            label_prob=0.8,
        )
        reasons.append(got.reason)
        sw, trade = so.state, got.state
    assert reasons == [Reason.BLOCKED, Reason.BLOCKED, Reason.ENTER]


@pytest.mark.spec("SPEC§7")
@pytest.mark.xfail(strict=True, reason=RED, raises=TypeError)
def test_invalidation_counts_this_bar_on_every_trade_bar() -> None:
    from quant.decide.primitives import Reason, step

    chop = playbook(CHOP)
    # Trade bar 21 holds; trade bar 22 (no exit signal) is invalidated.
    held = step(
        chop, in_trade(98.0, 92.0, bars=20), values(1, z=-0.4), priced(1, 98.5), label_prob=0.8
    )
    assert held.reason == Reason.HOLD
    assert held.state.bars_in_trade == 21
    assert held.invalidation is None
    hit = step(chop, held.state, values(2, z=-0.4), priced(2, 98.5), label_prob=0.8)
    assert hit.reason == Reason.INVALIDATED
    assert not hit.in_position
    # Without label_prob (the shadow book) invalidation doesn't apply: the exit is an exit.
    shadow = step(
        chop, in_trade(98.0, 92.0, bars=21), values(1, z=0.1), priced(1, 100.0), label_prob=None
    )
    assert shadow.reason == Reason.EXIT


# --- F: an entry and an invalidation on the same bar ---------------------------------------------


@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize(
    ("invalidation", "label_prob", "ema"),
    [
        # CALM_UP active by hysteresis at P(CALM_UP) = 0.2: label_prob_below 0.30 holds.
        (None, 0.2, {35: 101.0, 175: 100.0}),
        # close 103 > EMA(35) 101 (the trend entry) but < EMA(50) 104.
        ([{"kind": "close_below_ema", "span": 50}], 0.9, {35: 101.0, 50: 104.0, 175: 100.0}),
    ],
)
@pytest.mark.xfail(strict=True, reason=RED, raises=TypeError)
def test_an_entry_is_refused_while_an_invalidation_holds(
    invalidation: list[dict[str, Any]] | None, label_prob: float, ema: dict[int, float]
) -> None:
    from quant.decide.primitives import Reason, TradeState, step

    calm = playbook(CALM) if invalidation is None else playbook(CALM, invalidation=invalidation)
    refused = step(
        calm, TradeState.flat(), values(1, ema=ema), priced(1, 103.0), label_prob=label_prob
    )
    assert not refused.in_position
    assert refused.reason not in (Reason.ENTER, Reason.INVALIDATED)
    assert refused.state == TradeState.flat()  # no block: still armed
    # The next bar, the condition no longer holding, enters.
    calm_ema = {35: 101.0, 50: 100.0, 175: 100.0}
    entered = step(calm, refused.state, values(2, ema=calm_ema), priced(2, 103.0), label_prob=0.9)
    assert entered.reason == Reason.ENTER


# --- B: rules 4 and 5 under the ambient decimal context ---------------------------------------


def _switch(label_probs: dict[Label, float], next_probs: dict[Label, float], prec: int) -> Any:
    from quant.decide.switching import SwitchState, step

    with localcontext() as ctx:
        ctx.prec = prec
        return step(
            SwitchState.cold(SWITCHING),
            bar(label_probs, next_probs),
            cause=None,
            config=SWITCHING,
            caps=CAPS,
        )


@pytest.mark.spec("SPEC§8")
@pytest.mark.xfail(strict=True, reason=RED, raises=AssertionError)
def test_rule_4_sums_exactly_at_any_ambient_precision() -> None:
    from quant.decide.switching import RuleId

    # P(STRESS) + P(CRASH) = 0.25 + 1e-30 > 0.25 exactly; 28 digits round the sum to 0.25.
    nxt = {CALM: 0.75, STRESS: 0.25, CRASH: 1e-30}
    for prec in (60, 28):
        out = _switch(probs(CALM_UP=0.8), nxt, prec)
        assert out.fired(RuleId.R4), prec
        assert out.m_rule4 == 0.5


@pytest.mark.spec("SPEC§8")
@pytest.mark.xfail(strict=True, reason=RED, raises=AssertionError)
def test_rules_4_and_5_do_not_depend_on_the_ambient_context() -> None:
    from quant.decide.switching import RuleId

    # Top two 0.576 and 0.424: a gap of 0.152 > 0.15, so rule 5 doesn't fire. Next-bar
    # P(STRESS) + P(CRASH) = 0.251 > 0.25, so rule 4 does. Two digits would read 0.15 and 0.25.
    now = {CALM: 0.576, CHOP: 0.424, STRESS: 0.0, CRASH: 0.0}
    nxt = {CALM: 0.749, STRESS: 0.126, CRASH: 0.125}
    for prec in (28, 2):
        out = _switch(now, nxt, prec)
        assert not out.fired(RuleId.R5), prec
        assert out.m_rule5 == 1.0
        assert out.fired(RuleId.R4), prec


# --- C: an invalid YAML timestamp ----------------------------------------------------------------

BAD_TIME = "2026-13-19T00:00:00Z"


@pytest.mark.spec("SPEC§7", "ARCH§9.2")
@pytest.mark.xfail(strict=True, reason=RED, raises=ValueError)
def test_an_invalid_timestamp_in_a_playbook_is_a_config_error() -> None:
    from quant.decide.playbook import parse_playbook

    text = PLAYBOOKS[CHOP].read_text(encoding="utf-8").replace("2026-10-19T00:00:00Z", BAD_TIME)
    with pytest.raises(ConfigError):
        parse_playbook(text)


@pytest.mark.spec("ARCH§9.2", "P7")
@pytest.mark.xfail(strict=True, reason=RED, raises=ValueError)
def test_an_invalid_timestamp_in_the_manifest_is_a_config_error() -> None:
    from quant.decide.playbook import parse_manifest

    text = STRATEGY.read_text(encoding="utf-8").replace("2026-10-19T00:00:00Z", BAD_TIME)
    with pytest.raises(ConfigError):
        parse_manifest(text)


@pytest.mark.spec("ARCH§9.2", "SPEC§8")
@pytest.mark.xfail(strict=True, reason=RED, raises=ValueError)
def test_an_invalid_timestamp_in_a_release_gives_cause_release(tmp_path: Path) -> None:
    from quant.decide.health import Cause, cause_of
    from quant.decide.playbook import load_release

    root = copy_release(tmp_path / "r1")
    path = root / "playbooks" / "CALM_UP.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("2026-10-19T00:00:00Z", BAD_TIME),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as info:
        load_release(root)
    assert cause_of(info.value) == Cause.RELEASE


@pytest.mark.spec("ARCH§9.2")
@pytest.mark.xfail(strict=True, reason=RED, raises=ValueError)
def test_an_invalid_timestamp_in_engine_yaml_is_a_config_error() -> None:
    from quant.decide.config import parse_engine_config

    data = ENGINE_YAML.read_bytes().replace(
        b"feed_delay_min: 15", b"feed_delay_min: " + BAD_TIME.encode()
    )
    with pytest.raises(ConfigError):
        parse_engine_config(data)


@pytest.mark.spec("SPEC§10", "D13")
@pytest.mark.xfail(strict=True, reason=RED, raises=ValueError)
def test_an_invalid_timestamp_in_risk_yaml_is_a_config_error() -> None:
    from quant.risk.limits import parse_risk_config

    data = RISK_YAML.read_bytes().replace(
        b"uncertain_cap: 0.0", b"uncertain_cap: " + BAD_TIME.encode()
    )
    with pytest.raises(ConfigError):
        parse_risk_config(data)


# --- D: repeated YAML keys -------------------------------------------------------------------


def _edit(text: str, old: str, new: str) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


@pytest.mark.spec("SPEC§7", "ARCH§9.2")
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("max_size: 0.5", "max_size: 0.5\nmax_size: 0.25"),  # top level
        ("  entry_z: 1.5", "  entry_z: 1.5\n  entry_z: 2.0"),  # in params
        (
            "{kind: bars_in_trade_exceeds, bars: 21}",
            "{kind: bars_in_trade_exceeds, bars: 21, bars: 30}",
        ),
    ],
)
@pytest.mark.xfail(strict=True, reason=RED, raises=pytest.fail.Exception)
def test_a_repeated_key_in_a_playbook_is_a_config_error(old: str, new: str) -> None:
    from quant.decide.playbook import parse_playbook

    text = _edit(PLAYBOOKS[CHOP].read_text(encoding="utf-8"), old, new)
    with pytest.raises(ConfigError):
        parse_playbook(text)


@pytest.mark.spec("ARCH§9.2", "P7")
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("gate_report: null", "gate_report: null\ngate_report: reports/gate.md"),
        ("  tol: 0.01", "  tol: 0.01\n  tol: 0.5"),
        ("  CHOP: playbooks/CHOP.md", "  CHOP: playbooks/CHOP.md\n  CHOP: playbooks/CHOP.md"),
    ],
)
@pytest.mark.xfail(strict=True, reason=RED, raises=pytest.fail.Exception)
def test_a_repeated_key_in_the_manifest_is_a_config_error(old: str, new: str) -> None:
    from quant.decide.playbook import parse_manifest

    text = _edit(STRATEGY.read_text(encoding="utf-8"), old, new)
    with pytest.raises(ConfigError):
        parse_manifest(text)


@pytest.mark.spec("ARCH§9.2")
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("  cooldown_bars: 7 ", "  cooldown_bars: 9\n  cooldown_bars: 7 "),  # nested
        ("timing:\n", "timing:\n  feed_delay_min: 15\n  decision_delay_min: 16\ntiming:\n"),
    ],
)
@pytest.mark.xfail(strict=True, reason=RED, raises=pytest.fail.Exception)
def test_a_repeated_key_in_engine_yaml_is_a_config_error(old: str, new: str) -> None:
    from quant.decide.config import parse_engine_config

    text = _edit(ENGINE_YAML.read_text(encoding="utf-8"), old, new)
    with pytest.raises(ConfigError):
        parse_engine_config(text.encode())


@pytest.mark.spec("SPEC§10", "D13")
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("overnight_cap: 0.5 ", "overnight_cap: 0.25\novernight_cap: 0.5 "),  # top level
        ("  CHOP: 0.5", "  CHOP: 0.5\n  CHOP: 0.25"),  # in label_caps
        ("  poll_s: 1.0 ", "  poll_s: 2.0\n  poll_s: 1.0 "),  # in kill
    ],
)
@pytest.mark.xfail(strict=True, reason=RED, raises=pytest.fail.Exception)
def test_a_repeated_key_in_risk_yaml_is_a_config_error(old: str, new: str) -> None:
    from quant.risk.limits import parse_risk_config

    text = _edit(RISK_YAML.read_text(encoding="utf-8"), old, new)
    with pytest.raises(ConfigError):
        parse_risk_config(text.encode())


# --- E: float64 overflow in the indicators ---------------------------------------------------


def _huge_int(i: int) -> Any:
    big = 10**400  # an int no float64 can hold
    return replace(priced(i, 100.0), open=big, high=big, low=big, close=big)


@pytest.mark.spec("SPEC§7", "SPEC§8")
@pytest.mark.parametrize(
    ("first", "second"),
    [
        # Window 2: (2e200 - 1.5e200)^2 overflows the variance.
        (lambda: priced(0, 1e200), lambda: priced(1, 2e200, low=1e200)),
        # ATR 2: two true ranges of 1.7e308 overflow their sum.
        (lambda: priced(0, 1.0, high=1.7e308), lambda: priced(1, 1.0, high=1.7e308)),
        # An int price too big for a float.
        (lambda: priced(0, 100.0), lambda: _huge_int(1)),
    ],
    ids=["variance", "atr-sum", "int-price"],
)
@pytest.mark.xfail(strict=True, reason=RED, raises=OverflowError)
def test_an_overflowing_bar_is_refused_as_unhealthy(first: Any, second: Any) -> None:
    from quant.decide.health import Cause, cause_of
    from quant.decide.indicators import IndicatorConfig, IndicatorStream

    stream = IndicatorStream(IndicatorConfig(ema_spans=(14,), atr_bars=2, window=2))
    stream.update(first())
    before = canonical_dumps(stream.snapshot())
    with pytest.raises(FeatureError) as info:
        stream.update(second())
    assert canonical_dumps(stream.snapshot()) == before
    assert cause_of(info.value) == Cause.FEATURES


# --- G: a restored health streak above recovery_bars ---------------------------------------------


@pytest.mark.spec("SPEC§8", "C2")
@pytest.mark.xfail(strict=True, reason=RED, raises=ValueError)
def test_a_streak_above_recovery_bars_is_clamped() -> None:
    from quant.decide.health import Cause, Health, health_step

    assert health_step(2, None, recovery_bars=1) == Health(None, 1, True)
    assert health_step(3, None, recovery_bars=2) == Health(None, 2, True)
    assert health_step(2, Cause.MODEL, recovery_bars=1) == Health(Cause.MODEL, 0, False)


@pytest.mark.spec("SPEC§8", "C2")
@pytest.mark.xfail(strict=True, reason=RED, raises=ValueError)
def test_a_state_restored_under_a_shorter_recovery_keeps_switching() -> None:
    from quant.decide.config import SwitchingConfig
    from quant.decide.switching import SwitchState, step

    # engine.yaml's recovery went from 2 to 1 by a reviewed commit; the saved streak is 2.
    shorter = SwitchingConfig.model_validate({**SWITCHING.model_dump(), "health_recovery_bars": 1})
    saved = SwitchState.cold(SWITCHING)
    out = step(saved, top(CALM), cause=None, config=shorter, caps=CAPS)
    assert out.health.operating
    assert out.state.health_streak == 1


# --- H: IndicatorConfig accepts bad EMA spans ------------------------------------------------


@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize("spans", [(175, 35), (35, 35), (0,), (-5, 35)])
@pytest.mark.xfail(strict=True, reason=RED, raises=pytest.fail.Exception)
def test_indicator_config_itself_refuses_bad_spans(spans: tuple[int, ...]) -> None:
    from quant.decide.indicators import IndicatorConfig

    with pytest.raises(ConfigError):
        IndicatorConfig(ema_spans=spans, atr_bars=35, window=None)
