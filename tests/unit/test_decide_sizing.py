"""B8b: sizing (spec §9, C12, C13, design call 9; quant.decide.sizing).

The formula with each factor binding in turn, the calibration floor (none: 0.25 of the cap and
the raw P; a real Calibration's `cap_fraction` and `calibrated`, built by
tests.support.calibration), the zeros, the drift freeze and the daily-loss lock that never
increase, the deadband at exactly 0.10 on decimal values, and the forced reductions and exits
to 0 it lets through.
"""

from __future__ import annotations

import pytest

from quant.core.regime import Label
from tests.support.calibration import sizing_calibration
from tests.support.decide import engine_config, r1_caps, r1_playbook


def _size(label: Label, **overrides: object) -> object:
    from quant.decide.sizing import size

    fields: dict[str, object] = {
        "playbook": r1_playbook(label),
        "cap": r1_caps().label_caps[label],
        "calibration": None,
        "f_star": 100.0,
        "label_prob": 0.9,
        "entropy": 0.2,
        "m_rule4": 1.0,
        "m_rule5": 1.0,
        "config": engine_config().sizing,
    }
    fields.update(overrides)
    return size(label, **fields)  # type: ignore[arg-type]


# --- the formula -----------------------------------------------------------------------------


@pytest.mark.spec("SPEC§9", "C13")
def test_without_a_calibration_every_label_gets_a_quarter_of_its_cap_and_the_raw_p() -> None:
    from quant.decide.sizing import NO_CALIBRATION_CAP

    assert NO_CALIBRATION_CAP == 0.25
    s = _size(Label.CALM_UP)
    assert s.cap == 1.0
    assert s.cap_fraction == 0.25
    assert s.max_size == 1.0  # type: ignore[attr-defined]
    assert s.kelly == 0.25 * 100.0
    assert s.f_star == 100.0  # type: ignore[attr-defined]
    assert s.base == min(1.0 * 0.25, 1.0, 25.0) == 0.25  # type: ignore[attr-defined]
    assert s.p_raw == s.p_calibrated == 0.9  # type: ignore[attr-defined]
    assert s.value == 0.25 * 0.9 * (1 - 0.2) * 1.0 * 1.0  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9")
def test_each_factor_of_the_formula_scales_the_target() -> None:
    base = _size(Label.CALM_UP).value  # type: ignore[attr-defined]
    assert _size(Label.CALM_UP, label_prob=0.45).value == 0.25 * 0.45 * 0.8  # type: ignore[attr-defined]
    assert _size(Label.CALM_UP, entropy=0.6).value == 0.25 * 0.9 * (1 - 0.6)  # type: ignore[attr-defined]
    assert _size(Label.CALM_UP, m_rule4=0.5).value == base * 0.5  # type: ignore[attr-defined]
    assert _size(Label.CALM_UP, m_rule5=0.0).value == 0.0  # type: ignore[attr-defined]
    assert _size(Label.CALM_UP, entropy=1.0).value == 0.0  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9", "SPEC§7", "C12")
def test_the_smallest_of_cap_max_size_and_kelly_binds() -> None:
    passing = sizing_calibration()
    # The cap binds: CHOP's hard cap 0.5 at the failed fraction 0.25.
    chop = _size(Label.CHOP)
    assert chop.base == 0.5 * 0.25  # type: ignore[attr-defined]
    # max_size binds: CHOP passes calibration (fraction 1.0), so 0.5 x 1.0 meets max_size 0.5;
    # a playbook's max_size above the cap is clamped by the min, never refused (spec §7).
    passed = _size(Label.CHOP, calibration=passing)
    assert passed.cap_fraction == 1.0
    assert passed.base == 0.5  # type: ignore[attr-defined]
    # Kelly binds: 0.25 x f* = 0.1.
    kelly = _size(Label.CALM_UP, f_star=0.4)
    assert kelly.kelly == 0.25 * 0.4
    assert kelly.base == 0.25 * 0.4  # type: ignore[attr-defined]
    # No edge, no size (C12).
    assert _size(Label.CALM_UP, f_star=0.0).value == 0.0  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9", "A4", "C13", "ARCH§7.4")
def test_a_real_calibration_supplies_the_cap_fraction_and_p_tilde() -> None:
    passing = sizing_calibration()
    failed = sizing_calibration(outcome_passes=False)
    assert passing.cap_fraction(Label.CALM_UP) == 1.0
    assert passing.calibrated(Label.CALM_UP, 0.9) != 0.9  # its knots move P
    s = _size(Label.CALM_UP, calibration=passing)
    assert s.cap_fraction == passing.cap_fraction(Label.CALM_UP)  # type: ignore[attr-defined]
    assert s.p_raw == 0.9  # type: ignore[attr-defined]
    assert s.p_calibrated == passing.calibrated(Label.CALM_UP, 0.9)  # type: ignore[attr-defined]
    expected = min(1.0, 1.0, 25.0) * passing.calibrated(Label.CALM_UP, 0.9) * (1 - 0.2)
    assert s.value == expected  # type: ignore[attr-defined]
    # A failed outcome check caps every label at the failed cap; the knots still map P.
    f = _size(Label.CALM_UP, calibration=failed)
    assert f.cap_fraction == failed.cap_fraction(Label.CALM_UP) == 0.25  # type: ignore[attr-defined]
    assert f.p_calibrated == failed.calibrated(Label.CALM_UP, 0.9)  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9", "SPEC§7", "SPEC§10")
def test_crash_and_uncertain_size_zero() -> None:
    crash = _size(Label.CRASH, calibration=sizing_calibration())
    assert crash.cap == 0.0
    assert crash.value == 0.0  # type: ignore[attr-defined]
    assert _size(Label.CALM_UP, m_rule5=0.0, calibration=sizing_calibration()).value == 0.0  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9")
def test_size_refuses_inputs_outside_their_ranges() -> None:
    for bad in (
        {"playbook": r1_playbook(Label.CHOP)},
        {"label_prob": 1.5},
        {"entropy": -0.1},
        {"f_star": -1.0},
        {"cap": 1.5},
    ):
        with pytest.raises(ValueError, match=r"playbook|cap|label_prob|entropy|f_star"):
            _size(Label.CALM_UP, **bad)


# --- the zeros -------------------------------------------------------------------------------


def _target(**overrides: object) -> object:
    from quant.decide.sizing import target

    fields: dict[str, object] = {
        "playbook": r1_playbook(Label.CALM_UP),
        "in_position": True,
        "operating": True,
        "blocked": False,
        "halted": False,
        "caps": r1_caps(),
        "calibration": None,
        "f_star": 100.0,
        "label_prob": 0.9,
        "entropy": 0.2,
        "m_rule4": 1.0,
        "m_rule5": 1.0,
        "config": engine_config().sizing,
    }
    active = overrides.pop("active", Label.CALM_UP)
    fields.update(overrides)
    return target(active, **fields)  # type: ignore[arg-type]


@pytest.mark.spec("SPEC§9", "SPEC§8", "SPEC§10")
def test_the_target_is_zero_halted_unhealthy_inactive_blocked_or_out_of_position() -> None:
    from quant.decide.sizing import ZeroReason

    live = _target()
    assert live.value == 0.25 * 0.9 * 0.8
    assert live.zero is None  # type: ignore[attr-defined]
    assert live.sizing is not None  # type: ignore[attr-defined]
    for overrides, reason in (
        ({"halted": True}, ZeroReason.HALTED),
        ({"operating": False}, ZeroReason.UNHEALTHY),
        ({"active": None, "playbook": None}, ZeroReason.NO_ACTIVE),
        ({"blocked": True}, ZeroReason.BLOCKED),
        ({"in_position": False}, ZeroReason.OUT_OF_POSITION),
        # The order: halted wins over everything; unhealthy over a missing label.
        ({"halted": True, "operating": False, "in_position": False}, ZeroReason.HALTED),
        ({"operating": False, "active": None, "playbook": None}, ZeroReason.UNHEALTHY),
    ):
        t = _target(**overrides)
        assert (t.value, t.zero, t.sizing) == (0.0, reason, None), overrides  # type: ignore[attr-defined]
    # Without probabilities (an unhealthy bar) the formula never runs.
    t = _target(operating=False, label_prob=None, entropy=None)
    assert t.zero is ZeroReason.UNHEALTHY  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="label_prob"):
        _target(label_prob=None)


@pytest.mark.spec("SPEC§9", "SPEC§7")
def test_crash_active_targets_zero_through_its_cap() -> None:
    t = _target(active=Label.CRASH, playbook=r1_playbook(Label.CRASH))
    assert t.value == 0.0  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9")
def test_caps_name_every_label_and_none_is_zero() -> None:
    caps = r1_caps()
    assert [caps.cap(label) for label in Label] == [0.0, 0.25, 1.0, 0.5]
    assert caps.cap(None) == 0.0
    assert caps.caps_hash() == r1_caps().caps_hash()
    assert len(caps.caps_hash()) == 64
    other = caps.model_copy(update={"overnight_cap": 0.4})
    assert other.caps_hash() != caps.caps_hash()


# --- rebalancing: freeze, lock, deadband, forced ----------------------------------------------


def _rebalance(target: float, current: float, **overrides: object) -> object:
    from quant.decide.sizing import rebalance

    fields: dict[str, object] = {
        "ceiling": 1.0,
        "deadband": 0.10,
        "rule_forced": False,
        "halted": False,
        "frozen": False,
        "locked": False,
        "reduce_only": False,
    }
    fields.update(overrides)
    return rebalance(target, current, **fields)  # type: ignore[arg-type]


@pytest.mark.spec("SPEC§9")
def test_the_deadband_is_exactly_a_tenth_on_decimal_values() -> None:
    from quant.decide.sizing import exceeds_deadband

    assert 0.35 - 0.25 < 0.10  # floats would hold this resize back
    assert exceeds_deadband(0.35, 0.25, 0.10)
    assert exceeds_deadband(0.15, 0.25, 0.10)  # a reduction of exactly 0.10
    assert not exceeds_deadband(0.34, 0.25, 0.10)
    assert not exceeds_deadband(0.16, 0.25, 0.10)
    up = _rebalance(0.35, 0.25)
    assert (up.desired, up.move, up.held, up.forced) == (0.35, True, False, False)  # type: ignore[attr-defined]
    held = _rebalance(0.34, 0.25)
    assert (held.desired, held.move, held.held) == (0.25, False, True)  # type: ignore[attr-defined]
    down = _rebalance(0.16, 0.25)
    assert (down.desired, down.move, down.held) == (0.25, False, True)  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9", "SPEC§8", "SPEC§10")
def test_forced_reductions_ignore_the_deadband() -> None:
    # Rules 4-6 (rule_forced), a halt, and a ceiling below the position all force the cut.
    for overrides in ({"rule_forced": True}, {"halted": True}):
        r = _rebalance(0.20, 0.25, **overrides)
        assert (r.desired, r.move, r.forced, r.held) == (0.20, True, True, False), overrides  # type: ignore[attr-defined]
    capped = _rebalance(0.60, 0.55, ceiling=0.5)  # the overnight cap under the position
    assert (capped.desired, capped.move, capped.forced, capped.ceiling) == (0.5, True, True, 0.5)  # type: ignore[attr-defined]
    # A rule firing doesn't force an increase, nor a cut the target doesn't ask for.
    up = _rebalance(0.30, 0.25, rule_forced=True)
    assert (up.desired, up.move, up.forced) == (0.25, False, False)  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9", "SPEC§7")
def test_an_exit_to_zero_ignores_the_deadband() -> None:
    r = _rebalance(0.0, 0.05)
    assert (r.desired, r.move, r.forced, r.held) == (0.0, True, False, False)  # type: ignore[attr-defined]
    flat = _rebalance(0.0, 0.0)
    assert (flat.desired, flat.move) == (0.0, False)  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9", "C5")
def test_the_freeze_and_the_loss_lock_never_increase_but_let_reductions_through() -> None:
    for flag in ("frozen", "locked"):
        up = _rebalance(0.60, 0.20, **{flag: True})
        assert (up.desired, up.move, up.limited) == (0.20, False, True), flag  # type: ignore[attr-defined]
        entry = _rebalance(0.60, 0.0, **{flag: True})
        assert (entry.desired, entry.move) == (0.0, False), flag  # type: ignore[attr-defined]
        down = _rebalance(0.0, 0.40, **{flag: True})
        assert (down.desired, down.move) == (0.0, True), flag  # type: ignore[attr-defined]
        cut = _rebalance(0.25, 0.40, **{flag: True})
        assert (cut.desired, cut.move, cut.limited) == (0.25, True, False), flag  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9", "ARCH§5", "A3")
def test_the_closing_bar_decision_only_reduces() -> None:
    up = _rebalance(0.60, 0.30, reduce_only=True)
    assert (up.desired, up.move, up.limited) == (0.30, False, True)  # type: ignore[attr-defined]
    down = _rebalance(0.10, 0.30, reduce_only=True)
    assert (down.desired, down.move) == (0.10, True)  # type: ignore[attr-defined]


@pytest.mark.spec("SPEC§9", "D9")
def test_rebalance_refuses_negative_or_non_finite_exposures() -> None:
    import math

    for target, current in ((-0.1, 0.0), (0.1, -0.2), (math.nan, 0.0), (0.1, math.inf)):
        with pytest.raises(ValueError, match=r"target|current"):
            _rebalance(target, current)
