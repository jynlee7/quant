"""B8a: switching, rules 1-6 and A1, and invalidation blocks (spec §8, A1, C4, D5).

PLAN B8's scenario tables, one per rule: each row is a sequence of bars (label probabilities, or
an unhealthy bar) and the table states the active label after every bar, the switch bars and
the multipliers. r1's switching values and risk.yaml's caps (CALM_UP 1.0, CHOP 0.5, STRESS 0.25,
CRASH 0) throughout, except where a test names other caps.

Bars are numbered from 0. "CALM_UP x3" means three bars on which CALM_UP is the top label at
P = 0.80, the rest spread evenly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from quant.core.regime import Label
from quant.decide.health import Cause
from quant.decide.switching import BarProbs
from tests.support.decide import CAPS, SWITCHING, bar, probs, top

RED = pytest.mark.xfail(strict=True, reason="red:B8", raises=NotImplementedError)

CALM, CHOP, STRESS, CRASH = Label.CALM_UP, Label.CHOP, Label.STRESS, Label.CRASH
SICK = Cause.FEATURES  # an unhealthy bar

Bar = BarProbs | Cause


def run(
    bars: Sequence[Bar],
    *,
    caps: Mapping[Label, float] = CAPS,
    invalidate_after: Sequence[int] = (),
) -> list[Any]:
    """The outcome of every bar from a cold start; `invalidate_after` blocks the active label
    right after those bars, as decide does after an invalidation hit."""
    from quant.decide.switching import SwitchState, invalidate, step

    state = SwitchState.cold(SWITCHING)
    outcomes = []
    for i, b in enumerate(bars):
        if isinstance(b, Cause):
            out = step(state, None, cause=b, config=SWITCHING, caps=caps)
        else:
            out = step(state, b, cause=None, config=SWITCHING, caps=caps)
        outcomes.append(out)
        state = out.state
        if i in invalidate_after:
            state = invalidate(state)
    return outcomes


def actives(outcomes: Sequence[Any]) -> list[Label | None]:
    return [o.active for o in outcomes]


def switches(outcomes: Sequence[Any]) -> list[int]:
    return [i for i, o in enumerate(outcomes) if o.switched]


def fired(outcomes: Sequence[Any], rule: str) -> list[int]:
    from quant.decide.switching import RuleId

    return [i for i, o in enumerate(outcomes) if o.fired(RuleId(rule))]


def many(label: Label, n: int, p: float = 0.8) -> list[Bar]:
    return [top(label, p) for _ in range(n)]


# --- rules 1-3: takeover, persistence, cooldown ---------------------------------------------


@RED
@pytest.mark.spec("SPEC§8", "D5", "C4")
def test_a_three_bar_takeover_held_back_by_the_cooldown() -> None:
    # CHOP takes over cold at bar 2. CALM_UP (a higher cap: risk-increasing) qualifies from
    # bar 3, has its 3 bars at bar 5, and waits for the cooldown: 7 bars after the switch at 2,
    # so it takes over at bar 9 (design call 3: the count keeps accumulating meanwhile).
    out = run([*many(CHOP, 3), *many(CALM, 8)])
    assert actives(out) == [None, None, CHOP, *[CHOP] * 6, CALM, CALM]
    assert switches(out) == [2, 9]
    assert fired(out, "R3") == [5, 6, 7, 8]  # the cooldown vetoed a 3-bar takeover
    assert fired(out, "R2") == [2, 5, 6, 7, 8, 9]
    assert fired(out, "A1") == []
    assert not out[9].derisk


@RED
@pytest.mark.spec("SPEC§8", "D5")
def test_an_interrupted_challenger_starts_counting_again() -> None:
    # CHOP is active from bar 2 and the cooldown is long over by bar 10. CALM_UP qualifies on
    # bars 10 and 11, CHOP is top again on 12, then CALM_UP on 13, 14, 15: it takes over at 15.
    # A bar where CALM_UP is top but below 0.70 breaks the run the same way.
    head = [*many(CHOP, 10)]
    out = run([*head, *many(CALM, 2), top(CHOP), *many(CALM, 3)])
    assert switches(out) == [2, 15]
    out = run([*head, *many(CALM, 2), top(CALM, 0.69), *many(CALM, 3)])
    assert switches(out) == [2, 15]
    out = run([*head, *many(CALM, 2), top(CALM, 0.70), *many(CALM, 1)])
    assert switches(out) == [2, 12]  # exactly 0.70 qualifies


@RED
@pytest.mark.spec("SPEC§8", "A1")
def test_equal_caps_are_risk_increasing() -> None:
    # Design call 4: only a lower cap is a de-risk. With STRESS capped like CHOP, CHOP -> STRESS
    # needs 3 bars and the cooldown.
    caps = {**CAPS, STRESS: 0.5}
    out = run([*many(CHOP, 3), *many(STRESS, 8)], caps=caps)
    assert switches(out) == [2, 9]
    assert fired(out, "A1") == []


@RED
@pytest.mark.spec("SPEC§8", "C4")
@pytest.mark.parametrize("label", list(Label))
def test_a_cold_start_needs_three_bars_even_for_crash(label: Label) -> None:
    out = run(many(label, 4, 0.9))
    assert actives(out) == [None, None, label, label]
    assert switches(out) == [2]
    assert not any(o.derisk for o in out)
    assert fired(out, "A1") == []


@RED
@pytest.mark.spec("SPEC§8", "C4")
def test_no_label_is_active_without_a_full_takeover() -> None:
    # The top label changes every two bars: nothing ever takes over from cold.
    seq: list[Bar] = []
    for label in [CALM, CHOP, STRESS, CRASH, CALM]:
        seq.extend(many(label, 2))
    out = run(seq)
    assert actives(out) == [None] * len(seq)
    assert not any(o.entries_allowed for o in out)


# --- A1: de-risk after one bar, ignoring the cooldown ---------------------------------------


@RED
@pytest.mark.spec("SPEC§8", "A1", "D5")
def test_an_a1_derisk_switch_after_one_bar_ignores_the_cooldown() -> None:
    # CALM_UP takes over at 2. STRESS (cap 0.25 < 1.0) switches on its first qualifying bar,
    # 3, inside the cooldown; CRASH (0) does the same at 4. A de-risk switch restarts the
    # cooldown, so CALM_UP's way back needs 3 bars and 7 bars after bar 4: bar 11.
    out = run([*many(CALM, 3), top(STRESS, 0.75), top(CRASH, 0.9), *many(CALM, 8)])
    assert actives(out) == [None, None, CALM, STRESS, CRASH, *[CRASH] * 6, CALM, CALM]
    assert switches(out) == [2, 3, 4, 11]
    assert [i for i, o in enumerate(out) if o.derisk] == [3, 4]
    assert fired(out, "A1") == [3, 4]
    assert fired(out, "R3") == [7, 8, 9, 10]


@RED
@pytest.mark.spec("SPEC§8", "A1")
def test_a_derisk_challenger_still_needs_seventy_percent() -> None:
    out = run([*many(CALM, 3), top(STRESS, 0.69), top(STRESS, 0.69), top(STRESS, 0.70)])
    assert actives(out) == [None, None, CALM, CALM, CALM, STRESS]


@RED
@pytest.mark.spec("SPEC§8", "A1")
def test_a_derisk_into_chop_from_calm_up() -> None:
    # CHOP's cap (0.5) is below CALM_UP's (1.0): one bar, no cooldown.
    out = run([*many(CALM, 3), top(CHOP)])
    assert switches(out) == [2, 3]
    assert out[3].derisk


# --- health, rule 6 -------------------------------------------------------------------------


@RED
@pytest.mark.spec("SPEC§8", "C2")
def test_rule_6_acts_on_its_bar_and_recovers_on_the_second_healthy_one() -> None:
    out = run([*many(CALM, 3), SICK, top(CALM), top(CALM), top(CALM)])
    assert actives(out) == [None, None, CALM, CALM, CALM, CALM, CALM]  # the label stays
    assert [o.health.operating for o in out] == [True, True, True, False, False, True, True]
    assert [o.entries_allowed for o in out] == [False, False, True, False, False, True, True]
    assert fired(out, "R6") == [3, 4]
    assert out[3].health.cause == Cause.FEATURES
    assert out[4].health.cause is None
    # Nothing else is evaluated on the unhealthy bar.
    assert (out[3].m_rule4, out[3].m_rule5) == (1.0, 1.0)
    for rule in ("R1", "R2", "R3", "A1", "R4", "R5"):
        assert 3 not in fired(out, rule)


@RED
@pytest.mark.spec("SPEC§8", "D5")
def test_an_unhealthy_bar_resets_the_challenger() -> None:
    head = many(CHOP, 10)
    out = run([*head, *many(CALM, 2), SICK, *many(CALM, 3)])
    assert switches(out) == [2, 15]


@RED
@pytest.mark.spec("SPEC§8", "D5")
def test_the_cooldown_advances_through_unhealthy_bars() -> None:
    # Switch at 2; unhealthy 3-5; CALM_UP qualifies 6, 7, 8 (3 bars at 8, but only 6 bars since
    # the switch); the cooldown expires at 9, unhealthy bars included.
    out = run([*many(CHOP, 3), SICK, SICK, SICK, *many(CALM, 4)])
    assert switches(out) == [2, 9]
    # A recovery bar (healthy, not yet operating) is evaluated in full: an A1 switch can
    # happen on it, while the target stays 0.
    out = run([*many(CALM, 3), SICK, top(STRESS), top(STRESS)])
    assert switches(out) == [2, 4]
    assert out[4].active == STRESS
    assert not out[4].health.operating
    assert out[5].health.operating


# --- rule 5: uncertainty --------------------------------------------------------------------


@RED
@pytest.mark.spec("SPEC§8")
@pytest.mark.parametrize(
    ("dist", "uncertain"),
    [
        # 0.55 - 0.40 is 0.15000000000000002 in float but exactly 0.15 in decimal: it fires.
        ({CALM: 0.55, CHOP: 0.40, STRESS: 0.03, CRASH: 0.02}, True),
        ({CALM: 0.551, CHOP: 0.40, STRESS: 0.03, CRASH: 0.019}, False),
        ({CALM: 0.45, STRESS: 0.30, CHOP: 0.15, CRASH: 0.10}, True),
        ({CALM: 0.5, CHOP: 0.5}, True),  # a model with two labels
        ({CHOP: 1.0}, False),  # one label has no second
    ],
)
def test_rule_5_fires_within_the_gap_inclusive(dist: dict[Label, float], uncertain: bool) -> None:
    out = run([*many(CALM, 3), bar(dist), top(CALM)])
    assert out[3].m_rule5 == (0.0 if uncertain else 1.0)
    assert (3 in fired(out, "R5")) == uncertain
    assert out[4].m_rule5 == 1.0  # the next bar is certain again: no hysteresis


@RED
@pytest.mark.spec("SPEC§8", "C4")
def test_rule_5_fires_without_an_active_label() -> None:
    out = run([bar({CALM: 0.55, CHOP: 0.40, STRESS: 0.03, CRASH: 0.02})])
    assert out[0].active is None
    assert out[0].m_rule5 == 0.0


# --- rule 4: early de-risk ------------------------------------------------------------------


@RED
@pytest.mark.spec("SPEC§8")
@pytest.mark.parametrize(
    ("nxt", "derisk"),
    [
        ({CALM: 0.70, CHOP: 0.05, STRESS: 0.15, CRASH: 0.10}, False),  # exactly 0.25: strict
        ({CALM: 0.699, CHOP: 0.05, STRESS: 0.151, CRASH: 0.10}, True),
        ({CALM: 0.70, CHOP: 0.04, STRESS: 0.26}, True),  # a model without CRASH
        ({CALM: 0.70, CHOP: 0.30}, False),  # neither label in the model
        ({STRESS: 0.2, CRASH: 0.8}, True),
    ],
)
def test_rule_4_halves_the_multiplier_on_its_bar(nxt: dict[Label, float], derisk: bool) -> None:
    out = run([*many(CALM, 3), bar(probs(CALM_UP=0.8), nxt), top(CALM)])
    assert out[3].m_rule4 == (0.5 if derisk else 1.0)
    assert (3 in fired(out, "R4")) == derisk
    assert out[4].m_rule4 == 1.0
    assert out[3].active == CALM  # rule 4 sizes; it doesn't switch


# --- the trace ------------------------------------------------------------------------------


@RED
@pytest.mark.spec("SPEC§8")
def test_every_bar_traces_every_rule_in_order() -> None:
    from quant.decide.switching import RuleId

    out = run([*many(CALM, 3), SICK, top(STRESS, 0.75), bar(probs(CALM_UP=0.5, CHOP=0.45))])
    order = [RuleId.R6, RuleId.R1, RuleId.R2, RuleId.R3, RuleId.A1, RuleId.R5, RuleId.R4]
    for o in out:
        assert [t.rule for t in o.trace] == order
        assert all(t.reason for t in o.trace)
        assert {t.rule for t in o.trace if t.fired} == {r for r in order if o.fired(r)}


# --- invalidation blocks --------------------------------------------------------------------


@RED
@pytest.mark.spec("SPEC§7", "SPEC§8")
def test_an_invalidation_blocks_entries_until_a_fresh_three_bar_reconfirmation() -> None:
    # CALM_UP active at 2, invalidated after bar 4. Bars 5, 6, 7 qualify it again: the block
    # clears on bar 7 (design call 6), whatever the cooldown.
    out = run(many(CALM, 9), invalidate_after=[4])
    assert [o.blocked for o in out] == [False] * 5 + [True, True, False, False]
    assert [o.entries_allowed for o in out] == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    assert switches(out) == [2]


@RED
@pytest.mark.spec("SPEC§7", "SPEC§8")
def test_a_reconfirmation_is_consecutive_and_needs_seventy_percent() -> None:
    seq = [*many(CALM, 5), top(CALM), top(CALM, 0.6), *many(CALM, 3)]
    out = run(seq, invalidate_after=[4])
    assert [o.blocked for o in out][5:] == [True, True, True, True, False]
    seq = [*many(CALM, 5), top(CALM), SICK, *many(CALM, 3)]
    out = run(seq, invalidate_after=[4])
    assert [o.blocked for o in out][5:] == [True, True, True, True, False]


@RED
@pytest.mark.spec("SPEC§7", "SPEC§8", "D5")
def test_the_reconfirmation_ignores_the_cooldown() -> None:
    # Invalidated right after the takeover bar: bars 3, 4, 5 re-confirm, inside the cooldown.
    out = run(many(CALM, 6), invalidate_after=[2])
    assert [o.blocked for o in out] == [False, False, False, True, True, False]


@RED
@pytest.mark.spec("SPEC§7", "SPEC§8", "A1")
def test_a_switch_away_and_a_takeover_back_clears_the_block() -> None:
    # Invalidated after 4; STRESS de-risks at 5 (the block goes with the switch); CALM_UP
    # qualifies from 6, and takes over at 12 (3 bars, and 7 after the switch at 5): unblocked.
    out = run([*many(CALM, 5), top(STRESS), *many(CALM, 8)], invalidate_after=[4])
    assert switches(out) == [2, 5, 12]
    assert not any(o.blocked for o in out[5:])
    assert out[12].entries_allowed


@RED
@pytest.mark.spec("SPEC§7", "SPEC§8")
def test_invalidate_needs_an_active_label() -> None:
    from quant.decide.switching import SwitchState, invalidate

    with pytest.raises(ValueError, match="active"):
        invalidate(SwitchState.cold(SWITCHING))


# --- the state and the inputs ---------------------------------------------------------------


@RED
@pytest.mark.spec("SPEC§8", "C4")
def test_the_cold_state() -> None:
    from quant.decide.switching import SwitchState

    cold = SwitchState.cold(SWITCHING)
    assert (cold.active, cold.challenger, cold.count, cold.since_switch) == (None, None, 0, None)
    assert (cold.health_streak, cold.blocked, cold.reconfirm) == (2, False, 0)


@RED
@pytest.mark.spec("SPEC§8")
def test_step_refuses_bad_inputs() -> None:
    from quant.decide.switching import SwitchState, step

    cold = SwitchState.cold(SWITCHING)
    with pytest.raises(ValueError, match="probs"):
        step(cold, None, cause=None, config=SWITCHING, caps=CAPS)  # healthy without probs
    with pytest.raises(ValueError, match="probs"):
        step(cold, bar({}), cause=None, config=SWITCHING, caps=CAPS)
    partial = {label: cap for label, cap in CAPS.items() if label != CRASH}
    with pytest.raises(ValueError, match="caps"):
        step(cold, top(CALM), cause=None, config=SWITCHING, caps=partial)


@RED
@pytest.mark.spec("SPEC§8", "ARCH§4.3")
def test_the_state_round_trips_through_canonical_json() -> None:
    from quant.core.canonical import canonical_dumps, canonical_loads
    from quant.decide.switching import SwitchState

    out = run([*many(CHOP, 3), *many(CALM, 2)], invalidate_after=[3])
    state = out[-1].state
    again = SwitchState.model_validate_json(canonical_dumps(state))
    assert again == state
    assert canonical_loads(canonical_dumps(again)) == canonical_loads(canonical_dumps(state))
