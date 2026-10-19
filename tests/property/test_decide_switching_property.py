"""B8a: switching properties (spec §8, A1, C4, D5; ARCHITECTURE §13's "de-risking switches ignore
the cooldown").

Random sequences of bars (a top label at one of a few probabilities around the 0.70 threshold,
or an unhealthy bar) from a cold start. The active label after every bar must be the plain
reference's (tests.support.decide.ref_actives, recomputed from the whole history), and every
switch must be one the rules allow: a cold takeover after 3 qualifying bars, a de-risk on its
first qualifying bar whatever the cooldown, or a risk-increasing takeover after 3 qualifying bars
once 7 bars have passed since the last switch. Small example counts keep the fast tier fast.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant.core.regime import Label
from quant.decide.health import Cause
from tests.support.decide import CAPS, SWITCHING, RefBar, probs, ref_actives, top

RED = pytest.mark.xfail(strict=True, reason="red:B8", raises=NotImplementedError)

_bars = st.lists(
    st.one_of(
        st.tuples(st.sampled_from(list(Label)), st.sampled_from([0.5, 0.69, 0.70, 0.8, 0.95])),
        st.just(None),
    ),
    min_size=1,
    max_size=40,
)


def _run(seq: list[tuple[Label, float] | None]) -> list[Any]:
    from quant.decide.switching import SwitchState, step

    state = SwitchState.cold(SWITCHING)
    out = []
    for item in seq:
        if item is None:
            o = step(state, None, cause=Cause.MODEL, config=SWITCHING, caps=CAPS)
        else:
            o = step(state, top(*item), cause=None, config=SWITCHING, caps=CAPS)
        out.append(o)
        state = o.state
    return out


@RED
@pytest.mark.spec("SPEC§8", "A1", "C4", "D5")
@settings(max_examples=30)
@given(_bars)
def test_the_active_label_follows_the_rules(seq: list[tuple[Label, float] | None]) -> None:
    out = _run(seq)
    reference = ref_actives(
        [RefBar(None if item is None else probs(**{item[0].value: item[1]})) for item in seq],
        CAPS,
    )
    assert [o.active for o in out] == reference


@RED
@pytest.mark.spec("SPEC§8", "A1", "D5")
@settings(max_examples=30)
@given(_bars)
def test_every_switch_is_allowed_and_derisks_ignore_the_cooldown(
    seq: list[tuple[Label, float] | None],
) -> None:
    out = _run(seq)
    previous: Label | None = None
    last: int | None = None
    for t, o in enumerate(out):
        if o.switched:
            item = seq[t]
            assert item is not None
            assert item[0] == o.active
            assert item[1] >= 0.70
            assert o.derisk == (previous is not None and CAPS[o.active] < CAPS[previous])
            if not o.derisk:
                assert t >= 2
                run = seq[t - 2 : t + 1]
                assert all(x is not None and x[0] == o.active and x[1] >= 0.70 for x in run)
                if previous is not None:
                    assert last is not None
                    assert t - last >= 7
            previous, last = o.active, t
        else:
            assert o.active == previous
        assert [r.rule.value for r in o.trace] == ["R6", "R1", "R2", "R3", "A1", "R5", "R4"]
        assert o.m_rule5 in (0.0, 1.0)
        assert o.m_rule4 in (0.5, 1.0)
