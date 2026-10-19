"""B8b: EngineState and the DecisionRecord (ARCHITECTURE §4.3, §13, spec §8 "Log", T-NL2;
quant.decide.state, quant.decide.record).

The cold state (C4), the state's canonical JSON round trip and its stable hash; the record's
fields as ARCHITECTURE §4.3 lists them, and `with_verdicts`, which returns a new record with a
new hash and leaves the old one alone. Records come from decide() on the synthetic world
(tests.support.decide.DecideFixture).
"""

from __future__ import annotations

import math

import pytest

from quant.core.canonical import canonical_dumps, sha256_hex
from quant.core.errors import CanonicalJSONError
from quant.core.orders import Side
from quant.core.regime import Label
from quant.decide.primitives import TradeState
from quant.decide.switching import RuleId, SwitchState
from tests.support.decide import DecideFixture, engine_config, r1_caps, r1_release

# --- EngineState -----------------------------------------------------------------------------


@pytest.mark.spec("ARCH§4.3", "C4", "SPEC§8")
def test_the_cold_state_is_flat_with_no_active_label() -> None:
    from quant.decide.edge import EdgeBook
    from quant.decide.state import EngineState

    switching = engine_config().switching
    cold = EngineState.cold(switching)
    assert cold.features is None
    assert cold.filter is None
    assert cold.indicators == {}
    assert cold.shadows == {}
    assert cold.switching == SwitchState.cold(switching)
    assert cold.switching.active is None
    assert cold.trade == TradeState.flat()
    assert cold.edge == EdgeBook.empty()
    seeded = EdgeBook.seeded({Label.CHOP: [0.001, 0.002]})
    assert EngineState.cold(switching, edge=seeded).edge == seeded


@pytest.mark.spec("ARCH§4.3", "ARCH§14")
def test_the_state_round_trips_through_canonical_json() -> None:
    from quant.decide.state import EngineState

    state = DecideFixture.get().warm()
    assert state.features is not None
    assert state.filter is not None
    assert set(state.indicators) == {Label.CALM_UP, Label.CHOP}
    data = state.to_json()
    assert data == canonical_dumps(state)
    back = EngineState.from_json(data)
    assert back == state
    assert back.to_json() == data
    cold = EngineState.cold(engine_config().switching)
    assert EngineState.from_json(cold.to_json()) == cold


@pytest.mark.spec("ARCH§4.3", "ARCH§14")
def test_the_state_hash_is_stable_and_changes_with_the_state() -> None:
    from quant.decide.state import EngineState

    fx = DecideFixture.get()
    state = fx.warm()
    h = state.state_hash()
    assert h == sha256_hex(state) == state.state_hash()
    assert len(h) == 64
    assert h == h.lower()
    assert EngineState.from_json(state.to_json()).state_hash() == h
    moved = state.model_copy(update={"trade": fx.in_trade()})
    assert moved.state_hash() != h
    assert fx.warm().state_hash() == h  # the same decision from the same cold state


@pytest.mark.spec("ARCH§4.3")
def test_from_json_refuses_anything_else() -> None:
    from pydantic import ValidationError

    from quant.decide.state import EngineState

    data = EngineState.cold(engine_config().switching).to_json()
    with pytest.raises(ValidationError):
        EngineState.from_json(data.replace(b'"trade"', b'"trades"'))
    with pytest.raises(ValidationError):
        EngineState.from_json(b"[]")


# --- the DecisionRecord ----------------------------------------------------------------------


def _outcome(fx: DecideFixture) -> object:
    """CHOP active (the top label at the decision), in a trade, with the seeded edge: decide
    sizes a BUY."""
    return fx.decide(fx.warm_state(active=Label.CHOP, trade=fx.in_trade()))


@pytest.mark.spec("ARCH§4.3", "SPEC§8", "SPEC§6", "T-NL2", "ARCH§9.2")
def test_the_record_holds_what_architecture_4_3_lists() -> None:
    fx = DecideFixture.get()
    state = fx.warm_state(active=Label.CHOP, trade=fx.in_trade())
    view = fx.view()
    out = fx.decide(state, view)
    r = out.record
    d = fx.decision
    assert (r.decision_id, r.tau, r.as_of) == (d.id, d.tau, view.as_of)
    assert r.view_digest == view.digest()
    assert len(r.view_digest) == 64
    assert r.release_id == r1_release().release_id
    assert (r.model_hash, r.model_fit_end) == (fx.model.artifact_hash(), fx.model.fit_end)
    assert r.model_fit_end < r.tau  # T-NL2
    assert (r.calibration_hash, r.calibration_as_of) == (None, None)
    assert r.provenance_times() == (fx.model.fit_end,)
    assert r.engine_hash == engine_config().config_hash()
    assert r.caps_hash == r1_caps().caps_hash()
    assert r.code_commit == fx.inputs().code_commit
    assert r.features is not None
    assert len(r.features) == len(r1_release().manifest.features.columns)
    assert r.filter is not None
    assert math.isclose(math.fsum(r.filter.label_probs.values()), 1.0, abs_tol=1e-12)
    assert set(r.filter.label_probs) == set(fx.model.labels.labels)
    assert 0.0 <= r.filter.entropy <= 1.0
    assert math.isfinite(r.filter.loglik)
    assert len(r.filter.state_probs) == fx.model.params.n_states
    assert (r.health.cause, r.health.operating) == (None, True)
    assert r.active is Label.CHOP
    assert not r.switched
    assert [t.rule for t in r.trace] == [rule.value for rule in RuleId]
    assert all(t.reason for t in r.trace)
    assert r.trade_reason == "hold"
    assert r.target.value > 0
    assert r.target.price == fx.bar.close
    assert r.target.sizing is not None
    assert r.target.sizing["label"] == "CHOP"
    assert r.intents == out.intents
    assert len(out.intents) == 1
    assert out.intents[0].side is Side.BUY
    assert r.verdicts == ()
    assert r.state_before == state.state_hash()
    assert r.state_after == out.state.state_hash()
    assert r.record_hash() == sha256_hex(r)


@pytest.mark.spec("ARCH§4.3", "ARCH§4.1")
def test_with_verdicts_returns_a_new_record_with_a_new_hash() -> None:
    fx = DecideFixture.get()
    out = _outcome(fx)
    r = out.record
    before = r.record_hash()
    verdict = {"verdict": "approved", "quantity": out.intents[0].quantity, "flagged": False}
    judged = r.with_verdicts([verdict])
    assert judged is not r
    assert len(judged.verdicts) == 1
    assert judged.verdicts[0]["verdict"] == "approved"
    assert judged.record_hash() != before
    assert r.verdicts == ()
    assert r.record_hash() == before
    assert judged.model_copy(update={"verdicts": ()}) == r
    with pytest.raises(ValueError, match="verdict"):
        r.with_verdicts([])  # one verdict per intent
    with pytest.raises(ValueError, match="verdict"):
        r.with_verdicts([verdict, verdict])
    with pytest.raises(CanonicalJSONError):
        r.with_verdicts([{"verdict": "approved", "notional": math.nan}])


@pytest.mark.spec("ARCH§4.3", "ARCH§14")
def test_the_record_hash_is_the_same_on_every_run() -> None:
    fx = DecideFixture.get()
    first, second = _outcome(fx), _outcome(fx)
    assert first.record.record_hash() == second.record.record_hash()  # type: ignore[attr-defined]
    assert first.state.state_hash() == second.state.state_hash()  # type: ignore[attr-defined]
